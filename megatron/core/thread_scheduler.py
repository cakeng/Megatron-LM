# parallel_training_scheduler.py
#
# A thread-based scheduler for data-parallel training with fine-grained
# concurrency control for backward passes.
#
# Key Features:
# 1. Selective Deep-Copy: Efficiently shares model weights (nn.Parameter)
#    across threads while isolating other states.
# 2. Cooperative Multitasking: Forward pass hooks at pre-defined points
#    allow threads to yield, enabling concurrent execution.
# 3. Backward Pass Mutual Exclusion: A lock is assigned to each
#    transformer layer. Threads must acquire a layer's lock before
#    executing its backward pass, preventing gradient race conditions.
# 4. Optimizer Synchronization: A threading.Barrier ensures the optimizer
#    step only occurs after all threads have finished their backward pass.
#
# All code formatted within 80 columns as requested [2025-05-21]

from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from sortedcontainers import SortedList
import faulthandler
import time
import traceback
import sys
from dataclasses import dataclass, field
from typing import Dict, Callable, Tuple
import signal
from torch.autograd import Function
import torch.distributed as dist

from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging
logging.set_verbosity_error() # Suppress verbose warnings


def g_str(s): # green
    return "\033[32m" + s + "\033[0m"
def r_str(s): # red
    return "\033[31m" + s + "\033[0m"
def b_str(s): # blue
    return "\033[34m" + s + "\033[0m"
def y_str(s): # yellow
    return "\033[33m" + s + "\033[0m"
def m_str(s): # magenta
    return "\033[35m" + s + "\033[0m"
def c_str(s): # cyan
    return "\033[36m" + s + "\033[0m"
def w_str(s): # white
    return "\033[37m" + s + "\033[0m"
def k_str(s): # black
    return "\033[30m" + s + "\033[0m"

# Enable automatic stack dumps on SIGQUIT
faulthandler.enable()

class _PassThrough(Function):
    @staticmethod
    def forward(ctx, x):
        # No alloc, no math; just forwards the same storage
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Identity gradient
        return grad_output

class ContextSwitchModule(nn.Module):
    def __init__(self, force_switch: bool = False, 
                 do_fwd: bool = True, do_bwd: bool = True):
        super().__init__()
        self.force_switch = force_switch
        self.do_fwd = do_fwd
        self.do_bwd = do_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensures a grad_fn exists even if this would be a pure identity
        # return x
        return _PassThrough.apply(x)

class SyncEventModule(nn.Module):
    def __init__(self, do_fwd: bool = True, do_bwd: bool = True):
        super().__init__()
        self.do_fwd = do_fwd
        self.do_bwd = do_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensures a grad_fn exists even if this would be a pure identity
        # return x
        return _PassThrough.apply(x)

def dump_all_threads(signum, frame):
    print("\n" + "="*80)
    print("DETAILED THREAD DUMP (Ctrl+C detected)")
    print("="*80)
    
    # Get all threads
    threads = threading.enumerate()
    current_thread = threading.current_thread()
    
    print(f"Total threads: {len(threads)}")
    print(f"Current thread: {current_thread.name} (ID: {current_thread.ident})")
    print("-" * 80)
    
    # Dump each thread's stack trace
    for thread in threads:
        print(f"\nThread: {thread.name} (ID: {thread.ident})")
        print(f"  Alive: {thread.is_alive()}")
        print(f"  Daemon: {thread.daemon}")
        
        # Get the frame for this thread
        frame = sys._current_frames().get(thread.ident)
        if frame:
            print("  Stack trace:")
            stack = traceback.extract_stack(frame)
            for filename, lineno, name, line in stack[-10:]:  # Last 10 frames
                print(f"    File '{filename}', line {lineno}, in {name}")
                if line:
                    print(f"      {line}")
        else:
            print("  No frame available")
        print("-" * 40)
    
    print("="*80)
    
    # Also dump using faulthandler for complete info
    print("FAULTHANDLER DUMP:")
    faulthandler.dump_traceback()
    print("="*80)

# Register handlers
signal.signal(signal.SIGINT, dump_all_threads)   # Ctrl+C
signal.signal(signal.SIGUSR1, dump_all_threads)  # kill -USR1 <pid>
signal.signal(signal.SIGQUIT, dump_all_threads)  # Ctrl+\

# =============================================================================
# COMPONENT 1: SELECTIVE DEEP-COPY
# (This component remains unchanged)
# =============================================================================

def _get_parent_module_and_param_name(model: nn.Module, path: str):
    """
    A helper function to find the parent module and the final attribute
    name given a full parameter path.
    Example: for path 'layers.0.attn.weight', it returns the
    `model.layers[0].attn` module and the string 'weight'.
    """
    parts = path.split('.')
    parent_module = model
    for part in parts[:-1]:
        # getattr can handle both attribute access (like .layers) and
        # indexed access for nn.ModuleList (like [0])
        if part.isdigit():
            # Try integer indexing first (for ModuleList), then string key (for ModuleDict)
            try:
                parent_module = parent_module[int(part)]
            except (KeyError, TypeError):
                # If integer indexing fails, try string key (for ModuleDict)
                parent_module = parent_module[part]
        else:
            parent_module = getattr(parent_module, part)
    param_name = parts[-1]
    return parent_module, param_name

def materialize_meta_model(
    meta_model: nn.Module, 
    base_model: nn.Module
):
    """
    Materializes a model created on the 'meta' device by replacing its
    meta tensors with references to the corresponding tensors from a
    fully initialized base model.

    Args:
        meta_model (nn.Module): The model skeleton, initialized on the
                                'meta' device. This model will be
                                modified in-place.
        base_model (nn.Module): A fully initialized model (on CPU or GPU)
                                that contains the actual tensor data.
    """
    base_params: Dict[str, nn.Parameter] = dict(base_model.named_parameters())
    base_buffers: Dict[str, torch.Tensor] = dict(base_model.named_buffers())

    # ### FIX: Collect parameter names into a static list first ###
    # This prevents the "dictionary changed during iteration" error.
    meta_param_names = [name for name, _ in meta_model.named_parameters()]

    for param_name in meta_param_names:
        # We only need to handle meta tensors. If a tensor is already
        # materialized, we can skip it.
        parent_check, attr_check = _get_parent_module_and_param_name(
            meta_model, param_name
        )
        if getattr(parent_check, attr_check).device != torch.device("meta"):
            continue
            
        if param_name not in base_params:
            raise ValueError(
                f"Architecture mismatch: Parameter '{param_name}' found in "
                "meta_model but not in base_model."
            )

        parent_module, attr_name = _get_parent_module_and_param_name(
            meta_model, param_name
        )

        # Replace the meta parameter with a reference to the base parameter
        delattr(parent_module, attr_name)
        setattr(parent_module, attr_name, base_params[param_name])

    # ### FIX: Apply the same logic for buffers ###
    meta_buffer_names = [name for name, _ in meta_model.named_buffers()]

    for buffer_name in meta_buffer_names:
        parent_check, attr_check = _get_parent_module_and_param_name(
            meta_model, buffer_name
        )
        if getattr(parent_check, attr_check).device != torch.device("meta"):
            continue

        if buffer_name not in base_buffers:
            raise ValueError(
                f"Architecture mismatch: Buffer '{buffer_name}' found in "
                "meta_model but not in base_model."
            )

        parent_module, attr_name = _get_parent_module_and_param_name(
            meta_model, buffer_name
        )
        
        delattr(parent_module, attr_name)
        setattr(parent_module, attr_name, base_buffers[buffer_name])

# =============================================================================
# COMPONENT 2: SCHEDULER, THREAD, AND HOOKS (UPDATED FOR TRAINING)
# =============================================================================
@dataclass
class ExecContext:
    exec: threading.Thread
    signal: threading.Event
    event: torch.cuda.Event
    exec_stream: torch.Stream
    fwd_stream: torch.Stream
    bwd_stream: torch.Stream
    event_done: bool
    serialized_regions: list[threading.Event]
    ready_info: dict[str, any]
    comm_works_wait: list[dist.Work] = field(default_factory=list)
    comm_works_dispatch: list[dist.Work] = field(default_factory=list)
    pair_exec_id: int = None
    exec_iter: int = 0
    
class ContextScheduler:
    """ Manages and schedules ExecutionEngines in a round-robin fashion. """
    def __init__(self, device, model, num_execs, is_dist = False, num_serialized_regions = 6,
                 profiler = None, debug = False):
        self.device = device
        self.model = model
        self.num_execs = num_execs
        self.active_exec_id = None
        self.next_exec_id = None
        self.stop_scheduling = False
        self.waiting_exec_ids = SortedList()
        self.hooks = {} 
        self.execs = {}
        self.first_non_main_exec_id = None
        self.ident_to_exec_id = {}
        self.backward_exec_id = None
        self.profiler = profiler
        self.debug = debug
        self.skip_debug_num = 2
        self.is_dist = is_dist
        self.comm_order_key = 0
        self.completion_barrier = threading.Barrier(num_execs)
        self.completion_signal_m2c = threading.Event()
        self.completion_signal_m2c.clear()
        self.completion_signal_c2m = threading.Event()
        self.completion_signal_c2m.clear()
        self.context_lock = threading.Lock()
        self.context_lock.acquire()
        self.num_serialized_regions = num_serialized_regions
        self.thread_barriers = {}
        
    def format_print(self, message, color_fnc=b_str, ident=None):
        if ident is None:
            ident = threading.current_thread().ident
        prefix = f"T{str(ident)[-6:]}"
        if self.is_dist:
            prefix += f" R{dist.get_rank()}"
        prefix += f" C{self.active_exec_id} CK{self.comm_order_key}"
        prefix = color_fnc(prefix) if color_fnc is not None else prefix
        return f"[{prefix}] {message}"

    def _scheduler(self, force_switch = False, skip_debug=False):
        if self.stop_scheduling:
            self.next_exec_id = None
            return
        if self.debug and not skip_debug:
            ready_infos = []
            for exec_id, exec_context in self.execs.items():
                ready_infos.append(f"{exec_id}: {exec_context.ready_info}")
            print(self.format_print("Current execs: " + 
                    y_str(f"{ready_infos}"), r_str))
        if len(self.waiting_exec_ids) > 0:
            self.next_exec_id = None
            run_schedule_loop = True
            while run_schedule_loop:
                for exec_id in self.waiting_exec_ids:
                    exec_context = self.execs[exec_id]
                    if exec_context.ready_info["key"] == "CONTEXT_SWITCH_HOOK":
                        pair_exec_id = exec_context.pair_exec_id
                        pair_exec = self.execs[pair_exec_id] if pair_exec_id is not None else None
                        if pair_exec is None or pair_exec.exec_iter == -1:
                            self.next_exec_id = exec_id
                            run_schedule_loop = False
                            break
                        elif pair_exec.exec_iter >= exec_context.ready_info["pair_exec_iter"] + 1:
                            if pair_exec.event_done or pair_exec.event.query():
                                pair_exec.event_done = True
                                self.next_exec_id = exec_id
                                run_schedule_loop = False
                                break

                        # if exec_context.event_done or exec_context.event.query():
                        #     exec_context.event_done = True
                        #     if exec_context.pair_exec_id is None:
                        #         self.next_exec_id = exec_id
                        #         run_schedule_loop = False
                        #         break
                        #     elif self.execs[exec_context.pair_exec_id].exec_iter  ==  -1 or \
                        #         self.execs[exec_context.pair_exec_id].exec_iter >= exec_context.ready_info["pair_exec_iter"] + 1:
                        #         self.next_exec_id = exec_id
                        #         run_schedule_loop = False
                        #         break
                    elif exec_context.ready_info["key"] == "WAIT_COMMS":
                        if self._check_comm(exec_id):
                            self.next_exec_id = exec_id
                            run_schedule_loop = False
                            break
                    elif exec_context.ready_info["key"] == "PAIR_THREAD_BARRIER":
                        if exec_context.ready_info["num_threads"] >= self.thread_barriers[exec_context.ready_info["pair_key"]][0] and \
                            exec_context.ready_info["pass_id"] >= self.thread_barriers[exec_context.ready_info["pair_key"]][1]:
                            self.next_exec_id = exec_id
                            run_schedule_loop = False
                            break
                    elif exec_context.ready_info["key"] == "SERIALIZED_REGION":
                        if exec_context.serialized_regions[exec_context.ready_info["region_id"]].is_set():
                            self.next_exec_id = exec_id
                            run_schedule_loop = False
                            break
                    elif exec_context.ready_info["key"] == "ATTACHED_TO_CONTEXT":
                        self.next_exec_id = exec_id
                        run_schedule_loop = False
                        break
                    else:
                        raise ValueError(f"Unexpected ready info: {exec_context.ready_info}")
                
                if self.next_exec_id is None and not force_switch:
                    if self.active_exec_id is None:
                        self.next_exec_id = self.waiting_exec_ids[0]
                    else:
                        self.next_exec_id = self.active_exec_id
                    run_schedule_loop = False
            
            if self.debug and not skip_debug:
                print(self.format_print("Current waiting execs: " + 
                      y_str(f"{self.waiting_exec_ids}") + ", Scheduling next exec " + 
                      y_str(f"{self.next_exec_id}"), r_str))
        else: 
            # No execs waiting
            if force_switch:
                self.next_exec_id = None # No active exec
                if self.debug and not skip_debug:
                    print(self.format_print("No execs waiting, no active exec, setting next exec to NONE. " +
                          "Scheduler will be deactivated.", r_str))
            else:
                if self.debug and not skip_debug:
                    print(self.format_print(" No execs waiting, scheduling active exec " + 
                        y_str(f"{self.active_exec_id}"), r_str))
                self.next_exec_id = self.active_exec_id

    def _set_and_get_exec_stream(self, exec_id):
        output_stream = None
        exec_obj = self.execs[exec_id]
        output_stream = exec_obj.exec_stream
        # if self.backward_exec_id == exec_id:
        #     if exec_obj.pair_exec_id is not None:
        #         pair_exec_id = exec_obj.pair_exec_id
        #         if self.execs[pair_exec_id].exec_iter != -1:
        #             output_stream = exec_obj.bwd_stream
        # else:
        #     if self.execs[exec_id].pair_exec_id is not None:
        #         pair_exec_id = self.execs[exec_id].pair_exec_id
        #         if self.execs[pair_exec_id].exec_iter != -1:
        #             output_stream = exec_obj.fwd_stream
        torch.cuda.set_stream(output_stream)
        return output_stream

    def _release_context(self, exec_id, skip_debug=False):
        # Release the context lock and signal the next exec to resume.
        assert self.active_exec_id == exec_id, \
            f"Expected {self.active_exec_id} to be the active exec during release, " + \
            f"got {exec_id} running"
        if self.debug and not skip_debug:
            print(self.format_print("Yielding context of exec " + 
                  y_str(f"{self.active_exec_id}") + ", ready key " + 
                  y_str(f"{self.execs[self.active_exec_id].ready_info}") + ". " +
                  "Next scheduled exec " + 
                  y_str(f"{self.next_exec_id}") + "\n", b_str), end="")
        self.execs[self.active_exec_id].signal.clear()
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.active_exec_id = None
        self.context_lock.release()

    def _acquire_context(self, exec_id, skip_debug=False):
        # Acquire the context lock and execute.
        # Append to the front of the list to maintain order
        self.waiting_exec_ids.add(exec_id)
        if self.active_exec_id is None and self.next_exec_id is None:
            self.next_exec_id = exec_id
            if self.debug and not skip_debug:
                print(self.format_print("No active execs, acquiring context lock", b_str))
        else:
            if self.debug and not skip_debug:
                print(self.format_print(f"Waiting for context switch of exec " + y_str(f"{exec_id}"), b_str))
            self.execs[exec_id].signal.wait()
        self.context_lock.acquire() 
        if exec_id in self.waiting_exec_ids:
            self.waiting_exec_ids.remove(exec_id)
        self.active_exec_id = exec_id
        self._set_and_get_exec_stream(exec_id)
        if self.debug and not skip_debug:
            print(self.format_print(f"Resuming exec {exec_id} context"), b_str)

    def add_exec(self, exec, ident = None):
        # Add an exec to the scheduler.\
        assert exec.device == self.device, \
            f"Expected {self.device} device for exec, got {exec.device}"
        if ident is None:
            ident = exec.ident
        for exec_id in self.execs:
            assert self.execs[exec_id].exec.ident != ident, \
                f"Exec {exec} already exists, ident {ident} - ident_to_exec_id: {self.ident_to_exec_id}"
        new_exec_id = len(self.execs)
        assert new_exec_id < self.num_execs, \
            f"Expected {self.num_execs} execs, got {new_exec_id}"
        if self.first_non_main_exec_id is None and not exec.main_thread:
            self.first_non_main_exec_id = new_exec_id
        new_exec_signal = threading.Event()
        new_exec_signal.clear()
        dev = exec.device
        new_exec_stream = torch.cuda.Stream(dev)
        torch.cuda.set_stream(new_exec_stream)
        new_exec_event = torch.cuda.Event()
        new_exec_event.record()
        new_exec_event_done = True
        new_exec_serialized_regions = []
        new_exec_fwd_stream, new_exec_bwd_stream = green_ctx_stream_create(dev)
        new_exec_ready_info = None
        new_exec_comm_works_wait = []
        new_exec_comm_works_dispatch = []
        new_exec_pair_exec_id = None
        new_exec_exec_iter = 0
        for i in range (self.num_serialized_regions):
            new_exec_serialized_regions.append(threading.Event())
            new_exec_serialized_regions[i].clear()
        self.ident_to_exec_id[ident] = new_exec_id
        self.execs[new_exec_id] = ExecContext(exec, 
                                              new_exec_signal, 
                                              new_exec_event,
                                              new_exec_stream,
                                              new_exec_fwd_stream,
                                              new_exec_bwd_stream,
                                              new_exec_event_done,
                                              new_exec_serialized_regions,
                                              new_exec_ready_info,
                                              new_exec_comm_works_wait,
                                              new_exec_comm_works_dispatch,
                                              new_exec_pair_exec_id,
                                              new_exec_exec_iter)
        if self.debug:
            print(self.format_print(f"Added exec {ident} with id " + 
                  y_str(f"{new_exec_id}"), y_str))
        return new_exec_id
    
    def context_switch(self, exec_id, force_switch=False, skip_debug=False):
        # Schedule the next exec, release context of the current exec,
        # and enter the waiting queue until next context is acquired.
        self._scheduler(force_switch=force_switch, skip_debug=skip_debug)
        if self.next_exec_id != self.active_exec_id:
            self._release_context(exec_id, skip_debug=skip_debug)
            self._acquire_context(exec_id, skip_debug=skip_debug)
        elif self.debug and not skip_debug:
            print(self.format_print("No context switch needed", b_str))
        return

    def attach_exec_to_context(self, exec_id):
        # Attach the exec to the scheduler context
        # i.e., enter the waiting queue for next exec
        if self.debug:
            print(self.format_print(f"Attaching exec " + y_str(f"{exec_id}"), b_str))
        self.execs[exec_id].ready_info = {"key": "ATTACHED_TO_CONTEXT"}
        self._acquire_context(exec_id)
        return

    def detach_exec_from_context(self, exec_id):
        # Detach the exec from the scheduler context
        # i.e., release context without entering the waiting queue for context switch
        assert self.active_exec_id == exec_id, \
            f"Expected {self.active_exec_id} to be the active exec during detach, " + \
            f"got {exec_id} running"
        self.execs[exec_id].ready_info = {"key": "DETACHED_FROM_CONTEXT"}
        self._scheduler(force_switch=True)
        self._release_context(exec_id)
        if self.debug:
            print(self.format_print(f"Exec " + y_str(f"{exec_id}") + 
                                    " detached and running independently from the scheduler context.", b_str))
        return

    def pair_thread_barrier(self, exec_id, pair_id, pair_key, num_threads = 2, pass_id=0):
        if pair_key not in self.thread_barriers:
            self.thread_barriers[pair_key] = [1, 0]
        else:
            self.thread_barriers[pair_key][0] += 1
        self.execs[exec_id].ready_info = {"key": "PAIR_THREAD_BARRIER", 
                                          "pair_id": pair_id,
                                          "pair_key": pair_key,
                                          "num_threads": num_threads,
                                          "pass_id": pass_id}
        self.execs[exec_id].pair_exec_id = pair_id
        if self.debug:
            print(self.format_print("Exec " + y_str(f"{exec_id} ") + 
                                    g_str(f"Entering pair thread barrier ") + 
                                    y_str(f"pair id ") + f"{pair_id}, " +
                                    y_str(f"pair key ") + f"{pair_key}, " +
                                    f"{self.thread_barriers[pair_key][1]} / {pass_id} pass" +
                                    f", {self.thread_barriers[pair_key][0]} / {num_threads} threads"), b_str)
        loop_count = 0
        while self.thread_barriers[pair_key][0] < num_threads or \
            self.thread_barriers[pair_key][1] < pass_id:
            self.context_switch(exec_id, skip_debug=loop_count > self.skip_debug_num)
            loop_count += 1
        self.thread_barriers[pair_key][1] += 1
        self.execs[exec_id].ready_info = None
        if self.debug:
            print(self.format_print("Exec " + y_str(f"{exec_id} ") + 
                                    g_str(f"Passed pair thread barrier ") + 
                                    y_str(f"pair id ") + f"{pair_id}, " +
                                    y_str(f"pair key ") + f"{pair_key}, " +
                                    f"{self.thread_barriers[pair_key][1]} / {pass_id} pass" +
                                    f", {self.thread_barriers[pair_key][0]} / {num_threads} threads"), b_str)
        return
    
    def enter_serialized_region(self, exec_id, region_id=0, region_name=""):
        # Enters a serialized region of code in the exec_id order.
        self.execs[exec_id].ready_info = {"key": "SERIALIZED_REGION", "region_id": region_id, "region_name": region_name}
        loop_count = 0
        while not self.execs[exec_id].serialized_regions[region_id].is_set():
            self.context_switch(exec_id, skip_debug=loop_count > self.skip_debug_num)
            loop_count += 1
        self.execs[exec_id].ready_info = None
        if self.debug:
            print(self.format_print("Exec " + y_str(f"{exec_id} ") + g_str(f"Entering") + 
                                    " serialized region " + y_str(f"{region_id}, {region_name}"), b_str))
        if region_name == "Backward":
            self.backward_exec_id = exec_id
            self.execs[exec_id].exec_iter = 0
        return
    
    def exit_serialized_region(self, exec_id, region_id=0, region_name=""):
        # Exits a serialized region of code in the exec_id order.
        if region_name == "Backward":
            self.backward_exec_id = None
        self.execs[exec_id].serialized_regions[region_id].clear()
        if exec_id + 1 < self.num_execs:
            # Signal the next exec to enter the serialized region
            self.execs[exec_id + 1].serialized_regions[region_id].set()
        if self.debug:
            print(self.format_print("Exec " + y_str(f"{exec_id} ") + r_str(f"Exiting") + 
                                    " serialized region " + y_str(f"{region_id}, {region_name}"), b_str))
        return

    def schedule_comm(self, exec_id, comm_func, comm_key, wait_for_completion):
        # Wait for all comms ops to complete
        if self.debug:
            print(self.format_print(f"Waiting for comms ops to complete, comm_key: {comm_key}", r_str))
        self.execs[exec_id].ready_info = {"key": "WAIT_COMMS", 
                                          "comm_func": comm_func, 
                                          "comm_key": comm_key, 
                                          "wait_for_completion": wait_for_completion,
                                          "fired": False}
        loop_count = 0
        while not self._check_comm(exec_id):
            self.context_switch(exec_id, skip_debug=loop_count > self.skip_debug_num)
            loop_count += 1
        self.execs[exec_id].ready_info = None
        if self.debug:
            print(self.format_print(g_str(f"Comms ops completed: ") + 
                                    f"comm_key: {comm_key}", r_str))
        return
    
    def _check_comm(self, exec_id):  
        if self.comm_order_key == self.execs[exec_id].ready_info["comm_key"]:
            if not self.execs[exec_id].ready_info["wait_for_completion"]:
                self.execs[exec_id].comm_works_dispatch.extend(self.execs[exec_id].ready_info["comm_func"]())
            else:
                self.execs[exec_id].comm_works_wait.extend(self.execs[exec_id].ready_info["comm_func"]())
            self.execs[exec_id].ready_info["fired"] = True
            self.comm_order_key += 1
            if self.debug:
                print(self.format_print(y_str(f"Fired comm with comm_key: ") + 
                                        f"{self.execs[exec_id].ready_info['comm_key']}, " +
                                        f"wait: {self.execs[exec_id].ready_info['wait_for_completion']}", r_str))
        if self.execs[exec_id].ready_info["fired"] and not self.execs[exec_id].ready_info["wait_for_completion"]:
            return True
        if self.execs[exec_id].ready_info["fired"]:
            for work in self.execs[exec_id].comm_works_wait:
                if not work.is_completed():
                    return False
            self.execs[exec_id].comm_works_wait = []
            return True
            
        return False

    def record_event(self, exec_id):
        self.execs[exec_id].event_done = False
        self.execs[exec_id].event.record()
        return

    def sync_comm_works(self, exec_id):
        for work in self.execs[exec_id].comm_works_wait:
            work.wait()
        self.execs[exec_id].comm_works_wait = []
        for work in self.execs[exec_id].comm_works_dispatch:
            work.wait()
        self.execs[exec_id].comm_works_dispatch = []
        return

    def increment_comm_order_key(self):
        self.comm_order_key += 1
        return

    def reset(self):
        for exec_id in self.execs:
            self.execs[exec_id].signal.clear()
        for i in range (self.num_serialized_regions):
            for j in range (self.num_execs):
                if j == 0:
                    self.execs[j].serialized_regions[i].set()
                else:
                    self.execs[j].serialized_regions[i].clear()
        self.active_exec_id = None
        self.stop_scheduling = False
        self.comm_order_key = 0
        self.thread_barriers = {}
        self.completion_signal_c2m.clear()
        self.completion_signal_m2c.clear()
        for exec_id in self.execs:
            self.execs[exec_id].exec.release_event.set()
        return
    
    def start(self, reset=True):
        assert len(self.execs) == self.num_execs, \
            f"Expected {self.num_execs} execs, got {len(self.execs)}"
        if reset:
            self.reset()
        if self.next_exec_id is None:
            self._scheduler(force_switch=True)
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.context_lock.release()
        return
        
    def stop(self):
        self.stop_scheduling = True
        self.context_lock.acquire() # Acquire the context lock
        self.stop_scheduling = False

    def wait_completion(self):
        if self.num_execs == 1:
            if self.debug:
                print(self.format_print("Single exec! Resuming main thread."))
            self.context_lock.acquire()
            return
        if self.debug:
            print(y_str(f"[Main Thread]") + " Waiting for completion of all execs.")
        self.completion_signal_c2m.wait() # Wait for all execs to complete
        self.context_lock.acquire() # Acquire the context lock
        self.completion_signal_m2c.set() # Signal that the main thread has acquired the context lock
        if self.debug:
            print(y_str(f"[Main Thread]") + " All execs completed! Resuming main thread.")

    def tag_module_name(self, module, module_name=None):
        # Set the exec_name for the current module
        if module_name is None:
            module.module_name = module.__class__.__name__
        else:
            module.module_name = module_name    
        
        for child_name, child_module in module.named_children():
            self.tag_module_name(child_module, module.module_name + "_" + child_name)

    def attach_hooks(self):
        """ ### IMPLEMENTATION: Attach hooks to all modules. ### """
        print(self.format_print(" Attaching hooks...", y_str))
        self.tag_module_name(self.model)
        self.detach_hooks() # Clear any old hooks first
        for module in self.model.modules():
            module_name = module.module_name
            context_switch_module_list = ["context_switch_module"]
            sync_event_module_list = ["sync_event_module"]
            self.hooks[module_name] = []
            if any(module_class_str in module_name for module_class_str in context_switch_module_list):
                fwd_hook = module.register_forward_hook(self.forward_scheduler_hook)
                bwd_hook = module.register_full_backward_hook(self.backward_scheduler_hook)
                self.hooks[module_name].append(fwd_hook)
                self.hooks[module_name].append(bwd_hook)
                if self.debug:
                    print(self.format_print("Attached forward and backward context switch hooks to " + 
                          y_str(f"{module_name}") + "\n", g_str), end="")
            elif self.profiler is not None and module_name in self.profiler.modules:
                module_info = self.profiler.modules[module_name]
                if module_info.fire_context_switch:
                    fwd_hook = module.register_forward_hook(self.forward_scheduler_hook)
                    try:
                        bwd_hook = module.register_full_backward_hook(self.backward_scheduler_hook)
                        print(f"Successfully registered backward hook {bwd_hook} for {module_name}")
                    except Exception as e:
                        print(f"Failed to register backward hook for {module_name}: {e}")
                        bwd_hook = None
                    self.hooks[module_name] = (
                        fwd_hook,
                        bwd_hook
                    )
                    if self.debug:
                        print(self.format_print("Attached forward and backward profiler context switch hooks to " + 
                              y_str(f"{module_name}") + "\n", g_str), end="")
            elif any(module_class_str in module_name for module_class_str in sync_event_module_list):
                fwd_hook = module.register_forward_hook(self.forward_record_event_hook)
                bwd_hook = module.register_full_backward_hook(self.backward_record_event_hook)
                self.hooks[module_name].append(fwd_hook)
                self.hooks[module_name].append(bwd_hook)
                if self.debug:
                    print(self.format_print("Attached forward and backward sync event hooks to " + 
                          y_str(f"{module_name}") + "\n", g_str), end="")
            if len(self.hooks[module_name]) == 0:
                self.hooks.pop(module_name)
                              
        print(self.format_print("Hooks: " + y_str(f"{self.hooks}")))
    
    def detach_hooks(self):
        """ ### IMPLEMENTATION: Remove all attached hooks. ### """
        # Clear all events
        for fwd_hook, bwd_hook in self.hooks.values():
            fwd_hook.remove()
            if bwd_hook is not None:
                bwd_hook.remove()
        self.hooks = {}

    def signal_exec_iter_completion(self, exec_id):
        self.execs[exec_id].exec_iter = -1

    def forward_record_event_hook(self, module, input, output):
        if not module.do_fwd:
            return
        exec_id = self.ident_to_exec_id[threading.current_thread().ident]
        self.record_event(exec_id)
        return
        
    def backward_record_event_hook(self, module, grad_input, grad_output):
        if not module.do_bwd:
            return
        exec_id = self.backward_exec_id
        self.record_event(exec_id)
        return
        
    def forward_scheduler_hook(self, module, input, output):
        exec_id = self.ident_to_exec_id[threading.current_thread().ident]
        force_switch = False
        if isinstance(module, ContextSwitchModule):
            pair_exec_id = self.execs[exec_id].pair_exec_id
            force_switch = module.force_switch
            if pair_exec_id is not None and self.execs[pair_exec_id].exec_iter != -1:
                force_switch = True  
            if not module.do_fwd:
                return
        self.execs[exec_id].exec_iter += 1
        exec_iter = self.execs[exec_id].exec_iter
        if self.execs[exec_id].pair_exec_id is None:
            pair_exec_iter = None
        else:
            pair_exec_iter = self.execs[self.execs[exec_id].pair_exec_id].exec_iter
        self.execs[exec_id].ready_info = {"key": "CONTEXT_SWITCH_HOOK",
                                          "exec_iter": exec_iter,
                                          "pair_exec_id": self.execs[exec_id].pair_exec_id,
                                          "pair_exec_iter": pair_exec_iter,
                                          "force_switch": force_switch}
        if self.debug:
            print(self.format_print(b_str(f"Forward hook {exec_id}") + " fired on " + 
                y_str(f"{module.module_name}") + " Ready info: " + 
                y_str(f"{self.execs[exec_id].ready_info}") + "\n", g_str), end="")
        self.context_switch(exec_id, force_switch=force_switch)
        if self.debug:
            print(self.format_print(b_str(f"Forward hook {exec_id}") + " completed", g_str))

    def backward_scheduler_hook(self, module, grad_input, grad_output):
        exec_id = self.backward_exec_id
        if self.debug:
            print(self.format_print(r_str(f"Backward hook {exec_id}") + " fired on " + 
                    y_str(f"{module.module_name}") + "\n", g_str, 
                    self.backward_exec_id), end="")
        force_switch = False
        if isinstance(module, ContextSwitchModule):
            pair_exec_id = self.execs[exec_id].pair_exec_id
            force_switch = module.force_switch
            if pair_exec_id is not None and self.execs[pair_exec_id].exec_iter != -1:
                force_switch = True  
            if not module.do_bwd:
                return
        self.execs[exec_id].exec_iter += 1
        exec_iter = self.execs[exec_id].exec_iter
        if self.execs[exec_id].pair_exec_id is None:
            pair_exec_iter = None
        else:
            pair_exec_iter = self.execs[self.execs[exec_id].pair_exec_id].exec_iter
        self.execs[exec_id].ready_info = {"key": "CONTEXT_SWITCH_HOOK",
                                          "exec_iter": exec_iter,
                                          "pair_exec_id": self.execs[exec_id].pair_exec_id,
                                          "pair_exec_iter": pair_exec_iter}
        self.context_switch(exec_id, force_switch=force_switch)
        if self.debug:
            print(self.format_print(r_str(f"Backward hook {exec_id}") + " completed", g_str))

            
@dataclass
class ModuleInfo():
    avg_time_taken: float
    num_profiles: int
    input_shape: torch.Size
    output_shape: torch.Size
    fire_context_switch: bool
    time_since_last_context_switch: float
    
    new_module_queue_start: int
    module_queue_start: int
    module_queue_end: int
    profiler_event: torch.cuda.Event
    
    fwd_pre_hook: Callable
    fwd_hook: Callable
    bwd_hook: Callable
    
def print_module_info(module_name, module_info, module_queue = []):
    out_str = (g_str(f"\t[{module_name}]") + " Module info:\n" + 
               y_str(f"\t\tavg time taken:") + f"{module_info.avg_time_taken}\n" +
               y_str(f"\t\tnum profiles: ") + f"{module_info.num_profiles}\n" +
               y_str(f"\t\tinput shape: ") + f"{module_info.input_shape}\n" +
               y_str(f"\t\toutput shape: ") + f"{module_info.output_shape}\n" +
               y_str(f"\t\tfire context switch: ") + f"{module_info.fire_context_switch}\n" +
               y_str(f"\t\ttime since last context switch: ") + f"{module_info.time_since_last_context_switch}\n"
               )
    print(out_str, end="")
    
class ModelProfiler():
    def __init__(
        self, model, microbatch, label, loss_fn, do_profile=True, debug = False
    ):
        super().__init__()
        self.model = model
        self.microbatch = microbatch
        self.label = label
        self.loss_fn = loss_fn
        self.loss = None
        self.total_time = 0
        self.modules = {}
        self.module_queue = []
        self.debug = debug
        
        self.tag_module_name(self.model)
        if do_profile:
            self.profile_model()
            self.delete_non_leaf_modules()
            self.flag_context_switch()
            self.print_profiler_info()
    
    def tag_module_name(self, module, module_name=None):
        if module_name is None:
            module.module_name = module.__class__.__name__
        else:
            module.module_name = module_name    
        
        for child_name, child_module in module.named_children():
            self.tag_module_name(child_module, 
                                 module.module_name + "_" + child_name)
            
    def delete_non_leaf_modules(self):
        if self.debug:
            print(g_str(f"[Profiler]") + " Deleting non-leaf modules...")
        new_modules = {}
        new_module_queue = []
        total_time = 0
        for module_name in self.module_queue:
            module_info = self.modules[module_name]
            if module_info.module_queue_end - module_info.module_queue_start == 1:
                if self.debug:
                    print(g_str(f"[Profiler]") + " Keeping leaf module " + 
                          b_str(f"{module_name}") + ".")
                module_info.module_queue_start = len(new_module_queue)
                module_info.module_queue_end = len(new_module_queue) + 1
                new_modules[module_name] = module_info
                new_module_queue.append(module_name)
                total_time += module_info.avg_time_taken
            elif self.debug:
                print(g_str(f"[Profiler]") + " Deleting non-leaf module " + 
                      r_str(f"{module_name}") + ".")
        self.modules = new_modules
        self.module_queue = new_module_queue
        self.total_time = total_time
        
    def flag_context_switch(self):
        # Simple equal-time based context switch
        num_context_switch_points = len(self.module_queue) / 500 if len(self.module_queue) > 2500 else 5
        context_switch_time = self.total_time / num_context_switch_points
        current_time = 0    
        for module_name in self.module_queue:
            module_info = self.modules[module_name]
            module_info.time_since_last_context_switch = current_time
            if current_time + module_info.avg_time_taken > context_switch_time:
                if self.debug:
                    print(g_str(f"[Profiler]") + " Flagging context switch at " + 
                          y_str(f"{module_name}") + ".")
                module_info.fire_context_switch = True
                current_time = 0
            current_time += module_info.avg_time_taken
            
    def print_profiler_info(self):
        print(g_str(f"[Profiler]") + " Profiler info:")
        print(y_str(f"\tModel: ") + f"{self.model}")
        print(y_str(f"\tTotal time: ") + f"{self.total_time}")
        print(y_str(f"\tProfiled Modules: "))
        num_context_switch = 0
        for module_name in self.modules:
            if self.modules[module_name].fire_context_switch:
                num_context_switch += 1
                print_module_info(module_name, self.modules[module_name], self.module_queue)
        print(y_str(f"\tNumber of context switches: ") + f"{num_context_switch}")
        # print(y_str(f"\tProfiled Module queue: ") + f"{self.module_queue}")

    def profile_model(self):
        self.new_module_queue = []
        self.model_changed = False
        self.attach_hooks()
        outputs = self.model(self.microbatch)
        self.detach_hooks()
        self.module_queue = self.new_module_queue
        return outputs, self.model_changed

    def attach_hooks(self):
        print(g_str(f"[Profiler]") + " Attaching hooks...")
        self.detach_hooks() # Clear any old hooks first
        for module in self.model.modules():
            module_info = ModuleInfo(
                avg_time_taken=-1,
                time_since_last_context_switch=-1,
                input_shape=torch.Size([]),
                output_shape=torch.Size([]),
                fire_context_switch=False,
                new_module_queue_start=-1,
                module_queue_start=-1,
                module_queue_end=-1,
                num_profiles=0,
                profiler_event=torch.cuda.Event(enable_timing=True),
                fwd_pre_hook=None,
                fwd_hook=None,
                bwd_hook=None
            )
            module_info.fwd_pre_hook = module.register_forward_pre_hook(self.forward_pre_profiler_hook)
            module_info.fwd_hook = module.register_forward_hook(self.forward_profiler_hook)
            self.modules[module.module_name] = module_info
    
    def detach_hooks(self):
        for module_info in self.modules.values():
            if module_info.fwd_pre_hook is not None:
                module_info.fwd_pre_hook.remove()
            if module_info.fwd_hook is not None:
                module_info.fwd_hook.remove()
            if module_info.bwd_hook is not None:
                module_info.bwd_hook.remove()
        
    def forward_pre_profiler_hook(self, module, input):
        if isinstance(input, tuple):
            if len(input) > 0:
                input_shape = input[0].shape if hasattr(input[0], 'shape') else torch.Size([])
            else:
                input_shape = torch.Size([])
        else:
            input_shape = input.shape if hasattr(input, 'shape') else torch.Size([])
            
        if self.debug:
            print(g_str(f"[Profiler]") + " Forward pre profiler hook fired on " + 
                  y_str(f"{module.module_name}") + f", input {input_shape}")
        module_key = module.module_name
        self.modules[module_key].new_module_queue_start = len(self.new_module_queue)
        self.new_module_queue.append(module_key)
        self.modules[module_key].profiler_event.record()

    def forward_profiler_hook(self, module, input, output):
        if isinstance(input, tuple):
            if len(input) > 0:
                input_shape = input[0].shape if hasattr(input[0], 'shape') else torch.Size([])
            else:
                input_shape = torch.Size([])
        else:
            input_shape = input.shape if hasattr(input, 'shape') else torch.Size([])
        if isinstance(output, tuple):
            if len(output) > 0:
                output_shape = output[0].shape if hasattr(output[0], 'shape') else torch.Size([])
            else:
                output_shape = torch.Size([])
        else:
            output_shape = output.shape if hasattr(output, 'shape') else torch.Size([])
        if self.debug:
            print(g_str(f"[Profiler]") + " Forward profiler hook fired on " + 
                  y_str(f"{module.module_name}") + f", input {input_shape}" + 
                  f", output {output_shape}")

        module_key = module.module_name
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        time_taken = self.modules[module_key].profiler_event.elapsed_time(end_event)
        self.total_time += time_taken
        self.modules[module_key].input_shape = input_shape
        self.modules[module_key].output_shape = output_shape
        
        # If this is not the first profile, check if the module execution changed
        if self.modules[module_key].module_queue_start > -1:
            # If this is not the first profile and the model did not change yet, check if the module queue is the same
            old_start = self.modules[module_key].module_queue_start
            if old_start != self.modules[module_key].new_module_queue_start:
                self.model_changed = True
            old_end = self.modules[module_key].module_queue_end
            if old_end != len(self.new_module_queue):
                self.model_changed = True
            old_module_queue = self.module_queue[old_start:old_end]
            new_module_queue = self.new_module_queue[self.modules[module_key].new_module_queue_start:]
            if new_module_queue != old_module_queue:
                self.model_changed = True
            if self.model_changed and self.debug:
                print(g_str(f"[Profiler]") + " Model execution change detected on " + 
                        y_str(f"{module.module_name}"))
                
        if self.model_changed or self.modules[module_key].module_queue_start == -1:
            # If the module queue is different (execution changed), update the module queue start and end
            self.modules[module_key].avg_time_taken = time_taken
            self.modules[module_key].module_queue_start = self.modules[module_key].new_module_queue_start
            self.modules[module_key].module_queue_end = len(self.new_module_queue)
        else:
            # If the module queue is the same (execution did not change), update the average time taken
            self.modules[module_key].avg_time_taken = \
                (self.modules[module_key].avg_time_taken * self.modules[module_key].num_profiles + 
                    time_taken) / (self.modules[module_key].num_profiles + 1)
        self.modules[module_key].num_profiles += 1
        if self.debug:
            print_module_info(module_key, self.modules[module_key], 
                              self.new_module_queue)

class ExecutionEngine(threading.Thread):
    
    """ A dedicated thread to run a single training step (fwd/bwd). """
    def __init__(
        self, x, label, loss_fn, scheduler, start_exec = True, 
        is_dist = False, device = None, debug = False, 
        main_thread = False,
    ):
        super().__init__(daemon=not main_thread)
        self.x = x
        self.label = label
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.loss = None
        self.exec_id = None
        
        self.release_event = threading.Event()
        self.release_event.clear()
        self.backward_tid = -1
        self.debug = debug
        self.device = device
        self.is_dist = is_dist        
        self.main_thread = main_thread
        if self.main_thread:
            ident = threading.current_thread().ident
            self.exec_id = self.scheduler.add_exec(self, ident)
        if start_exec:
            self.start()
     
    def start_exec(self):
        self.start()
        while self.exec_id is None:
            time.sleep(0.01)
        return self.exec_id

    def format_print(self, message, color_fnc=b_str, tid=None):
        if self.exec_id is None:
            raise ValueError(r_str("ExecutionEngine not initialized. Please call init_exec() first."))
        prefix = f"T{str(threading.current_thread().ident)[-6:]}"
        if self.is_dist:
            prefix += f" R{dist.get_rank()}"
        prefix += f" E{self.exec_id} CK{self.scheduler.comm_order_key}"
        prefix = color_fnc(prefix) if color_fnc is not None else prefix
        return f"[{prefix}] {message}"

    def wait_release(self):
        self.release_event.wait()
        self.release_event.clear()
        return

    def run(self):
        self.exec_id = self.scheduler.add_exec(self)
        print(self.format_print(f"Running exec {self.exec_id}..."))
        while True:
            self.wait_release()
            self.step()
            self.sync_step_completion()
            
    def sync_step_completion(self):
        # Wait for all execs to complete.
        if self.debug:
            print(self.format_print("Waiting for all execs to complete."))
        self.scheduler.completion_barrier.wait() # Wait for all execs to complete
        if self.scheduler.num_execs == 1:
            if self.debug:
                print(self.format_print("Single exec! Waiting for next iteration..."))
            return
        if self.exec_id == self.scheduler.first_non_main_exec_id:
            self.scheduler.completion_signal_c2m.set() # Signal that the main thread can proceed to acquire the context lock
            if self.debug:
                print(self.format_print("All execs completed! Signaling main thread to acquire context lock."))
            self.scheduler.completion_signal_m2c.wait() # Wait for the main thread to acquire the context lock
            if self.debug:
                print(self.format_print("All execs completed! Waiting for next iteration..."))
            
    def step(self):
        # Attach the exec to the scheduler context, 
        # The execution of this thread will be controlled by the scheduler.
        self.scheduler.attach_exec_to_context(self.exec_id)

        # Forward pass
        self.scheduler.model.train()
        outputs = self.scheduler.model(self.x, labels=self.label)
        self.loss = self.loss_fn(outputs, self.label)
        
        # Detach the exec from the scheduler context,
        # The execution will now be independent of the scheduler and
        # the scheduler will no longer switch to this thread.
        
        print(self.format_print(f"Fwd pass finished. Loss: {self.loss}"))
        
        self.scheduler.enter_serialized_region(self.exec_id, region_name="Backward")
        
        # Backward pass
        self.loss.backward()
        
        self.scheduler.exit_serialized_region(self.exec_id, region_name="Backward")
        self.scheduler.detach_exec_from_context(self.exec_id)
        
        print(self.format_print("Bwd pass finished"))

# =============================================================================
# E2E TEST AND MODEL DEFINITION
# =============================================================================

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_ff=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + self.dropout(attn_out)
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + self.dropout(ffn_out)
        return x

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_layers=2, d_model=128, nhead=4, d_ff=512, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, nhead, d_ff) 
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, labels=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def simple_loss_fn(outputs, label):
    # Reshape for CrossEntropyLoss
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), label.view(-1))

def run_training_test(
    model_name, model_factory_fn, get_data_fn, loss_fn, num_threads, debug
):
    print("\n" + "="*80)
    print(f" E2E Training Test: {model_name} with {num_threads} Threads")
    print("="*80)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available.")
        return

    # 1. Model and Optimizer Setup
    base_model = model_factory_fn(materialized=True)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    
    # 2. Data and Thread Setup
    microbatches, label = get_data_fn(num_threads)
    
    # 3. Profiler Setup
    profiler = ModelProfiler(base_model, microbatches[0], label[0], loss_fn)

    # 3. Hook and Lock Setup
    scheduler = ContextScheduler(base_model.device, num_threads, debug)
    for t in range(num_threads):
        model = model_factory_fn(materialized=False)
        materialize_meta_model(model, base_model)
        ExecutionEngine(model, microbatches[t], label[t], 
                        loss_fn, scheduler, profiler=profiler, debug=debug)

    # 4. Run Training Step
    scheduler.start()
    scheduler.wait_completion() # Main thread waits here
    
    # print(y_str(f"[Main Thread]") + " All threads complete. Stepping optimizer...")
    # optimizer.step()
    # optimizer.zero_grad()
    
    # final_weight = transformer_layers[0].attn.in_proj_weight.data

    # # 5. Cleanup and Verification
    # for t in threads: t.join(timeout=5)
    # scheduler.stop()

    # assert not torch.equal(initial_weight, final_weight), \
    #     "Weights did not change after optimizer step."
    # print("\nVerification PASSED: Model weights were updated.")

# =============================================================================
# TEST CONFIGURATIONS AND MAIN BLOCK
# =============================================================================

def get_simple_transformer_model(materialized=True):
    if materialized:
        return SimpleTransformerModel().to("cuda")
    else:
        with init_empty_weights():
            return SimpleTransformerModel()

def get_simple_transformer_data(num_threads):
    batch_size, seq_len, vocab_size = 32, 64, 100
    assert batch_size % num_threads == 0
    micro_bs = batch_size // num_threads
    
    full_batch = torch.randint(
        0, vocab_size, (batch_size, seq_len), device='cuda'
    )
    full_label = torch.randint(
        0, vocab_size, (batch_size, seq_len), device='cuda'
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    label = list(torch.split(full_label, micro_bs))
    return microbatches, label

def get_deepseek_model(materialized=True):
    """
    Factory function for the DeepSeekV2-Lite model.

    Args:
        materialized (bool): If True, loads the full model with weights
            and device_map. If False, creates a memoryless shell on the
            'meta' device.
    """
    model_name = "deepseek-ai/deepseek-v2-lite"
    
    if materialized:
        # This is for the base_model: load it completely with accelerate
        print(y_str(f"[Main Thread]") + " Loading materialized base model with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        print(y_str(f"[Main Thread]") + " Base model loaded.")
        return model
    else:
        # This is for the meta_model_shell: create a shell from config
        # without loading weights.
        print(y_str(f"[Main Thread]") + " Creating meta model shell...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print(y_str(f"[Main Thread]") + " Meta model shell created.")
        return model

def get_deepseek_data(num_threads):
    # This is a dummy data generator for demonstration
    batch_size, seq_len = 16, 64
    assert batch_size % num_threads == 0
    micro_bs = batch_size // num_threads
    
    full_batch = torch.randint(
        0, 1000, (batch_size, seq_len), device='cuda', dtype=torch.long
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    microbatch_shapes = [microbatch.shape for microbatch in microbatches]
    print(y_str(f"[Main Thread]") + " Microbatches: " + f"{microbatch_shapes}")
    # For causal LM, label are the same as inputs
    return microbatches, microbatches

def deepseek_loss_fn(outputs, label):
    # The model itself returns a loss object if label are provided
    return outputs.loss

if __name__ == "__main__":
    # Test 1: Simple Transformer
    debug = True
    for num_threads in [1, 2, 4]:
        run_training_test(
            "Simple Transformer",
            get_simple_transformer_model,
            get_simple_transformer_data,
            simple_loss_fn,
            num_threads,
            debug
        )
        torch.cuda.empty_cache()
    # NOTE: Running with more threads on large models is very
    # memory-intensive due to activations stored for backward pass.
    for num_threads in [4]:
        run_training_test(
            "DeepSeekV2-Lite",
            get_deepseek_model,
            get_deepseek_data,
            deepseek_loss_fn,
            num_threads,
            debug
        )
        torch.cuda.empty_cache()