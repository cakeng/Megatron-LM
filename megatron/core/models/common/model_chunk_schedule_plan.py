# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import (
    AbstractSchedulePlan,
    NoopScheduleNode,
    get_comm_stream,
    get_comp_stream,
)
from megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build


class ModelChunkState:
    """State shared across a model chunk.

    This class holds state that is shared between different components
    of a model chunk, such as input tensors, parameters, and configuration.
    """

    pass


class LayerChunkType(str, Enum):
    """Logical buckets used to balance compute-heavy and latency-bound ops."""

    COMPUTATION = "computation_bound"
    NON_GEMM = "non_gemm"


@dataclass
class LayerChunk:
    """Simple wrapper that tags a schedule node with its chunk classification."""

    name: str
    node: object
    chunk_type: LayerChunkType

    def forward(self, activations, ctx_supplier):
        ctx = ctx_supplier() if self.chunk_type == LayerChunkType.COMPUTATION else nullcontext()
        with ctx:
            return self.node.forward(activations)

    def backward(self, grad):
        return self.node.backward(grad)

    def backward_dw(self):
        if hasattr(self.node, "backward_dw"):
            self.node.backward_dw()

class TransformerLayerSchedulePlan2:
    """Schedule the executing plan of the nodes in a transformer/mtp layer.

    This class organizes the sub-modules of a transformer/mtp layer into
    computation-bound chunks (QKV projections, attention/FlashAttention,
    out projections, shared expert linear layers, dense/MoE MLP) and
    non-GEMM chunks (pre/post-QKV, post-attention processing, router +
    dispatch/combine collectives, mtp post process nodes).

    layer (TransformerLayerSchedulePlan)
    ├── attn (TransformerLayerNode): attention module
    ├── post_attn (TransformerLayerNode): layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All
    └── mtp_post_process (PostProcessNode): mtp post process

    Note that MTP layer has the same operation and execution order with TransformerLayer regarding
    post_attn, moe_dispatch, mlp, moe_combine, but contains extra operations in attn and
    mtp_post_process:
    * mtp.attn wraps around transformer_layer.attn with extra norm, proj and embedding operations.
    * mtp.mtp_post_process contains output_layer, mtp loss operations, whereas
      transformer_layer.mtp_post_process is empty.
    """

    attn = None
    post_attn = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    mtp_post_process = None

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args={}):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer):
                split a transformer layer into multiple nodes for fine-grained scheduling.
            event (torch.cuda.Event):
                record CUDA event across multiple nodes on different streams for synchronization.
            chunk_state (ModelChunkState): model state shared in the model chunk.
            comp_stream (torch.cuda.Stream): CUDA stream for computation.
            comm_stream (torch.cuda.Stream): CUDA stream for communication.
            extra_args (dict): extra arguments for the layer.

        The event and chunk_state are binded to the TransformerModelChunkSchedulePlan
        and shared across all layers in the model chunk.
        """
        from megatron.core.models.gpt.fine_grained_callables import TransformerLayerState

        # print(f"//// Using TransformerLayerSchedulePlan2 ////")

        self.layer_state = TransformerLayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.use_fine_grained_attn = getattr(
            self.layer, "supports_fine_grained_attn", lambda: False
        )()
        # print (f"//// USE FINE-GRAINED ATTN: {self.use_fine_grained_attn} ////")
        self.event = event
        self.comp_stream = comp_stream
        self.comm_stream = comm_stream
        self.layer_chunks = {}

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn, post_attn, mlp, moe_dispatch and moe_combine, and mtp_post_process.
        """
        from megatron.core.models.gpt.fine_grained_callables import (
            TransformerLayerNode,
            TransformerLayerNode2,
            build_layer_callables,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)
        

        # get flags for latter use
        is_mtp = isinstance(self.layer, MultiTokenPredictionLayer)
        is_moe = (
            isinstance(self.layer.transformer_layer.mlp, MoELayer)
            if is_mtp
            else isinstance(self.layer.mlp, MoELayer)
        )
        enable_deepep = self.layer.config.moe_enable_deepep
        extra_args["enable_deepep"] = enable_deepep
        extra_args["is_moe"] = is_moe
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = is_mtp

        # wrapper to help create TransformerLayerNode
        NodeClass = TransformerLayerNode2 if self.use_fine_grained_attn else TransformerLayerNode

        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            return NodeClass(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=extra_args,
            )

        idx = 0
        attn_chunk_names = []
        if self.use_fine_grained_attn:
            attn_qkv_module = fwd_callables[idx]
            attn_core_module = fwd_callables[idx + 1]
            attn_out_module = fwd_callables[idx + 2]
            idx += 3
            self.attn_qkv = create_node(comp_stream, attn_qkv_module, "attn_qkv")
            self.attn_core = create_node(comp_stream, attn_core_module, "attn_core")
            self.attn_out = create_node(comp_stream, attn_out_module, "attn_out")
            attn_chunk_names = ["attn_qkv", "attn_core", "attn_out"]
            self.attn = self.attn_out
        else:
            attn_module = fwd_callables[idx]
            idx += 1
            self.attn = create_node(comp_stream, attn_module, "attn")
            attn_chunk_names = ["attn"]

        (
            post_attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables[idx : idx + 5]

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.post_attn = create_node(comp_stream, post_attn_module, "post_attn")
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.post_attn = NoopScheduleNode()
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        if is_mtp:
            self.mtp_post_process = create_node(
                comp_stream, mtp_post_process_module, "mtp_post_process"
            )
        else:
            self.mtp_post_process = NoopScheduleNode()
        for name in attn_chunk_names:
            self._register_chunk(name, getattr(self, name), LayerChunkType.COMPUTATION)
        if self.use_fine_grained_attn:
            self._register_chunk("attn", self.attn, LayerChunkType.COMPUTATION)

        self._register_chunk("mlp", self.mlp, LayerChunkType.COMPUTATION)
        self._register_chunk("post_attn", self.post_attn, LayerChunkType.NON_GEMM)
        self._register_chunk("moe_dispatch", self.moe_dispatch, LayerChunkType.NON_GEMM)
        self._register_chunk("moe_combine", self.moe_combine, LayerChunkType.NON_GEMM)
        self._register_chunk("mtp_post_process", self.mtp_post_process, LayerChunkType.NON_GEMM)

        self.attn_forward_order = attn_chunk_names
        self.attn_backward_order = list(reversed(attn_chunk_names))
        if self.use_fine_grained_attn:
            self.attn_backward_dw_chunks = ["attn_out", "attn_qkv"]
        else:
            self.attn_backward_dw_chunks = ["attn"]

    def _register_chunk(self, name, node, chunk_type):
        self.layer_chunks[name] = LayerChunk(name=name, node=node, chunk_type=chunk_type)

    def _run_forward_chunk(self, name, activations):
        chunk = self.layer_chunks.get(name)
        if chunk is None:
            return activations
        return chunk.forward(activations, self.get_fp8_context)

    def _run_backward_chunk(self, name, grad):
        chunk = self.layer_chunks.get(name)
        if chunk is None:
            return grad
        return chunk.backward(grad)

    def _run_backward_dw_chunk(self, name):
        chunk = self.layer_chunks.get(name)
        if chunk is None:
            return
        chunk.backward_dw()

    def get_fp8_context(self):
        """
        Get the fp8 context for the transformer layer.
        """
        use_inner_fp8_context = (
            self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        )
        return (
            get_fp8_context(self.layer.config, self.layer.layer_number - 1)
            if use_inner_fp8_context
            else nullcontext()
        )

    @staticmethod
    def run(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations, overlapping the communications
        (dispatch or combine) of one with the computations (att or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd            | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: attn_fwd->post_attn_fwd| mlp_bwd->mlp_bwd_dw->mlp_fwd| post_attn_bwd->attn_bwd
        For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
        and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation

        Note:
            After changing the chunk partition, run a representative 1f1b
            pipeline step (e.g., `primus.examples.megatron.train`) to ensure
            comm/comp overlap and loss parity remain intact.
        """

        if b_layer is not None:
            b_grad = b_layer._run_backward_chunk("mtp_post_process", b_grad)
            b_grad = b_layer._run_backward_chunk("moe_combine", b_grad)

        if f_layer is not None:
            for chunk_name in f_layer.attn_forward_order:
                f_input = f_layer._run_forward_chunk(chunk_name, f_input)
            f_input = f_layer._run_forward_chunk("post_attn", f_input)

        if b_layer is not None:
            b_grad = b_layer._run_backward_chunk("mlp", b_grad)

        if f_layer is not None:
            f_input = f_layer._run_forward_chunk("moe_dispatch", f_input)

        if b_layer is not None:
            b_layer._run_backward_dw_chunk("mlp")
            b_grad = b_layer._run_backward_chunk("moe_dispatch", b_grad)

        if f_layer is not None:
            f_input = f_layer._run_forward_chunk("mlp", f_input)

        if f_layer is not None:
            f_input = f_layer._run_forward_chunk("moe_combine", f_input)
            f_input = f_layer._run_forward_chunk("mtp_post_process", f_input)

        if b_layer is not None:
            b_grad = b_layer._run_backward_chunk("post_attn", b_grad)
            for chunk_name in b_layer.attn_backward_order:
                b_grad = b_layer._run_backward_chunk(chunk_name, b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_layer is not None and not is_last_layer_in_bwd:
            for chunk_name in b_layer.attn_backward_dw_chunks:
                b_layer._run_backward_dw_chunk(chunk_name)

        return f_input, b_grad

class TransformerLayerSchedulePlan:
    """Schedule the executing plan of the nodes in a transformer/mtp layer.

    This class organizes the sub-modules of a transformer/mtp layer,
    including attention, post attention, MLP, dispatch, combine and
    mtp post process nodes.

    layer (TransformerLayerSchedulePlan)
    ├── attn (TransformerLayerNode): attention module
    ├── post_attn (TransformerLayerNode): layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All
    └── mtp_post_process (PostProcessNode): mtp post process

    Note that MTP layer has the same operation and execution order with TransformerLayer regarding
    post_attn, moe_dispatch, mlp, moe_combine, but contains extra operations in attn and
    mtp_post_process:
    * mtp.attn wraps around transformer_layer.attn with extra norm, proj and embedding operations.
    * mtp.mtp_post_process contains output_layer, mtp loss operations, whereas
      transformer_layer.mtp_post_process is empty.
    """

    attn = None
    post_attn = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    mtp_post_process = None

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args={}):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer):
                split a transformer layer into multiple nodes for fine-grained scheduling.
            event (torch.cuda.Event):
                record CUDA event across multiple nodes on different streams for synchronization.
            chunk_state (ModelChunkState): model state shared in the model chunk.
            comp_stream (torch.cuda.Stream): CUDA stream for computation.
            comm_stream (torch.cuda.Stream): CUDA stream for communication.
            extra_args (dict): extra arguments for the layer.

        The event and chunk_state are binded to the TransformerModelChunkSchedulePlan
        and shared across all layers in the model chunk.
        """
        from megatron.core.models.gpt.fine_grained_callables import TransformerLayerState

        self.layer_state = TransformerLayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.event = event
        self.comp_stream = comp_stream
        self.comm_stream = comm_stream

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn, post_attn, mlp, moe_dispatch and moe_combine, and mtp_post_process.
        """
        from megatron.core.models.gpt.fine_grained_callables import (
            TransformerLayerNode,
            build_layer_callables,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)

        # get flags for latter use
        is_mtp = isinstance(self.layer, MultiTokenPredictionLayer)
        is_moe = (
            isinstance(self.layer.transformer_layer.mlp, MoELayer)
            if is_mtp
            else isinstance(self.layer.mlp, MoELayer)
        )
        enable_deepep = self.layer.config.moe_enable_deepep
        extra_args["enable_deepep"] = enable_deepep
        extra_args["is_moe"] = is_moe
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = is_mtp

        # wrapper to help create TransformerLayerNode
        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            return TransformerLayerNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=extra_args,
            )

        (
            attn_module,
            post_attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn = create_node(comp_stream, attn_module, "attn")
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.post_attn = create_node(comp_stream, post_attn_module, "post_attn")
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.post_attn = NoopScheduleNode()
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        if is_mtp:
            self.mtp_post_process = create_node(
                comp_stream, mtp_post_process_module, "mtp_post_process"
            )
        else:
            self.mtp_post_process = NoopScheduleNode()

    def get_fp8_context(self):
        """
        Get the fp8 context for the transformer layer.
        """
        use_inner_fp8_context = (
            self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        )
        return (
            get_fp8_context(self.layer.config, self.layer.layer_number - 1)
            if use_inner_fp8_context
            else nullcontext()
        )

    @staticmethod
    def run(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations, overlapping the communications
        (dispatch or combine) of one with the computations (att or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd            | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: attn_fwd->post_attn_fwd| mlp_bwd->mlp_bwd_dw->mlp_fwd| post_attn_bwd->attn_bwd
        For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
        and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)
                f_input = f_layer.post_attn.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)
                f_input = f_layer.mtp_post_process.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.post_attn.backward(b_grad)
            b_grad = b_layer.attn.backward(b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_layer is not None and not is_last_layer_in_bwd:
            b_layer.attn.backward_dw()

        return f_input, b_grad


class TransformerModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule the executing plan of the sub-modules in a model chunk sub-modules.

    This class organizes the computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing.

    TransformerModelChunkSchedulePlan
    ├── pre_process: PreProcessNode
    ├── layers: List[TransformerLayerSchedulePlan]
    │   ├── layer[0]: TransformerLayerSchedulePlan
    │   ├── layer[1]: TransformerLayerSchedulePlan
    │   └── ...
    └── post_process: PostProcessNode
    """

    def __init__(
        self,
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
    ):
        """Initialize the schedule plan of all Transformer layers' sub-modules.

        This function creates a schedule plan for a model chunk, including
        preprocessing, transformer layers, and postprocessing.

        Args:
            model: The model to build a schedule plan for.
            input_ids: Input token IDs.
            position_ids: Position IDs.
            attention_mask: Attention mask.
            decoder_input: Decoder input tensor.
            labels: Labels for loss computation.
            packed_seq_params: Parameters for packed sequences.
            extra_block_kwargs: Additional keyword arguments for blocks.
            runtime_gather_output: Whether to gather output at runtime.
            loss_mask (torch.Tensor): Used to mask out some portions of the loss

        Returns:
            The model chunk schedule plan.
        """
        from megatron.core.models.gpt.fine_grained_callables import PostProcessNode, PreProcessNode

        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self.pre_process = None
        self.post_process = None
        self.vp_stage = model.vp_stage
        self.use_fine_grained_attn = getattr(model.config, "fine_grained_attn", False)

        comp_stream = get_comp_stream()
        comm_stream = get_comm_stream()

        # save the inputs of model.forward() to ModelChunkState
        self._model_chunk_state.input_ids = input_ids
        self._model_chunk_state.position_ids = position_ids
        self._model_chunk_state.attention_mask = attention_mask
        self._model_chunk_state.decoder_input = decoder_input
        self._model_chunk_state.labels = labels
        self._model_chunk_state.mtp_hidden_states = None
        self._model_chunk_state.loss_mask = loss_mask
        self._model_chunk_state.packed_seq_params = packed_seq_params
        self._model_chunk_state.extra_block_kwargs = extra_block_kwargs
        self._model_chunk_state.runtime_gather_output = runtime_gather_output
        self._model_chunk_state.model = model
        self._model_chunk_state.context = None
        self._model_chunk_state.context_mask = None
        self._model_chunk_state.attention_bias = None

        transformer_num_layers = model.decoder.num_layers_per_pipeline_rank
        mtp_num_layers = get_mtp_num_layers_to_build(model.config, vp_stage=self.vp_stage)

        # build preprocess
        self.pre_process = PreProcessNode(model, self._model_chunk_state, self._event, comp_stream)
        # build layer schedule plan for each layer
        for layer_idx in range(transformer_num_layers):
            layer = model.decoder._get_layer(layer_idx)
            layer_plan_cls = (
                TransformerLayerSchedulePlan2
                if self.use_fine_grained_attn
                else TransformerLayerSchedulePlan
            )
            layer_plan = layer_plan_cls(
                layer, self._event, self._model_chunk_state, comp_stream, comm_stream
            )
            self._transformer_layers.append(layer_plan)

        # build mtp layers
        for layer_idx in range(mtp_num_layers):
            extra_args = {
                "is_first_layer": layer_idx == 0,
                "is_last_layer": layer_idx == mtp_num_layers - 1,
            }
            layer = model.mtp.layers[layer_idx]
            layer_plan = TransformerLayerSchedulePlan(
                layer, self.event, self.state, comp_stream, comm_stream, extra_args
            )
            self._transformer_layers.append(layer_plan)

        # build post process
        if model.post_process:
            self.post_process = PostProcessNode(
                model, self._model_chunk_state, self._event, comp_stream
            )

    @property
    def event(self):
        """Gets the CUDA event for synchronization."""
        return self._event

    def record_current_stream(self):
        """Records the current CUDA stream in the event."""
        stream = torch.cuda.current_stream()
        self.event.record(stream)

    def wait_current_stream(self):
        """Waits for the event to complete on the current CUDA stream."""
        stream = torch.cuda.current_stream()
        self.event.wait(stream)

    def get_layer(self, i):
        """Gets the transformer layer at the specified index."""
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def num_layers(self):
        """Gets the number of transformer layers."""
        return len(self._transformer_layers)

    @property
    def state(self):
        """Gets the model chunk state."""
        return self._model_chunk_state

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        self.pre_process = None

        if self.post_process is not None:
            self.post_process.model_chunk_state = None
            self.post_process = None

    @staticmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        This function schedules the forward and backward passes for a model chunk,
        which interleaves forward and backward function of multiple Transformer layers
        within a model chunk, and this is needed to overlap the submodules between the individual
        forward and backward functions.

        Assume there are 4 layers in the given model chunk:
        Phase 0: p2p_comm_sync -> forward_preprocess -> p2p_comm_sync -> backward_postprocess
        Phase 1: forward_layer[0] + backward_layer[3], overlapped execution by schedule_layer_1f1b
        Phase 2: forward_layer[1] + backward_layer[2], overlapped execution by schedule_layer_1f1b
        Phase 3: forward_layer[2] + backward_layer[1], overlapped execution by schedule_layer_1f1b
        Phase 4: forward_layer[3] + backward_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: send_forward_recv_backward -> send_backward_recv_forward
        Phase 6: backward_dw of the first layer -> forward_postprocess -> backward_preprocess

        Args:
            f_schedule_plan (TransformerModelChunkSchedulePlan): The forward schedule plan
            b_schedule_plan (TransformerModelChunkSchedulePlan): The backward schedule plan
            b_grad (Tensor or None): The gradient of the loss function
            pre_forward (callable or None): The function to call before the forward pass
            pre_backward (callable or None): The function to call before the backward pass
            post_forward (callable or None): The function to call after the forward pass
            post_backward (callable or None): The function to call after the backward pass
        Returns:
            The output of the forward pass.
        """
        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                pre_forward(f_schedule_plan.vp_stage)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            if b_schedule_plan.post_process is not None:
                b_grad = b_schedule_plan.post_process.backward(b_grad)

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlapped_layers = min(f_num_layers, b_num_layers)

        use_fine_grained_attn = (
            getattr(f_schedule_plan, "use_fine_grained_attn", False)
            or getattr(b_schedule_plan, "use_fine_grained_attn", False)
        )
        run_layer_plan = (
            TransformerLayerSchedulePlan2.run
            if use_fine_grained_attn
            else TransformerLayerSchedulePlan.run
        )

        # combined forward and backward pass for overlapped layers
        for i in range(overlapped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
            f_input, b_grad = run_layer_plan(
                f_layer,
                b_layer,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
            )
            torch.cuda.nvtx.range_pop()

        # backward pass for the remaining layers
        for i in range(overlapped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            _, b_grad = run_layer_plan(
                None, b_layer, b_grad=b_grad, is_last_layer_in_bwd=(i == b_num_layers - 1)
            )
            torch.cuda.nvtx.range_pop()

        # forward pass for the remaining layers
        for i in range(overlapped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, _ = run_layer_plan(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and post_forward is not None:
            # post_forward()/send_forward_recv_forward() is running in the communication stream,
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of attn backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, b_schedule_plan.vp_stage)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_num_layers > 0:
            b_schedule_plan.get_layer(0).attn.backward_dw()

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()

        # Release reference as early as possible, this helps avoid memory leak.
        if b_schedule_plan is not None:
            b_schedule_plan.release_state()

        return f_input
