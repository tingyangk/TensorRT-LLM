import bisect
import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeAlias

import torch

from tensorrt_llm.llmapi.llm_args import (BaseSparseAttentionConfig,
                                          DecodingBaseConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...inputs.multimodal import MultimodalParams
from ..distributed import MPIDist
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import get_memory_buffers
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.eagle3 import Eagle3ResourceManager
from ..speculative.mtp import SampleStateTensorsMTP
from ..utils import make_weak_ref, piecewise_cuda_graph
from .llm_request import LlmRequest, SamplingConfig, get_draft_token_length
from tensorrt_llm.sampling_params import SamplingParams
from .resource_manager import (BaseResourceManager, ResourceManager,
                               ResourceManagerType)
from ..attention_backend.trtllm import TrtllmAttentionMetadata
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1
KeyType: TypeAlias = Tuple[int, int, bool, bool]


@dataclass
class EncoderCUDAGraphRunnerConfig:
    """
    Configuration for the EncoderCUDAGraphRunner.
    
    Encoder-only models (like BERT) need different graph keying than decoders
    since they process variable-length sequences without autoregressive generation.
    
    Attributes:
        use_cuda_graph: Master switch for CUDA graph execution. When False, forces
            eager mode. When True, the runner checks eligibility and uses graphs
            when possible, falling back to eager mode when not.
        cuda_graph_padding_enabled: When True, batches/sequences are padded to the
            nearest supported size. When False, only exact matches use graphs.
            Disabling this avoids padding overhead but may result in more eager fallbacks.
        cuda_graph_batch_sizes: List of batch sizes to capture graphs for.
        cuda_graph_num_tokens: List of total number of tokens to capture graphs for.
        cuda_graph_max_seq_len: List of maximum sequence length per request to capture graphs for.
        max_cuda_graph_batch_size: Maximum batch size for CUDA graphs.
        max_cuda_graph_num_tokens: Maximum total number of tokens for CUDA graphs.
        max_cuda_graph_max_seq_len: Maximum maximum sequence length per request for CUDA graphs.
        max_num_tokens: Maximum total tokens (batch_size * seq_len limit).
        cuda_graph_mem_pool: Shared CUDA memory pool for graph capture.
        use_token_type_ids: Whether to allocate static buffer for token_type_ids.
            Only needed for models like BERT that use segment embeddings.
            Set to False for mm_encoder_only models to avoid memory overhead.
    """
    use_cuda_graph: bool
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    cuda_graph_num_tokens: list[int]
    cuda_graph_seq_lens: list[int]
    max_cuda_graph_batch_size: int
    max_cuda_graph_num_tokens: int
    max_num_tokens: int
    cuda_graph_mem_pool: Any
    use_token_type_ids: bool
    enable_attention_dp: bool
    mapping: Optional[Mapping]
    dist: Optional[MPIDist]


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner, passed from the ModelEngine."""
    use_cuda_graph: bool
    """
    Master switch controlling the model's execution path.

    This flag determines one of three distinct execution paths for the
    model engine:

    1.  **`False` (Pure Eager Path):**
        * Forces all execution to be in eager mode.
        * The `CUDAGraphRunner` instance is mostly dormant
        * Methods like `maybe_get_cuda_graph` and `pad_batch`
            will return immediately, signaling the model engine to
            run in eager mode.

    2.  **`True` (Eager Fallback Path):**
        * The runner is active and checks for graph eligibility.
        * If a batch is ineligible (e.g., it's a prefill batch,
            stats collection is on, or it's an unsupported batch size),
            the runner signals a fallback to eager mode for that batch.

    3.  **`True` (CUDA Graph Path):**
        * The runner finds an eligible batch and a matching graph.
        * The graph is then captured (if new) or replayed.

    Note: As of this implementation, the model engine *always* calls
    `cuda_graph_runner.pad_batch` and `cuda_graph_runner.maybe_get_cuda_graph`
    even when this is `False`. This could be refactored in the future
    so that the engine bypasses the `CUDAGraphRunner` entirely in Case 1.
    """
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    max_cuda_graph_batch_size: int
    max_beam_width: int
    max_num_tokens: int
    spec_config: Optional[DecodingBaseConfig]
    cuda_graph_mem_pool: Any
    use_mrope: bool
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    is_draft_model: bool
    enable_attention_dp: bool
    batch_size: int
    mapping: Optional[Mapping]
    dist: Optional[MPIDist]
    kv_cache_manager_key: Any
    sparse_attention_config: Optional[BaseSparseAttentionConfig] = None


class EncoderCUDAGraphRunner:
    """
    CUDA graph runner for encoder-only models (e.g., BERT, RoBERTa).
    
    Encoder models key graphs by (batch_size, total_seq_len, max_context_q_len) since:
    1. All tokens are processed in a single forward pass (no generation steps)
    2. Sequence lengths can vary between batches
    3. No KV cache is used
    
    This runner handles three execution paths (similar to CUDAGraphRunner):
    
    1. **Pure Eager Path** (enabled=False):
       - All execution in eager mode
       - `maybe_get_cuda_graph` returns None, signaling eager fallback
       
    2. **Eager Fallback Path** (enabled=True, but ineligible batch):
       - Runner checks eligibility based on batch_size and seq_len
       - Falls back to eager if batch doesn't match supported sizes
       
    3. **CUDA Graph Path** (enabled=True, eligible batch):
       - Graph is captured (if new) or replayed (if exists)
    
    This runner handles:
    - Static input buffer allocation for capture/replay
    - Padding batches and sequences to graph-compatible sizes
    - Graph keying by (padded_batch_size, padded_num_tokens, padded_max_seq_len)
    """
    
    WARMUP_STEPS = 2
    
    def __init__(self, config: EncoderCUDAGraphRunnerConfig):
        self.config = config
        
        # High-level configuration
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = sorted(config.cuda_graph_batch_sizes)
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.supported_num_tokens = sorted(config.cuda_graph_num_tokens)
        self.max_supported_num_tokens = config.max_cuda_graph_num_tokens
        self.supported_seq_lens = sorted(config.cuda_graph_seq_lens)
        
        # Graph storage - keyed by (padded_batch_size, padded_num_tokens, padded_max_seq_len)
        self.graphs: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[Tuple[int, int, int], Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool
        
        # Reusable dummy request for batch padding
        self.padding_dummy_request: Optional[LlmRequest] = None

        # Static tensors for graph capture/replay
        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()
    
    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the max supported number of tokens."""
        max_total_tokens = min(self.max_supported_num_tokens, self.config.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids": torch.ones((max_total_tokens,), device="cuda", dtype=torch.int32),
            "position_ids": torch.zeros((1, max_total_tokens), device="cuda", dtype=torch.int32),
        }

        if self.config.use_token_type_ids:
            self.shared_static_tensors["token_type_ids"] = torch.zeros(
                (max_total_tokens, ), device="cuda", dtype=torch.int32)
    
    def get_graph_key(self, batch: ScheduledRequests) -> Tuple[int, int, int]:
        """Compute the encoder CUDA graph key.

        batch_size is already padded by pad_batch, so no round-up needed.
        Only num_tokens and max_seq_len are rounded up here.
        """
        num_tokens = 0
        max_seq_len = 0
        for request in batch.context_requests:
            seq_len = len(request.get_tokens(0))
            num_tokens += seq_len
            max_seq_len = max(max_seq_len, seq_len)

        # batch_size is already padded by pad_batch, so no round-up needed.
        padded_batch_size = batch.batch_size
        padded_num_tokens = self._round_up(num_tokens, self.supported_num_tokens)
        padded_max_seq_len = self._round_up(max_seq_len, self.supported_seq_lens)

        is_padding_performed = (padded_num_tokens != num_tokens or
                                padded_max_seq_len != max_seq_len)
        is_padding_successful = (padded_num_tokens != 0 and
                                 padded_max_seq_len != 0)

        return (padded_batch_size, padded_num_tokens, padded_max_seq_len), is_padding_performed, is_padding_successful

    @staticmethod
    def _round_up(value: int, supported: list[int]) -> int:
        """Finds the smallest value in `supported` >= `value`. Returns 0 if none."""
        if not supported:
            return 0
        idx = bisect.bisect_left(supported, value)
        if idx == len(supported):
            return 0
        return supported[idx]
    
    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        attn_metadata: Any,
        spec_metadata: Optional[Any] = None,
    ) -> Tuple[Optional[Any], Optional[Tuple[int, int]]]:
        """
        Determine if the current batch can use a CUDA graph.
        
        Returns:
            Tuple of (attn_metadata, graph_key) if graph can be used,
            (None, None) otherwise.
        """
        # A sanity check, flashinfer attention backend doesn't support kv_cache_manager = None
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            logger.warning("Encoder CUDA graph only supports TrtllmAttentionMetadata for now")
            return None, None
        
        # disable when doing statistic
        if ExpertStatistic.should_record():
            return None, None

        can_run_encoder_cuda_graph = not batch.generation_requests
        batch_size = batch.batch_size
        new_batch_size = batch_size
        
        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            all_can_encoder_graph_batch = self.config.dist.tp_allgather(
                [can_run_encoder_cuda_graph, batch_size])
            is_all_can_encoder_graph = all(all_can_encoder_graph[0]
                                  for all_can_encoder_graph in all_can_encoder_graph_batch)

            if is_all_can_encoder_graph:
                new_batch_size = max(encoder_only_batch[1]
                                     for encoder_only_batch in all_can_encoder_graph_batch)

            if not is_all_can_encoder_graph:
                return None, None

        if (not self.enabled
                or not can_run_encoder_cuda_graph
                or spec_metadata is not None # encoder cuda graph doesn't support spec decode
                or new_batch_size > self.max_supported_batch_size):
            return None, None

        # key: (padded_batch_size, padded_num_tokens, padded_max_seq_len)
        key, is_padding_performed, is_padding_successful = self.get_graph_key(batch)
        if (not self.padding_enabled and is_padding_performed) or not is_padding_successful:
            return None, None
        
        if key in self.graphs:
            # The seq_lens in attn_metadata will still be updated in _prepare_tp_inputs_no_cache
            return self.graph_metadata[key]["attn_metadata"], key
        
        # Graph doesn't exist yet - store metadata for capture
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            key[0], # key[0] is the padded batch size
            False,
            0,
            self.cuda_graph_meta_buffers,
            for_encoder_only=True,
        )
        assert graph_attn_metadata.is_cuda_graph

        # Set the max context q len override to the padded max sequence length
        graph_attn_metadata.max_context_q_len_override = key[2]

        # Skip qkv slicing in the attention layer
        graph_attn_metadata.skip_qkv_slicing = True

        return graph_attn_metadata, key
    
    def needs_capture(self, key: Tuple[int, int, int]) -> bool:
        """Check if graph needs to be captured for this key."""
        return key not in self.graphs
    
    def capture(
        self,
        key: Tuple[int, int, int],
        forward_fn: Callable,
        inputs: Dict[str, Any],
    ) -> None:
        """Capture a CUDA graph for the given encoder key.

        Args:
            key: (padded_batch_size, padded_num_tokens, padded_max_seq_len)
            forward_fn: Callable that runs the model forward pass.
            inputs: The inputs dict produced by _prepare_tp_inputs_no_cache.

        All graphs share a single memory pool (set via
        self.memory_pool = graph.pool() after each capture) so the CUDA
        driver can reuse memory across independent graph replays.
        """
        _, padded_num_tokens, _ = key

        sliced_static_tensors = {
            "input_ids": self.shared_static_tensors["input_ids"][:padded_num_tokens],
            "position_ids": self.shared_static_tensors["position_ids"][:, :padded_num_tokens],
        }

        capture_inputs = inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        if "token_type_ids" in self.shared_static_tensors and "token_type_ids" in inputs:
            sliced_static_tensors["token_type_ids"] = self.shared_static_tensors["token_type_ids"][:padded_num_tokens]
            capture_inputs["token_type_ids"] = sliced_static_tensors["token_type_ids"]

        self.graph_metadata[key] = {
            "attn_metadata": inputs["attn_metadata"],
        }

        # Warmup runs required by CUDA graph semantics:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                forward_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = forward_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()
    
    def replay(
        self,
        key: Tuple[int, int, int],
        inputs: Dict[str, Any],
    ) -> Any:
        """
        Replay a previously captured CUDA graph.
        
        Copies current input data into the static tensors used during capture,
        then replays the graph.
        """
        stored_meta = self.graph_metadata[key]
        assert inputs["attn_metadata"] is stored_meta["attn_metadata"]

        _, padded_num_tokens, _ = key
        static_tensors = self.shared_static_tensors

        input_ids = inputs["input_ids"]
        actual_tokens = input_ids.shape[0]

        static_tensors["input_ids"][:actual_tokens].copy_(input_ids)
        static_tensors["input_ids"][actual_tokens:padded_num_tokens].fill_(0)

        position_ids = inputs["position_ids"]
        static_tensors["position_ids"][:, :actual_tokens].copy_(position_ids)
        static_tensors["position_ids"][:, actual_tokens:padded_num_tokens].fill_(0)

        if ("token_type_ids" in inputs and "token_type_ids" in static_tensors):
            static_tensors["token_type_ids"][:actual_tokens].copy_(inputs["token_type_ids"])
            static_tensors["token_type_ids"][actual_tokens:padded_num_tokens].fill_(0)

        self.graphs[key].replay()

        return self.graph_outputs[key]
    
    def get_graph_pool(self):
        """Returns the CUDA memory pool used by this graph runner."""
        return self.memory_pool
    
    def _get_padded_batch(self, batch: ScheduledRequests) -> int:
        """Pad context_requests with 1-token dummy requests to reach padded_batch_size."""
        can_run_encoder_cuda_graph = not batch.generation_requests
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            graph_batch_size = self.config.dist.tp_allgather(
                [can_run_encoder_cuda_graph, batch_size])
            all_can_graph = all(gb[0] for gb in graph_batch_size)
            if all_can_graph:
                new_batch_size = max(gb[1] for gb in graph_batch_size)

        if (not self.enabled or not self.padding_enabled
                or not can_run_encoder_cuda_graph
                or new_batch_size > self.max_supported_batch_size):
            return 0

        padded_batch_size = self._round_up(new_batch_size,
                                           self.supported_batch_sizes)
        if batch_size == padded_batch_size or padded_batch_size == 0:
            return 0

        padding_size = padded_batch_size - batch_size

        if self.padding_dummy_request is None:
            self.padding_dummy_request = LlmRequest(
                request_id=CUDA_GRAPH_DUMMY_REQUEST_ID,
                max_new_tokens=1,
                input_tokens=[0],
                sampling_config=SamplingConfig(
                    SamplingParams()._get_sampling_config()),
                is_streaming=False,
                seq_slot=0,
            )
            self.padding_dummy_request.is_dummy_request = True
            self.padding_dummy_request.is_cuda_graph_dummy = True

        batch.context_requests.extend([self.padding_dummy_request] *
                                      padding_size)
        return padding_size

    @contextlib.contextmanager
    def pad_batch(self, scheduled_requests: ScheduledRequests):
        """Context manager to pad context_requests with 1-token dummy requests."""
        padding_size = self._get_padded_batch(scheduled_requests)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.context_requests = (scheduled_requests.context_requests[:-padding_size])
    
    def clear(self) -> None:
        """Release all captured graphs and associated resources."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len, is_first_draft).
    """
    WARMUP_STEPS = 2

    def __init__(self, config: CUDAGraphRunnerConfig):
        self.config = config

        # High-level configuration from the config object
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = config.cuda_graph_batch_sizes
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.max_beam_width = config.max_beam_width
        self.spec_config = config.spec_config
        self.sparse_config = config.sparse_attention_config

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType,
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[KeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool
        self.padding_dummy_request: Optional["Request"] = None

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest possible batch."""
        max_draft_len = self.config.original_max_total_draft_tokens if self.config.spec_config is not None else 0
        token_per_request = max_draft_len + 1
        max_total_tokens = (self.max_supported_batch_size *
                            self.max_beam_width * token_per_request)
        max_total_tokens = min(max_total_tokens, self.config.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        if self.config.use_mrope:
            self.shared_static_tensors["position_ids"] = torch.zeros(
                (3, 1, max_total_tokens), device="cuda", dtype=torch.int32)
            self.shared_static_tensors["multimodal_params"] = [
                MultimodalParams(
                    multimodal_data={
                        "mrope_config": {
                            "mrope_position_deltas":
                            torch.zeros(
                                (1, 1), device="cuda", dtype=torch.int32)
                        }
                    }) for _ in range(max_total_tokens)
            ]

    def _get_seq_len_mode(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None):
        if self.sparse_config is not None and self.sparse_config.needs_separate_short_long_cuda_graphs(
        ):
            # Some sparse attention algorithms need to use different forward paths for short and long sequences.
            # For example, the DSA can skip the MQA and Top-K in the indexer for short sequences to reduce the
            # computational overhead. To support this feature, we need to capture separate CUDA graphs for short
            # and long sequences. We need to first collect the sequence length of the requests and then determine
            # the sequence length mode. For long sequences, use the default maximum sequence length. For short
            # sequences, use the sequence length threshold as the maximum sequence length.
            total_seq_lens = []
            new_tokens_device, next_draft_tokens_device = None, None
            if new_tensors_device is not None:
                new_tokens_device = new_tensors_device.new_tokens
                if isinstance(new_tensors_device, SampleStateTensorsMTP):
                    next_draft_tokens_device = new_tensors_device.next_draft_tokens
            overlap_scheduler_enabled = new_tokens_device is not None
            for request in batch.generation_requests:
                is_spec_request = get_draft_token_length(
                    request) > 0 or next_draft_tokens_device is not None
                num_draft_tokens = self.spec_config.max_draft_len if is_spec_request else 0
                # First draft
                if request.py_is_first_draft:
                    total_seq_len = len(request.get_tokens(0))
                # With overlap scheduler disabled or dummy request or not assigned to a batch,
                elif not overlap_scheduler_enabled or request.is_dummy or request.py_batch_idx is None:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens
                # Other cases
                else:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens + 1
                total_seq_lens.append(total_seq_len)
            # Determine the sequence length mode.
            from ..speculative import get_num_extra_kv_tokens
            num_extra_kv_tokens = get_num_extra_kv_tokens(self.spec_config)
            max_seq_len = max(total_seq_lens)
            if max_seq_len <= self.sparse_config.seq_len_threshold - num_extra_kv_tokens:
                short_seq_len_mode = True
            else:
                short_seq_len_mode = False
        else:
            # For non-sparse attention or sparse attention that does not need separate short and long CUDA graphs,
            # use the default sequence length mode.
            short_seq_len_mode = False
        return short_seq_len_mode

    def get_graph_key(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            spec_resource_manager: Optional[BaseResourceManager] = None):
        batch_size = batch.batch_size

        # Get the sequence length mode.
        short_seq_len_mode = self._get_seq_len_mode(batch, new_tensors_device)

        if self.config.is_draft_model and spec_resource_manager is not None and isinstance(
                spec_resource_manager, Eagle3ResourceManager):
            # If 'is_first_draft' is True, even with tree decoding, the length of draft_len will only be 'max_draft_len', not 'max_total_draft_token'.
            # Because we will pad the input to 'max_draft_len' length for the first draft layer.
            draft_len = self.config.original_max_draft_len if spec_resource_manager.is_first_draft else 0
            key = (batch_size, draft_len, spec_resource_manager.is_first_draft,
                   short_seq_len_mode)
        else:
            # With dynamic spec decode, the draft length maybe zero even when enable_spec_decode is True,
            # so we need to get the draft length from the batch instead of using enable_spec_decode.
            draft_len_list = []
            for request in batch.generation_requests:
                draft_len_list.append(len(request.py_draft_tokens))
            draft_len = max(draft_len_list)
            assert len(
                set(draft_len_list)) == 1, "All draft lengths must be the same"
            key = (batch_size, draft_len, False, short_seq_len_mode)
        return key

    def __del__(self):
        self.clear()

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        enable_spec_decode: bool,
        attn_metadata: Any,
        spec_metadata: Optional[Any] = None,
        draft_tokens_cuda: Optional[torch.Tensor] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_resource_manager: Optional[BaseResourceManager] = None,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Tuple[int, int, bool]]]:
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        - The key for the graph, if applicable.
        """
        # disable when doing statistic
        if ExpertStatistic.should_record():
            return None, None, None

        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            all_can_graph_batch = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return None, None, None

        if not self.enabled or not can_run_cuda_graph:
            return None, None, None
        key = self.get_graph_key(batch, new_tensors_device,
                                 spec_resource_manager)

        if key in self.graphs:
            return self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"], key

        if batch_size not in self.supported_batch_sizes:
            return None, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, key[1], self.cuda_graph_meta_buffers)
        assert graph_attn_metadata.is_cuda_graph

        if enable_spec_decode:
            graph_spec_metadata = spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            graph_spec_metadata.draft_tokens = draft_tokens_cuda
        else:
            graph_spec_metadata = None
        return graph_attn_metadata, graph_spec_metadata, key

    def needs_capture(self, key: KeyType):
        return key not in self.graph_outputs

    def get_graph_pool(self):
        """Returns the CUDA memory pool used by this graph runner.

        Returns:
            The CUDA memory pool associated with captured graphs, or None if
            no graphs have been captured yet.
        """
        return self.memory_pool

    def capture(self,
                key: KeyType,
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                enable_spec_decode: bool = False,
                postprocess_fn: Optional[Callable] = None):
        """Captures the forward pass for a given batch size."""
        batch_size = key[0]
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        max_draft_len = key[1]
        token_per_request = max_draft_len + 1
        num_tokens_for_capture = (batch_size * self.max_beam_width *
                                  token_per_request)

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:num_tokens_for_capture],
            "position_ids":
            self.shared_static_tensors["position_ids"]
            [:, :num_tokens_for_capture],
        }
        if self.config.use_mrope:
            sliced_static_tensors["position_ids"] = self.shared_static_tensors[
                "position_ids"][:, :, :num_tokens_for_capture]
            sliced_static_tensors[
                "multimodal_params"] = self.shared_static_tensors[
                    "multimodal_params"][:batch_size * self.max_beam_width]

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        def _setup_spec_decoding_and_forward(key: KeyType, forward_fn: Callable,
                                             capture_inputs: Dict[str, Any]):
            is_first_draft = key[2]
            needs_kv_cache_recompute = True if enable_spec_decode and self.config.spec_config.spec_dec_mode.needs_kv_cache_recompute(
            ) else False
            if is_first_draft and self.config.is_draft_model and needs_kv_cache_recompute:
                capture_inputs['attn_metadata'].use_spec_decoding = True
            return forward_fn(capture_inputs)

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                _setup_spec_decoding_and_forward(key, forward_fn,
                                                 capture_inputs)
                if postprocess_fn is not None:
                    postprocess_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = _setup_spec_decoding_and_forward(
                    key, forward_fn, capture_inputs)
            if postprocess_fn is not None:
                postprocess_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay(self, key: KeyType,
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        stored_meta = self.graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]
        if stored_meta["spec_metadata"] is not None:
            assert current_inputs.get(
                "spec_metadata") is stored_meta["spec_metadata"]

        static_tensors = self.shared_static_tensors

        input_ids = current_inputs["input_ids"]
        seqlen = input_ids.shape[0]
        static_tensors["input_ids"][:seqlen].copy_(input_ids)

        position_ids = current_inputs["position_ids"]
        if self.config.use_mrope and current_inputs.get(
                'multimodal_params') is not None:
            static_tensors["position_ids"][:, :, :seqlen].copy_(position_ids)
            for i, multimodal_param in enumerate(
                    current_inputs['multimodal_params']):
                # NOTE: Currently, we only need 'mrope_position_deltas' on generation phase for multimodal models.
                static_tensors['multimodal_params'][i].multimodal_data[
                    'mrope_config']['mrope_position_deltas'].copy_(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'],
                        non_blocking=True)
        else:
            static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_padded_batch(self, batch: ScheduledRequests,
                          resource_manager: ResourceManager,
                          runtime_draft_len: int) -> int:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            graph_batch_size = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            all_can_graph = all(graph_batch[0]
                                for graph_batch in graph_batch_size)
            if all_can_graph:
                new_batch_size = max(gen_only_batch[1]
                                     for gen_only_batch in graph_batch_size)

        if (not self.enabled or not self.padding_enabled
                or not can_run_cuda_graph
                or new_batch_size > self.max_supported_batch_size):
            return 0

        padded_batch_size = self._round_up_batch_size(new_batch_size)
        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size + batch.batch_size > self.config.batch_size:
            return 0

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        if self.padding_dummy_request is None:
            available_blocks = kv_cache_manager.get_num_free_blocks()
            # No padding if not enough KV cache space
            if available_blocks < 1:
                return 0

            self.padding_dummy_request = kv_cache_manager.add_dummy_requests(
                [CUDA_GRAPH_DUMMY_REQUEST_ID],
                is_gen=True,
                max_num_draft_tokens=runtime_draft_len,
                use_mrope=self.config.use_mrope,
                max_beam_width=self.config.max_beam_width)[0]
            self.padding_dummy_request.is_cuda_graph_dummy = True
            spec_res_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_res_mgr:
                spec_res_mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

        self.padding_dummy_request.py_draft_tokens = [0] * runtime_draft_len
        batch.generation_requests.extend([self.padding_dummy_request] *
                                         padding_size)
        return padding_size

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    @contextlib.contextmanager
    def pad_batch(self,
                  scheduled_requests: ScheduledRequests,
                  resource_manager: ResourceManager,
                  runtime_draft_len: int = 0):
        """Context manager to pad a batch to a graph-compatible size."""
        padding_size = self._get_padded_batch(scheduled_requests,
                                              resource_manager,
                                              runtime_draft_len)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        self.padding_dummy_request = None
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
