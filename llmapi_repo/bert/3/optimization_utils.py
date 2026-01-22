"""
Optimization utilities for BERT Triton backend: torch.compile and CUDA graph management.

EncoderCUDAGraphManager is a lightweight graph runner adapted from TRT-LLM's
EncoderCUDAGraphRunner, stripped of parallelism, spec-decode, and executor
dependencies.  It manages capture/replay of CUDA graphs keyed by
(padded_batch_size, padded_num_tokens, padded_max_seq_len).
"""

import bisect
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.utils import make_weak_ref

WARMUP_STEPS = 3


def apply_torch_compile(model: torch.nn.Module):
    """Apply torch.compile to the model.

    Args:
        model: The model to compile.

    Returns:
        Compiled model, or original model if compilation fails.
    """
    try:
        compiled = torch.compile(model, dynamic=True)
        print(f"[optimization_utils] torch.compile applied")
        return compiled
    except Exception as e:
        print(f"[optimization_utils] torch.compile failed, using eager: {e}")
        return model


class EncoderCUDAGraphManager:
    """Lightweight CUDA graph manager for encoder-only models.

    Manages a set of CUDA graphs keyed by (padded_batch_size, padded_num_tokens,
    padded_max_seq_len).  Each graph has its own TrtllmAttentionMetadata with
    address-stable buffers, and shares a single set of static input tensors.
    """

    def __init__(
        self,
        supported_batch_sizes: List[int],
        supported_num_tokens: List[int],
        supported_seq_lens: List[int],
        max_num_tokens: int,
        max_batch_size: int,
        use_token_type_ids: bool = True,
    ):
        self.supported_batch_sizes = sorted(supported_batch_sizes)
        self.supported_num_tokens = sorted(supported_num_tokens)
        self.supported_seq_lens = sorted(supported_seq_lens)
        self.max_num_tokens = max_num_tokens
        self.max_batch_size = max_batch_size
        self.use_token_type_ids = use_token_type_ids

        self.graphs: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[Tuple[int, int, int], Any] = {}
        self.graph_metadata: Dict[Tuple[int, int, int], TrtllmAttentionMetadata] = {}
        self.memory_pool = None

        max_supported_nt = min(
            max(supported_num_tokens) if supported_num_tokens else max_num_tokens,
            max_num_tokens,
        )

        # Create a single combined tensor for all static tensors to reduce fill_ calls during cuda graph replay
        num_segments = 3 if use_token_type_ids else 2
        self._combined = torch.zeros(
            num_segments * max_supported_nt, device="cuda", dtype=torch.int32
        )
        self.shared_static_tensors: Dict[str, torch.Tensor] = {
            "input_ids": self._combined[:max_supported_nt],
            "position_ids": self._combined[max_supported_nt:2 * max_supported_nt].view(1, -1),
        }
        if use_token_type_ids:
            self.shared_static_tensors["token_type_ids"] = self._combined[2 * max_supported_nt:]

        self.cuda_graph_meta_buffers = get_memory_buffers()

        self._seq_lens_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32
        ).pin_memory()

    # ------------------------------------------------------------------
    # Key computation & batch padding
    # ------------------------------------------------------------------

    def _round_up(self, value: int, supported: List[int]) -> int:
        idx = bisect.bisect_left(supported, value)
        if idx == len(supported):
            return 0
        return supported[idx]

    def pad_batch(
        self,
        batch_size: int,
        num_tokens: int,
        max_seq_len: int,
        seq_lengths: List[int],
    ) -> Tuple[int, int, int, List[int], int]:
        """Pad batch with 1-token dummy entries to reach the next supported batch size.

        Mirrors EncoderCUDAGraphRunner._get_padded_batch: instead of zero-length
        entries in seq_lens (which cause the BERT pooler to index out of bounds),
        each dummy slot gets a 1-token sequence.

        Returns:
            (padded_bs, adjusted_num_tokens, max_seq_len, padded_seq_lengths, padding_size)
        """
        padded_batch_size = self._round_up(batch_size, self.supported_batch_sizes)
        if padded_batch_size == 0 or padded_batch_size == batch_size:
            return batch_size, num_tokens, max_seq_len, list(seq_lengths)

        padding_size = padded_batch_size - batch_size
        padded_seq_lengths = list(seq_lengths) + [1] * padding_size
        adjusted_num_tokens = num_tokens + padding_size

        return padded_batch_size, adjusted_num_tokens, max_seq_len, padded_seq_lengths

    def get_graph_key(
        self, batch_size: int, num_tokens: int, max_seq_len: int
    ) -> Optional[Tuple[int, int, int]]:
        """Compute graph key from already-padded batch_size and adjusted num_tokens.

        batch_size is expected to already be at a supported value (from pad_batch).
        Only num_tokens and max_seq_len are rounded up here.
        """
        padded_num_tokens = self._round_up(num_tokens, self.supported_num_tokens)
        padded_max_seq_len = self._round_up(max_seq_len, self.supported_seq_lens)
        if batch_size == 0 or padded_num_tokens == 0 or padded_max_seq_len == 0:
            return None
        return (batch_size, padded_num_tokens, padded_max_seq_len)

    # ------------------------------------------------------------------
    # Warmup — enumerate all feasible keys, capture each one
    # ------------------------------------------------------------------

    def warmup(
        self,
        model_fn: Callable[[Dict[str, Any], TrtllmAttentionMetadata], Any],
        eager_attn_metadata: TrtllmAttentionMetadata,
    ):
        """Capture CUDA graphs for all feasible (bs, nt, sl) combinations.

        Adapted from model_engine._run_cuda_graph_warmup_no_cache.
        Iterates the 3D grid largest-first so smaller graphs can reuse the
        shared memory pool.
        """
        batch_sizes = sorted(self.supported_batch_sizes, reverse=True)
        num_tokens_list = sorted(self.supported_num_tokens)
        seq_lens_list = sorted(self.supported_seq_lens)

        num_captured = 0
        for bs in batch_sizes:
            if bs > self.max_batch_size:
                continue
            for sl in reversed(seq_lens_list):
                sl_idx = seq_lens_list.index(sl)
                prev_sl = seq_lens_list[sl_idx - 1] if sl_idx > 0 else 0

                for nt in reversed(num_tokens_list):
                    nt_idx = num_tokens_list.index(nt)
                    prev_nt = num_tokens_list[nt_idx - 1] if nt_idx > 0 else 0

                    if nt < prev_sl + bs or prev_nt >= bs * sl:
                        continue
                    if nt > self.max_num_tokens:
                        continue

                    key = (bs, nt, sl)
                    print(f"[cuda_graph] Capturing key (bs={bs}, nt={nt}, sl={sl})")
                    self._capture(key, model_fn, eager_attn_metadata)
                    num_captured += 1
                    torch.cuda.synchronize()

        print(f"[cuda_graph] Captured {num_captured} encoder CUDA graphs")

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(
        self,
        key: Tuple[int, int, int],
        model_fn: Callable,
        eager_attn_metadata: TrtllmAttentionMetadata,
    ):
        padded_bs, padded_nt, padded_sl = key

        graph_meta = eager_attn_metadata.create_cuda_graph_metadata(
            padded_bs,
            False,
            0,
            self.cuda_graph_meta_buffers,
            for_encoder_only=True,
        )
        assert graph_meta.is_cuda_graph
        graph_meta.max_context_q_len_override = padded_sl
        graph_meta.skip_qkv_slicing = True

        # Populate seq_lens with a valid synthetic pattern for capture
        synthetic_seq_lens = self._make_synthetic_seq_lens(padded_bs, padded_nt, padded_sl)
        graph_meta.seq_lens = synthetic_seq_lens
        graph_meta.num_contexts = padded_bs
        graph_meta.max_seq_len = padded_sl
        graph_meta.request_ids = list(range(padded_bs))
        graph_meta.host_request_types.fill_(0) # Required for prepare_encoder_only
        graph_meta.prepare_encoder_only()

        sliced = {
            "input_ids": self.shared_static_tensors["input_ids"][:padded_nt],
            "position_ids": self.shared_static_tensors["position_ids"][:, :padded_nt],
        }
        if "token_type_ids" in self.shared_static_tensors:
            sliced["token_type_ids"] = self.shared_static_tensors["token_type_ids"][:padded_nt]

        self.graph_metadata[key] = graph_meta

        graph = torch.cuda.CUDAGraph()
        for _ in range(WARMUP_STEPS):
            model_fn(sliced, graph_meta)

        with torch.cuda.graph(graph, pool=self.memory_pool):
            output = model_fn(sliced, graph_meta)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    @staticmethod
    def _make_synthetic_seq_lens(
        padded_bs: int, padded_nt: int, padded_sl: int
    ) -> torch.Tensor:
        """Create a synthetic seq_lens tensor that sums to padded_nt with max <= padded_sl."""
        if padded_bs == 0:
            return torch.zeros((0,), dtype=torch.int32)
        base = padded_nt // padded_bs
        remainder = padded_nt % padded_bs
        lens = [min(base + (1 if i < remainder else 0), padded_sl) for i in range(padded_bs)]
        # If total is short due to clamping, distribute among unclamped entries
        total = sum(lens)
        deficit = padded_nt - total
        for i in range(padded_bs):
            if deficit <= 0:
                break
            add = min(deficit, padded_sl - lens[i])
            lens[i] += add
            deficit -= add
        return torch.tensor(lens, dtype=torch.int32).pin_memory()

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        key: Tuple[int, int, int],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        seq_lengths: List[int],
    ) -> Any:
        """Copy real inputs into static tensors, update metadata, replay graph.

        seq_lengths is expected to already include 1-token dummy entries
        from pad_batch (length == padded_bs).
        """
        _, padded_nt, _ = key
        actual_tokens = input_ids.shape[0]
        graph_meta = self.graph_metadata[key]

        self._combined.fill_(0)
        static = self.shared_static_tensors
        static["input_ids"][:actual_tokens].copy_(input_ids, non_blocking=True)
        static["position_ids"][0, :actual_tokens].copy_(
            position_ids if position_ids.ndim == 1 else position_ids[0],
            non_blocking=True,
        )
        if token_type_ids is not None and "token_type_ids" in static:
            static["token_type_ids"][:actual_tokens].copy_(token_type_ids, non_blocking=True)
        n = len(seq_lengths)
        self._seq_lens_buffer[:n].copy_(torch.tensor(seq_lengths, dtype=torch.int32))
        graph_meta.seq_lens = self._seq_lens_buffer[:n]
        graph_meta.prepare_encoder_only()

        self.graphs[key].replay()
        return self.graph_outputs[key]

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def try_run(
        self,
        batch_size: int,
        num_tokens: int,
        max_seq_len: int,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lengths: List[int],
    ) -> Optional[Any]:
        """Try to replay a matching CUDA graph. Returns None for eager fallback.

        Workflow mirrors the LLM API's EncoderCUDAGraphRunner:
        1. pad_batch — extend seq_lengths with 1-token dummies, adjust num_tokens
        2. get_graph_key — round up num_tokens and max_seq_len
        3. replay — copy inputs into static tensors, update metadata, replay
        4. slice output to actual batch_size (discard dummy rows)
        """
        padded_batch_size, adjusted_num_tokens, max_seq_len, padded_seq_lengths = self.pad_batch(
            batch_size, num_tokens, max_seq_len, seq_lengths,
        )
        key = self.get_graph_key(padded_batch_size, adjusted_num_tokens, max_seq_len)
        if key is None or key not in self.graphs:
            return None

        output = self.replay(key, input_ids, position_ids, token_type_ids, padded_seq_lengths)
        return output[:batch_size]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self):
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        self.memory_pool = None
        torch.cuda.empty_cache()
