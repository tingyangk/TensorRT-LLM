"""
High-performance BERT inference runner using TRT-LLM's BertForSequenceClassification
with pre-allocated static buffers, bypassing the LLM API executor loop.
"""

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Tuple

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bert import BertForSequenceClassification
from tensorrt_llm.mapping import Mapping


class BertRunner:
    """Direct BERT inference runner with pre-allocated static buffers.

    Avoids per-iteration overhead by:
    - Reusing a single TrtllmAttentionMetadata instance for eager mode
    - Pre-allocating all CUDA and pinned-CPU tensors once at init
    - Calling the model forward() directly (no executor/scheduler)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_batch_size: int = 64,
        max_num_tokens: int = 8192,
        max_seq_len: int = 512,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.torch_dtype = dtype
        hf_config.num_key_value_heads = hf_config.num_attention_heads

        model_config = ModelConfig(
            pretrained_config=hf_config,
            mapping=Mapping(),
            attn_backend="TRTLLM",
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )

        self.model = BertForSequenceClassification(model_config)
        self._load_weights(model_name, hf_config)
        self.model.to(self.device)
        # self.model.half() # DUMMY weights hack
        self.model.eval()

        self.eager_attn_metadata = self._create_eager_metadata()

    def _load_weights(self, model_name: str, hf_config):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=hf_config.torch_dtype
        )
        hf_model.eval()
        self.model.load_weights(hf_model.state_dict())
        del hf_model
        torch.cuda.empty_cache()

    def _create_eager_metadata(self) -> TrtllmAttentionMetadata:
        """Create a single reusable TrtllmAttentionMetadata for eager mode."""
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=self.max_num_tokens,
            kv_cache_manager=None,
            mapping=Mapping(),
        )
        metadata.kv_cache_params = KVCacheParams(use_cache=False)
        metadata.kv_cache_block_offsets = None
        metadata.host_kv_cache_block_offsets = None
        metadata.block_ids_per_seq = None
        metadata.host_request_types.fill_(0)
        metadata.host_total_kv_lens[1] = 0
        return metadata

    def tokenize_and_pack(
        self, texts: List[str], max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Tokenize texts and return packed (flattened) tensors.

        Returns:
            (flat_input_ids, flat_token_type_ids, flat_position_ids, seq_lengths)
            All tensors are on CPU; the caller or forward() copies them to CUDA.
        """
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        flat_input_ids = []
        flat_token_type_ids = []
        flat_position_ids = []
        seq_lengths = []

        for i in range(len(texts)):
            ids = encoded["input_ids"][i]
            length = len(ids)
            seq_lengths.append(length)
            flat_input_ids.extend(ids)
            if "token_type_ids" in encoded:
                flat_token_type_ids.extend(encoded["token_type_ids"][i])
            else:
                flat_token_type_ids.extend([0] * length)
            flat_position_ids.extend(range(length))

        # Intentionally NOT using pin_memory=True here to avoid cudaEventQuery calls from caching allocator.
        # Though cudaStreamSynchronize may be called in this way, it's fine since these memcpy ops are CPU bounded.
        result = (
            torch.tensor(flat_input_ids, dtype=torch.int32),
            torch.tensor(flat_token_type_ids, dtype=torch.int32),
            torch.tensor(flat_position_ids, dtype=torch.int32),
            seq_lengths,
        )
        return result

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lengths: List[int],
        cuda_graph_manager=None,
    ) -> torch.Tensor:
        """Run BERT forward pass with minimal overhead.

        Args:
            input_ids: Flat (packed) input token IDs, shape (num_tokens,). CPU or CUDA.
            token_type_ids: Flat token type IDs, same shape. CPU or CUDA.
            position_ids: Flat position IDs, same shape. CPU or CUDA.
            seq_lengths: List of per-sequence lengths.
            cuda_graph_manager: Optional EncoderCUDAGraphManager for graph replay.

        Returns:
            Classification logits, shape (batch_size, num_labels).
        """
        num_tokens = input_ids.shape[0]
        batch_size = len(seq_lengths)
        max_sl = max(seq_lengths)

        if cuda_graph_manager is not None:
            result = cuda_graph_manager.try_run(
                batch_size=batch_size,
                num_tokens=num_tokens,
                max_seq_len=max_sl,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                seq_lengths=seq_lengths,
            )
            if result is not None:
                return result

        result = self._eager_forward_from_raw(
            input_ids, token_type_ids, position_ids, seq_lengths
        )
        return result

    def _eager_forward_from_raw(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lengths: List[int],
    ) -> torch.Tensor:
        """Eager forward: copies raw inputs to static CUDA buffers, prepares metadata, runs model."""
        batch_size = len(seq_lengths)
        max_sl = max(seq_lengths)

        inputs = {
            "input_ids": input_ids.cuda(non_blocking=True),
            "token_type_ids": token_type_ids.cuda(non_blocking=True),
            "position_ids": position_ids.cuda(non_blocking=True),
        }
        attn_metadata = self._prepare_eager_metadata(batch_size, seq_lengths, max_sl)
        result = self._eager_forward(inputs, attn_metadata)
        return result

    def _eager_forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attn_metadata: TrtllmAttentionMetadata,
    ) -> torch.Tensor:
        """Run model forward with pre-prepared inputs dict and metadata.
        Uses self.model which may be the torch.compile'd version."""
        result = self.model(
            attn_metadata=attn_metadata,
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
        )
        return result

    def _prepare_eager_metadata(
        self,
        batch_size: int,
        seq_lengths: List[int],
        max_seq_len: int,
    ) -> TrtllmAttentionMetadata:
        """Update the reusable eager metadata in-place for this batch."""
        metadata = self.eager_attn_metadata
        metadata.seq_lens = torch.tensor(seq_lengths, dtype=torch.int32, pin_memory=True)
        metadata.num_contexts = batch_size
        metadata.max_seq_len = max_seq_len
        metadata.request_ids = list(range(batch_size))
        metadata.prepare_encoder_only()

        return metadata

    @torch.inference_mode()
    def generate(self, texts: List[str], max_length: int = 512, cuda_graph_manager=None):
        """High-level API: tokenize texts and return classification logits."""
        input_ids, token_type_ids, position_ids, seq_lengths = self.tokenize_and_pack(
            texts, max_length=max_length
        )
        result = self.forward(
            input_ids, token_type_ids, position_ids, seq_lengths,
            cuda_graph_manager=cuda_graph_manager,
        )
        return result
