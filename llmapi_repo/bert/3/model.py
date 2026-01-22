"""
Triton Python Backend for high-performance BERT inference.

Bypasses the LLM API executor loop and calls TRT-LLM's
BertForSequenceClassification directly with pre-allocated static
buffers, optional torch.compile, and encoder CUDA graph support.
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

# ────────────────────────────────────────────────────────────────────
# Configuration — adjust these for your deployment
# ────────────────────────────────────────────────────────────────────
MODEL_NAME = "textattack/bert-base-uncased-yelp-polarity"
DTYPE = torch.float16
MAX_BATCH_SIZE = 64
MAX_NUM_TOKENS = 8192
MAX_SEQ_LEN = 512

# Note that there's no torch.compile warmup in this implementation
# Use torch.compile with encoder CUDA graph for better performance
ENABLE_TORCH_COMPILE = True # Use default inductor backend

ENABLE_CUDA_GRAPH = True
SUPPORTED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SUPPORTED_NUM_TOKENS = [8, 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
SUPPORTED_SEQ_LENS = [8, 16, 32, 64, 128, 256, 512]
# ────────────────────────────────────────────────────────────────────


class TritonPythonModel:
    """Triton Python Backend Model for BERT classification."""

    def initialize(self, args: Dict[str, Any]) -> None:
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from bert_runner import BertRunner
        from optimization_utils import (
            EncoderCUDAGraphManager,
            apply_torch_compile,
        )

        pb_utils.Logger.log_info(f"Initializing BertRunner for {MODEL_NAME}")

        self.runner = BertRunner(
            model_name=MODEL_NAME,
            device="cuda",
            dtype=DTYPE,
            max_batch_size=MAX_BATCH_SIZE,
            max_num_tokens=MAX_NUM_TOKENS,
            max_seq_len=MAX_SEQ_LEN,
        )

        if ENABLE_TORCH_COMPILE:
            pb_utils.Logger.log_info(f"Applying torch.compile")
            self.runner.model = apply_torch_compile(self.runner.model)

        self.cuda_graph_manager = None
        if ENABLE_CUDA_GRAPH:
            pb_utils.Logger.log_info("Building encoder CUDA graphs")
            self.cuda_graph_manager = EncoderCUDAGraphManager(
                supported_batch_sizes=SUPPORTED_BATCH_SIZES,
                supported_num_tokens=SUPPORTED_NUM_TOKENS,
                supported_seq_lens=SUPPORTED_SEQ_LENS,
                max_num_tokens=MAX_NUM_TOKENS,
                max_batch_size=MAX_BATCH_SIZE,
            )
            self.cuda_graph_manager.warmup(
                model_fn=self.runner._eager_forward,
                eager_attn_metadata=self.runner.eager_attn_metadata,
            )

        pb_utils.Logger.log_info("BERT model ready")

    def execute(self, requests: List[Any]) -> List[Any]:
        """Batch all requests into a single forward pass.

        With dynamic_batching enabled in config.pbtxt, Triton groups
        concurrent requests and delivers them together in `requests`.
        We collect one text per request, run one batched forward pass,
        then split the logits back into per-request responses.
        """
        all_texts = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            text_data = text_tensor.as_numpy()
            text = text_data.flatten()[0]
            if isinstance(text, bytes):
                text = text.decode("utf-8").rstrip()
            else:
                text = str(text).rstrip()
            all_texts.append(text)

        logits = self.runner.generate(
            all_texts,
            max_length=MAX_SEQ_LEN,
            cuda_graph_manager=self.cuda_graph_manager,
        )
        logits_np = logits.cpu().float().numpy()

        responses = []
        for i in range(len(requests)):
            out_tensor = pb_utils.Tensor("logits", logits_np[i:i+1].flatten())
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self) -> None:
        if hasattr(self, "cuda_graph_manager") and self.cuda_graph_manager is not None:
            self.cuda_graph_manager.clear()
        if hasattr(self, "runner"):
            del self.runner
        pb_utils.Logger.log_info("BERT model cleanup completed")
