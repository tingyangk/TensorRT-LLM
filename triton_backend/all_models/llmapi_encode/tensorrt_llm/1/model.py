# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
import triton_python_backend_utils as pb_utils
import yaml

# NOTE: tensorrt_llm and transformers are imported in initialize(),
# not at module level. This matches the decoder llmapi/ backend — deferring
# CUDA-initializing imports until after Triton has set up the process's GPU
# context preserves correct device assignment. See llmapi/tensorrt_llm/1/model.py
# for the full rationale around MPI_Comm_spawn and CUDA_VISIBLE_DEVICES.


def get_model_config(filename):
    """Load YAML config from the model directory."""
    filepath = os.path.join(pb_utils.get_model_dir(), filename)
    with open(filepath) as f:
        return yaml.safe_load(f)


class TritonPythonModel:
    def initialize(self, args):
        """Load the model and tokenizer once at startup."""
        self.logger = pb_utils.Logger

        config = get_model_config(os.environ.get("LLM_CONFIG_PATH", "model.yaml"))
        model_name = config["model"]
        self.max_seq_len = config.get("max_seq_len", 512)

        from transformers import AutoTokenizer

        from tensorrt_llm import LLM

        # Recommended to create tokenizer here for batch tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build LLM engine args from model.yaml, excluding triton-specific
        # keys that aren't LLM constructor arguments.
        llm_args = {k: v for k, v in config.items() if k not in ("triton_config",)}
        self.llm = LLM(**llm_args)

        self.logger.log_info(f"Initialized {model_name} with encode_only=True")

    def execute(self, requests):
        """Run a single forward pass on the dynamically batched requests.

        Args:
            requests: List of pb_utils.InferenceRequest from Triton's
                dynamic batcher.

        Returns:
            List of pb_utils.InferenceResponse, one per request. A malformed
            request produces an error response at its own index without
            failing the rest of the batch.
        """
        responses = [None] * len(requests)
        texts = []
        valid_indices = []

        # Step 1: Per-request validation. A bad request only fails itself.
        for i, request in enumerate(requests):
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                if text_tensor is None:
                    raise ValueError("text_input is missing from the request")
                # Use .item() because we set dims: [1] for "text_input" in config.pbtxt
                text = text_tensor.as_numpy().item()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                texts.append(text)
                valid_indices.append(i)
            except Exception as e:
                responses[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))

        if texts:
            # Step 2 (recommended): Batch tokenize here to avoid the per-prompt
            # tokenization loop inside encode().
            encoded = self.tokenizer(
                texts, padding=False, truncation=True, max_length=self.max_seq_len
            )

            # Step 3: Single forward pass via encode().
            outputs = self.llm.encode(encoded["input_ids"])

            # Step 4: Build Triton responses for the valid subset.
            for j, i in enumerate(valid_indices):
                logits = outputs[j].logits.float().numpy()
                predictions = np.expand_dims(
                    np.array([logits.argmax(axis=-1)], dtype=np.int32), axis=0
                )
                responses[i] = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("logits", np.expand_dims(logits, axis=0)),
                        pb_utils.Tensor("predictions", predictions),
                    ]
                )

        return responses

    def finalize(self):
        """Clean up on model unload."""
        self.logger.log_info("Shutting down LLM engine")
        if hasattr(self, "llm") and self.llm is not None:
            self.llm.shutdown()
