# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy tests for the llm.encode() path across encoder and decoder models.

These tests exercise the LLM(encode_only=True) / llm.encode() single-forward
prefill path and verify output correctness by direct logits comparison against
HuggingFace reference models. Unlike the dataset-driven AccuracyTask flow, we
compare tensors in-process — encode() doesn't generate, so there's no
SamplingParams / EVALUATOR_CLS to plug into.

Each decoder model is chosen as the *sole representative* of a distinct TRT-LLM
model architecture class (e.g. LlamaForCausalLM, Gemma3ForCausalLM).
Memory-gated models use pytest.param(marks=skip_less_device_memory(...)) so
too-small GPUs auto-skip.

Note: encode() is single-GPU only (no TP/PP). Every listed model is
architecturally required to fit on one GPU for these tests.
"""

import pytest
import torch

from tensorrt_llm import LLM

from ..conftest import llm_models_root
from .accuracy_core import LlmapiAccuracyTestHarness

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


# --------------------------------------------------------------------------- #
# Encoder-only models (non-multimodal)
# --------------------------------------------------------------------------- #
#
# Only architectures registered in tensorrt_llm/_torch/models/ via
# @register_auto_model are resolvable. Plain BertModel / RobertaModel / DeBERTa
# are NOT currently registered in the PyTorch backend.
#
# The third tuple element is an output-type tag that selects the HF reference
# class and comparison strategy.
ENCODER_MODELS = [
    pytest.param(
        "textattack/bert-base-uncased-yelp-polarity",
        f"{llm_models_root()}/bert/bert-base-uncased-yelp-polarity",
        "classification",
        id="bert-yelp",
    ),
    pytest.param(
        "Qwen/Qwen2.5-Math-PRM-7B",
        f"{llm_models_root()}/Qwen2.5-Math-PRM-7B",
        "per_token_reward",
        marks=pytest.mark.skip_less_device_memory(32000),
        id="qwen2.5-prm-7b",
    ),
]


class TestEncoderEncode(LlmapiAccuracyTestHarness):
    """HF logits-level accuracy for encoder-only (non-MM) architectures.

    Inherits LlmapiAccuracyTestHarness only for its class-scoped logger-level
    fixture. The harness' MODEL_NAME / MODEL_PATH attributes are not used
    here because the model differs per parametrize invocation.
    """

    @pytest.mark.parametrize("model_name,model_path,output_type", ENCODER_MODELS)
    def test_encode_matches_huggingface(self, model_name, model_path, output_type):
        """encode() logits match HF reference within tolerance / argmax."""
        if output_type == "classification":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            hf_cls = AutoModelForSequenceClassification
        else:
            # Reward models share the causal-LM backbone + a scoring head.
            # Loading via AutoModelForCausalLM gives us the raw token logits;
            # trust_remote_code handles models whose heads are defined in
            # their own modeling file (e.g., Qwen2ForProcessRewardModel).
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_cls = AutoModelForCausalLM

        with LLM(model_path, encode_only=True) as llm:
            outs = llm.encode(PROMPTS)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_model = hf_cls.from_pretrained(model_path, trust_remote_code=True).half().cuda().eval()
        with torch.inference_mode():
            inputs = tokenizer(PROMPTS, return_tensors="pt", padding="longest").to(hf_model.device)
            hf_logits = hf_model(**inputs).logits.float().cpu()

        if output_type == "classification":
            # Classification heads produce identically-shaped outputs;
            # direct tensor comparison with the same FP16 tolerance as the
            # unit test (rtol=atol=1.5e-2).
            tllm_logits = torch.stack([o.logits.cpu() for o in outs])
            torch.testing.assert_close(tllm_logits, hf_logits, rtol=1.5e-2, atol=1.5e-2)
        else:
            # Per-token reward output may be nested / per-sequence in TRT-LLM
            # (see Qwen2ForProcessRewardModel). Compare last-token argmax
            # per prompt — robust to shape differences.
            for i in range(len(PROMPTS)):
                t = outs[i].logits.cpu().float()
                t_last = t[-1] if t.dim() > 1 else t
                hf_last = hf_logits[i, -1]
                assert t_last.argmax(dim=-1) == hf_last.argmax(dim=-1), (
                    f"[{model_name}] prompt#{i} argmax mismatch: "
                    f"TLLM={t_last.argmax(dim=-1)} vs HF={hf_last.argmax(dim=-1)}"
                )


# --------------------------------------------------------------------------- #
# Decoder models used in single-prefill mode
# --------------------------------------------------------------------------- #
#
# encode() on a decoder model runs a single prefill and returns logits without
# running the autoregressive loop. Use case: embedding extraction, reward /
# classification scoring on a causal LM backbone.
#
# One representative per distinct TRT-LLM architecture class:
#   LlamaForCausalLM   — TinyLlama (also covers Mistral, which aliases LlamaModel)
#   Gemma3ForCausalLM  — Gemma-3-1B (sliding window + global alternation)
#   Phi3ForCausalLM    — Phi-4-mini (SuRoPE, merged QKV)
#   Qwen2ForCausalLM   — Qwen2-7B (distinct GQA head config, SwiGLU variant)
#   Qwen3ForCausalLM   — Qwen3-0.6B (QKNorm, architecturally distinct from Qwen2)
#   Starcoder2ForCausalLM — StarCoder2-3B (MQA, sliding window, code model)
DECODER_MODELS = [
    # -- LlamaForCausalLM (covers Llama + Mistral family) --
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        id="tinyllama-1.1b",
    ),
    # -- Gemma3ForCausalLM --
    pytest.param(
        "google/gemma-3-1b-it",
        f"{llm_models_root()}/gemma/gemma-3-1b-it/",
        id="gemma-3-1b",
    ),
    # -- Phi3ForCausalLM --
    pytest.param(
        "microsoft/Phi-4-mini-instruct",
        f"{llm_models_root()}/Phi-4-mini-instruct",
        marks=pytest.mark.skip_less_device_memory(24000),
        id="phi-4-mini",
    ),
    # -- Qwen2ForCausalLM --
    pytest.param(
        "Qwen/Qwen2-7B-Instruct",
        f"{llm_models_root()}/Qwen2-7B-Instruct",
        marks=pytest.mark.skip_less_device_memory(32000),
        id="qwen2-7b",
    ),
    # -- Qwen3ForCausalLM --
    pytest.param(
        "Qwen/Qwen3-0.6B",
        f"{llm_models_root()}/Qwen3/Qwen3-0.6B",
        id="qwen3-0.6b",
    ),
    # -- Starcoder2ForCausalLM --
    pytest.param(
        "bigcode/starcoder2-3b",
        f"{llm_models_root()}/starcoder2-3b/",
        id="starcoder2-3b",
    ),
]


class TestDecoderEncode(LlmapiAccuracyTestHarness):
    """Validates encode() on decoder models used in single-prefill mode.

    The encode_only flag claims support for decoder models running a single
    prefill (e.g., for embedding extraction). This class exercises one
    representative per distinct TRT-LLM architecture class.
    """

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("model_name,model_path", DECODER_MODELS)
    def test_encode_matches_huggingface(self, model_name, model_path):
        """encode() last-token argmax matches HF causal-LM prefill."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        prompts = ["The quick brown fox"]
        with LLM(model_path, encode_only=True) as llm:
            tllm_logits = llm.encode(prompts)[0].logits.cpu().float()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda().eval()
        with torch.inference_mode():
            inputs = tokenizer(prompts, return_tensors="pt").to(hf_model.device)
            hf_logits = hf_model(**inputs).logits.float().cpu()

        hf_last_pred = hf_logits[0, -1].argmax(dim=-1)
        # TRT-LLM may return [vocab_size] (last-token only) or
        # [seq_len, vocab_size] depending on the path. Handle both.
        tllm_last_pred = (
            tllm_logits[-1].argmax(dim=-1) if tllm_logits.dim() > 1 else tllm_logits.argmax(dim=-1)
        )
        assert tllm_last_pred == hf_last_pred, (
            f"[{model_name}] TLLM predicted {tllm_last_pred} vs HF predicted {hf_last_pred}"
        )
