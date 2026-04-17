# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi.llm import EncoderOutput

# isort: off
from .test_llm import get_model_path

# isort: on

BERT_MODEL_PATH = "bert/bert-base-uncased-yelp-polarity"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.fixture(scope="module")
def bert_encode_llm():
    """Create an LLM with encode_only=True for BERT, shared across tests."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    llm = LLM(model=model_dir, encode_only=True)
    yield llm
    llm.shutdown()


# --------------------------------------------------------------------------- #
# Basic encode() functionality
# --------------------------------------------------------------------------- #


def test_encode_single_string(bert_encode_llm):
    """encode() with a single string returns a single EncoderOutput."""
    result = bert_encode_llm.encode("Hello, my name is")

    assert isinstance(result, EncoderOutput)
    assert isinstance(result.logits, torch.Tensor)
    assert result.logits.dim() == 1  # [num_classes] for classification
    assert result.logits.shape[0] == 2  # yelp-polarity has 2 classes
    assert result.prompt == "Hello, my name is"
    assert isinstance(result.prompt_token_ids, list)
    assert len(result.prompt_token_ids) > 0


def test_encode_batch(bert_encode_llm):
    """encode() with a list of strings returns a list of EncoderOutput."""
    results = bert_encode_llm.encode(PROMPTS)

    assert isinstance(results, list)
    assert len(results) == len(PROMPTS)
    for i, result in enumerate(results):
        assert isinstance(result, EncoderOutput)
        assert result.logits.shape == (2,)  # 2 classes
        assert result.prompt == PROMPTS[i]


def test_encode_token_ids(bert_encode_llm):
    """encode() accepts pre-tokenized token ID lists."""
    token_ids = [101, 7592, 1010, 2026, 2171, 2003, 102]  # "[CLS] hello, my name is [SEP]"
    result = bert_encode_llm.encode(token_ids)

    assert isinstance(result, EncoderOutput)
    assert result.logits.shape == (2,)
    assert result.prompt is None  # no text prompt when passing token IDs
    assert result.prompt_token_ids == token_ids


def test_encode_mixed_batch(bert_encode_llm):
    """encode() handles mixed input types in a batch."""
    from tensorrt_llm.inputs import TextPrompt, TokensPrompt

    inputs = [
        "Hello world",
        TextPrompt(prompt="Test sentence"),
        TokensPrompt(prompt_token_ids=[101, 7592, 2088, 102]),
    ]
    results = bert_encode_llm.encode(inputs)

    assert len(results) == 3
    assert results[0].prompt == "Hello world"
    assert results[1].prompt == "Test sentence"
    assert results[2].prompt is None


# --------------------------------------------------------------------------- #
# Output correctness — compare with HuggingFace
# --------------------------------------------------------------------------- #


def test_encode_matches_huggingface(bert_encode_llm):
    """encode() logits match HuggingFace BertForSequenceClassification."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = get_model_path(BERT_MODEL_PATH)

    # Get TRT-LLM results
    results = bert_encode_llm.encode(PROMPTS)
    tllm_logits = torch.stack([r.logits.cpu() for r in results])

    # Get HuggingFace results
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    hf_model = hf_model.half().cuda()

    with torch.inference_mode():
        inputs = tokenizer(PROMPTS, return_tensors="pt", padding="longest").to(hf_model.device)
        hf_outputs = hf_model(**inputs)
        hf_logits = hf_outputs.logits.float().cpu()

    torch.testing.assert_close(tllm_logits, hf_logits, rtol=1.5e-2, atol=1.5e-2)


# --------------------------------------------------------------------------- #
# Cross-API guards
# --------------------------------------------------------------------------- #


def test_generate_raises_on_encoder_only(bert_encode_llm):
    """generate() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.generate(PROMPTS)


def test_generate_async_raises_on_encoder_only(bert_encode_llm):
    """generate_async() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.generate_async("Hello")


def test_encode_raises_without_encoder_only():
    """encode() raises RuntimeError on a decoder model (encode_only=False)."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    with LLM(model=model_dir, encode_only=False, disable_overlap_scheduler=True) as llm:
        with pytest.raises(RuntimeError, match="encode_only=True"):
            llm.encode("Hello")


def test_get_stats_raises_on_encoder_only(bert_encode_llm):
    """get_stats() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.get_stats()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_encode_empty_string(bert_encode_llm):
    """encode("") should either raise or produce a valid (empty-ish) result.

    Tokenizing "" with add_special_tokens=True produces [CLS][SEP] (2 tokens),
    so this is actually a valid input for BERT.
    """
    result = bert_encode_llm.encode("")
    assert isinstance(result, EncoderOutput)
    assert result.logits.shape == (2,)


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #


def test_check_health_encoder_only(bert_encode_llm):
    """_check_health() returns True for a live encoder-only LLM."""
    assert bert_encode_llm._check_health() is True
