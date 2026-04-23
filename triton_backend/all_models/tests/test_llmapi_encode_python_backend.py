# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock pb_utils before the backend module is imported.
sys.modules["triton_python_backend_utils"] = MagicMock()

# Imported via PYTHONPATH=all_models/llmapi_encode/tensorrt_llm/1
from model import TritonPythonModel  # noqa: E402

# ------------------- Mock Triton objects -------------------


@dataclass
class MockTritonTensor:
    _name: str
    _tensor: np.ndarray

    def name(self) -> str:
        return self._name

    def as_numpy(self) -> np.ndarray:
        return self._tensor


@dataclass
class MockTritonError:
    message: str


class MockTritonResponse:
    def __init__(
        self,
        output_tensors: Optional[List[MockTritonTensor]] = None,
        error: Optional[MockTritonError] = None,
    ):
        self.tensors: Dict[str, MockTritonTensor] = {}
        for t in output_tensors or []:
            self.tensors[t.name()] = t
        self.error = error

    def has_error(self) -> bool:
        return self.error is not None


class MockTritonRequest:
    def __init__(self, tensors: Dict[str, np.ndarray]):
        self._tensors = {name: MockTritonTensor(name, arr) for name, arr in tensors.items()}

    def get_input_tensor_by_name(self, name: str) -> Optional[MockTritonTensor]:
        return self._tensors.get(name)


def _mock_get_input_tensor_by_name(request: MockTritonRequest, name: str):
    return request.get_input_tensor_by_name(name)


@pytest.fixture(autouse=True)
def apply_patches():
    patchers = [
        patch("model.pb_utils.Tensor", new=MockTritonTensor),
        patch("model.pb_utils.InferenceResponse", new=MockTritonResponse),
        patch("model.pb_utils.TritonError", new=MockTritonError),
        patch("model.pb_utils.get_input_tensor_by_name", new=_mock_get_input_tensor_by_name),
    ]
    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()


# ------------------- Fake LLM output -------------------


@dataclass
class FakeEncoderOutput:
    """Stand-in for tensorrt_llm.llmapi.llm.EncoderOutput.

    Only .logits is accessed by execute(); a CPU torch tensor makes
    .float().numpy() behave like the real object.
    """

    logits: torch.Tensor


NUM_CLASSES = 2
_FAKE_TOKEN_IDS = [101, 200, 102]


def build_model(max_seq_len: int = 128) -> TritonPythonModel:
    """Instantiate a TritonPythonModel with mocked LLM and tokenizer."""
    model = TritonPythonModel()
    model.logger = MagicMock()
    model.max_seq_len = max_seq_len

    # Tokenizer: one token_ids list per input text.
    def fake_tokenize(texts, **_kwargs):
        return {"input_ids": [list(_FAKE_TOKEN_IDS) for _ in texts]}

    model.tokenizer = MagicMock(side_effect=fake_tokenize)

    # LLM: one EncoderOutput per input_ids list. argmax is always class 1.
    def fake_encode(input_ids):
        return [FakeEncoderOutput(logits=torch.tensor([0.1, 0.9])) for _ in input_ids]

    model.llm = MagicMock()
    model.llm.encode.side_effect = fake_encode
    return model


def _request_with_text(text: str) -> MockTritonRequest:
    # Shape (1, 1) — matches dims: [1] with max_batch_size > 0.
    return MockTritonRequest(
        {
            "text_input": np.array([[text.encode("utf-8")]], dtype=object),
        }
    )


def _request_missing_text() -> MockTritonRequest:
    return MockTritonRequest({})


# ------------------- Tests -------------------


def test_execute_happy_path():
    model = build_model()
    requests = [_request_with_text("hello"), _request_with_text("world")]
    responses = model.execute(requests)

    assert len(responses) == 2
    for r in responses:
        assert not r.has_error()
        assert set(r.tensors) == {"logits", "predictions"}
        assert r.tensors["logits"].as_numpy().shape == (1, NUM_CLASSES)
        assert r.tensors["predictions"].as_numpy().shape == (1, 1)
        # logits[0, 1] > logits[0, 0] → argmax = 1
        assert r.tensors["predictions"].as_numpy()[0, 0] == 1


def test_execute_missing_text_input():
    model = build_model()
    responses = model.execute([_request_missing_text()])

    assert len(responses) == 1
    assert responses[0].has_error()
    assert "missing" in responses[0].error.message.lower()
    # encode() must not be called when every request was malformed.
    model.llm.encode.assert_not_called()


def test_execute_partial_failure():
    """A bad request in the middle must not poison the valid ones."""
    model = build_model()
    requests = [
        _request_with_text("good-1"),
        _request_missing_text(),
        _request_with_text("good-2"),
    ]
    responses = model.execute(requests)

    assert len(responses) == 3
    assert not responses[0].has_error()
    assert responses[1].has_error()
    assert not responses[2].has_error()

    for i in (0, 2):
        assert "logits" in responses[i].tensors
        assert responses[i].tensors["logits"].as_numpy().shape == (1, NUM_CLASSES)

    # encode() called exactly once, with only the two valid input_ids lists.
    model.llm.encode.assert_called_once()
    input_ids_arg = model.llm.encode.call_args[0][0]
    assert len(input_ids_arg) == 2
