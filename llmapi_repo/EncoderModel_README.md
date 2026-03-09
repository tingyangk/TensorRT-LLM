# Serving Encoder Models with TensorRT-LLM and Triton Inference Server

This guide walks through how to deploy encoder-only models (e.g., BERT) on
[Triton Inference Server](https://github.com/triton-inference-server/server)
using TensorRT-LLM's PyTorch backend. All example code lives under
`llmapi_repo/bert/`.

---

## Overview of Examples

The `llmapi_repo/bert/` directory contains three model versions, each
demonstrating a different serving strategy:

| Version | Path | Description |
|---------|------|-------------|
| **1** | `bert/1/` | **LLM API baseline.** Uses TensorRT-LLM's `LLM` API (`generate_async`) with the PyTorch backend. Configuration is in `model.yaml`; no torch.compile or CUDA graph optimizations are enabled (`cuda_graph_config: null`). This is the simplest way to serve an encoder model through Triton. |
| **2** | `bert/2/` | **LLM API optimized for BERT.** Same `LLM` API approach as version 1, but enables torch.compile (`enable_fullgraph: true`, `enable_inductor: true`) and encoder CUDA graphs via `cuda_graph_config` in `model.yaml`. This leverages the built-in LLM API optimizations for higher throughput. |
| **3** | `bert/3/` | **Standalone BERT runner.** Bypasses the LLM API executor loop entirely, calling TensorRT-LLM's `BertForSequenceClassification` model directly with pre-allocated static buffers. Implements its own `BertRunner`, `torch.compile` wrapper, and `EncoderCUDAGraphManager` for maximum control and performance. |

### Which `config.pbtxt` to use

- **`config.pbtxt`** — For the standalone runner (version 3). Uses
  `dynamic_batching` with a `max_batch_size` of 64 and `preferred_batch_size`
  values that match CUDA graph supported sizes. The `version_policy` selects
  only version 3.
- **`config_llmapi.pbtxt`** — For LLM API-based models (versions 1 and 2).
  Contains additional input/output fields for sampling parameters, streaming,
  and performance metrics. The `version_policy` selects versions 1 and 2.

To switch between them, rename or symlink the desired file to `config.pbtxt`
before launching Triton, or maintain both and use Triton's `--model-repository`
with separate directories.

---

## Deep Dive: Deploying with the Standalone Runner (Version 3)

Version 3 bypasses the LLM API overhead, making it the most performant and best
starting point for deploying custom encoder models. This section covers end-to-end deployment.

### 1. Building the Triton Docker Image

Build a Docker image that includes both TensorRT-LLM and Triton Server from the
repository root:

```bash
docker build -f docker/Dockerfile.multi \
  --target tritonrelease \
  --build-arg DEVEL_IMAGE=tritondevel \
  --build-arg TRITON_BASE_TAG=25.12-py3 \
  --build-arg TRT_LLM_VER="1.2.0rc7" \
  --build-arg BUILD_WHEEL_ARGS="--clean --benchmarks --cuda_architectures '80-real;90-real;120-real'" \
  -t my_triton_trtllm .
```

Key build arguments:

| Argument | Purpose |
|----------|---------|
| `--target tritonrelease` | Selects the Triton release stage from the multi-stage Dockerfile. |
| `DEVEL_IMAGE` | Name of the devel image to base on (build the devel stage first, or use `tritondevel`). |
| `TRITON_BASE_TAG` | NGC Triton base image tag (e.g., `25.12-py3`). |
| `TRT_LLM_VER` | TensorRT-LLM version string baked into the wheel. |
| `BUILD_WHEEL_ARGS` | Flags for `build_wheel.py`; set `--cuda_architectures` to match your target GPUs. |

### 2. Repository Layout

Triton expects a specific model repository structure. The minimal layout for
version 3:

```
model_repository/
└── bert/
    ├── config.pbtxt
    └── 3/
        ├── model.py              # Triton Python backend entry point
        ├── bert_runner.py        # Standalone BERT inference runner
        └── optimization_utils.py # torch.compile + CUDA graph manager
```

### 3. Configuration: `config.pbtxt`

The key sections to adjust:

```protobuf
name: "bert"
backend: "python"

max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8, 16, 32, 64 ]
}

version_policy {
    specific {
        versions: [ 3 ]
    }
}

instance_group [
  {
    count: 1
    kind : KIND_CPU
  }
]

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
```

Things to consider:

- **`max_batch_size` and `preferred_batch_size`**: Align these with
  `SUPPORTED_BATCH_SIZES` in `model.py` and `EncoderCUDAGraphManager` to
  ensure CUDA graphs cover all batch sizes that Triton will produce.
- **`input` / `output`**: If your model returns something other than float
  logits (e.g., embeddings), adjust the output name, data type, and dims.
- **`instance_group`**: Set to `KIND_CPU` because the Python backend manages
  GPU placement internally. Increase `count` to run multiple model instances.

### 4. Configuration: `model.py` Constants

At the top of `bert/3/model.py`, adjust these constants for your deployment:

```python
MODEL_NAME = "textattack/bert-base-uncased-yelp-polarity"
DTYPE = torch.float16
MAX_BATCH_SIZE = 64
MAX_NUM_TOKENS = 8192
MAX_SEQ_LEN = 512

ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPH = True
SUPPORTED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SUPPORTED_NUM_TOKENS = [8, 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
SUPPORTED_SEQ_LENS = [8, 16, 32, 64, 128, 256, 512]
```

| Constant | Purpose |
|----------|---------|
| `MODEL_NAME` | HuggingFace model name or local path. Must be compatible with `AutoConfig` and `AutoTokenizer`. |
| `DTYPE` | Model precision. Use `torch.float16` or `torch.bfloat16` for GPU inference. |
| `MAX_BATCH_SIZE` | Upper bound on batch size; should match `max_batch_size` in `config.pbtxt`. |
| `MAX_NUM_TOKENS` | Maximum total tokens across all sequences in a batch (batch_size x avg_seq_len). |
| `MAX_SEQ_LEN` | Maximum sequence length per input (truncation limit). |
| `ENABLE_TORCH_COMPILE` | Set `True` to apply `torch.compile` for kernel fusion. |
| `ENABLE_CUDA_GRAPH` | Set `True` to capture and replay CUDA graphs for reduced launch overhead. |
| `SUPPORTED_BATCH_SIZES` | Batch sizes for CUDA graph capture. Requests are padded to the nearest supported size. |
| `SUPPORTED_NUM_TOKENS` | Token count buckets for CUDA graph keys. |
| `SUPPORTED_SEQ_LENS` | Sequence length buckets for CUDA graph keys. |

### 5. Launching the Server

Inside the Docker container:

```bash
python triton_backend/scripts/launch_triton_server.py \
  --model_repo /path/to/model_repository \
  --no-mpi
```

Or launch Triton directly:

```bash
/opt/tritonserver/bin/tritonserver --model-repository=/path/to/model_repository
```

### 6. Benchmarking with `perf_analyzer` (Optional)

You can use Triton's
[`perf_analyzer`](https://github.com/triton-inference-server/perf_analyzer)
to measure throughput and latency. First, install the Triton client library:

```bash
pip install tritonclient[all]
```

Create an `inputs.json` file with sample data:

```json
{
  "data": [
    {"text_input": ["The food at this restaurant was absolutely delicious"]},
    {"text_input": ["I hated every minute of that boring film"]},
    {"text_input": ["Great service and friendly staff overall"]},
    {"text_input": ["The product broke after just one week"]},
    {"text_input": ["Wonderful experience from start to finish"]},
    {"text_input": ["Terrible quality for such a high price"]},
    {"text_input": ["I would highly recommend this to everyone"]},
    {"text_input": ["The worst customer service I have experienced"]},
    {"text_input": ["Beautiful design and very easy to use"]},
    {"text_input": ["Complete waste of money do not buy"]},
    {"text_input": ["Exceeded all my expectations in every way"]},
    {"text_input": ["The hotel room was dirty and uncomfortable"]},
    {"text_input": ["Amazing value for the price you pay"]},
    {"text_input": ["I regret purchasing this useless product"]},
    {"text_input": ["The staff were incredibly helpful and kind"]},
    {"text_input": ["Disappointed with the overall build quality"]},
    {"text_input": ["This is the best purchase I ever made"]},
    {"text_input": ["Slow shipping and the package arrived damaged"]},
    {"text_input": ["Loved the atmosphere and the live music"]},
    {"text_input": ["Would never go back to that awful place"]}
  ]
}
```

Then run:

```bash
perf_analyzer -m bert \
    -x 3 \
    --input-data inputs.json \
    --concurrency-range 1:16 \
    --percentile=99 \
    --shape text_input:1 \
    -i grpc \
    --collect-metrics
```

| Flag | Purpose |
|------|---------|
| `-m bert` | Model name in the Triton repository. |
| `-x 3` | Selects model version 3 (the standalone runner). |
| `--input-data inputs.json` | Path to the sample inputs file. |
| `--concurrency-range 1:16` | Sweeps concurrency from 1 to 16. |
| `--percentile=99` | Reports p99 latency. |
| `--shape text_input:1` | Declares the shape of the `text_input` tensor. |
| `-i grpc` | Uses gRPC protocol (default port 8001). |
| `--collect-metrics` | Collects Triton server-side metrics. |

---

## Adapting for Other Encoder Models

If you want to serve a model other than `BertForSequenceClassification` (e.g.,
`BertForTokenClassification`, a RoBERTa variant, or a completely different
encoder architecture), you will need to make changes at multiple levels.

### Step 1: Implement the TensorRT-LLM Model

TensorRT-LLM's PyTorch backend requires a model class that uses its optimized
modules (`Attention`, `Linear`, `Embedding`, etc.) and follows the
expected interface for `AttentionMetadata`.

For a full guide on implementing new models, refer to the official documentation:
[Adding a New Model in PyTorch Backend](https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html)

The existing BERT implementation at
`tensorrt_llm/_torch/models/modeling_bert.py` is a good reference. Key points:

- **`BertAttention`** extends `tensorrt_llm._torch.modules.attention.Attention`,
  which handles the attention computation compatible with the PyTorch runtime.
- **`BertEncoderLayer`** extends `tensorrt_llm._torch.modules.decoder_layer.DecoderLayer`
  and wires together attention, MLP, and layer norms.
- **`BertForSequenceClassification`** is the top-level model that combines
  embeddings, encoder layers, pooling, and a classification head. It is
  registered with `@register_auto_model("BertForSequenceClassification")`.
- **`load_weights`** handles mapping HuggingFace checkpoint weight names to
  TensorRT-LLM's internal naming (e.g., fusing Q/K/V projections into
  `qkv_proj`).

If your model has a different architecture (e.g., no pooler, different head,
cross-attention), you will need to implement the corresponding modules
and weight conversion logic.

### Step 2: Modify `bert_runner.py`

The `BertRunner` class in `bert/3/bert_runner.py` handles model initialization,
tokenization, and forward pass orchestration. You will need to adapt:

1. **Model import and construction**: Replace `BertForSequenceClassification`
   with your model class.

   ```python
   # Before
   from tensorrt_llm._torch.models.modeling_bert import BertForSequenceClassification
   self.model = BertForSequenceClassification(model_config)

   # After (example for a token classification model)
   from tensorrt_llm._torch.models.modeling_bert import BertForTokenClassification
   self.model = BertForTokenClassification(model_config)
   ```

2. **HuggingFace config adjustments**: The runner sets `num_key_value_heads`
   on the config object because the BERT HuggingFace config does not include
   it. If your model's config already has this field (or has different
   requirements), adjust accordingly.

   ```python
   hf_config = AutoConfig.from_pretrained(model_name)
   hf_config.torch_dtype = dtype
   hf_config.num_key_value_heads = hf_config.num_attention_heads  # adjust if needed
   ```

3. **Weight loading**: If your model's HuggingFace checkpoint has different
   parameter names, update the weight loading to use the appropriate
   `AutoModel*` class:

   ```python
   hf_model = AutoModelForTokenClassification.from_pretrained(
       model_name, torch_dtype=hf_config.torch_dtype
   )
   ```

4. **Forward pass signature**: If your model requires different inputs (e.g.,
   no `token_type_ids`, or additional `attention_mask`), update
   `_eager_forward` and the CUDA graph static buffers initialization and replay logic in `optimization_utils.py`.

5. **Tokenization**: The `tokenize_and_pack` method produces packed
   (flattened) tensors. If your tokenizer returns different fields or your model
   doesn't use `token_type_ids`, adjust this method.

### Step 3: Modify `model.py` (Triton entry point)

Update `bert/3/model.py` to match your model's inputs and outputs:

- Change `MODEL_NAME` to your model's name or path.
- If your model returns embeddings instead of classification logits, update the
  `execute` method to reshape and return the output accordingly.
- Adjust the output tensor name to match what you defined in `config.pbtxt`.

### Step 4: Update `config.pbtxt`

Adjust the Triton model configuration to match your model's I/O:

- **Inputs**: If your model requires additional inputs beyond `text_input`
  (e.g., pre-tokenized IDs, segment masks), add them as input tensors.
- **Outputs**: Change output tensor names, data types, and dimensions to match
  what your model produces.

### Step 5: Register the Model (for LLM API usage)

> **Note:** This step is **not required** for the standalone runner (version 3).
> Version 3's `bert_runner.py` imports and instantiates the model class
> directly — it never goes through the LLM API's auto-discovery registry.
> You only need model registration if you want your model to work with the
> LLM API (versions 1 and 2), where `LLM(model=...)` looks up the model
> class by the `architectures` field in the HuggingFace config.

To register a model, add the `@register_auto_model` decorator to your model
class. There are two ways to make the registration visible to the LLM API:

- **Core registration:** Place your model file in
  `tensorrt_llm/_torch/models/` and add the import to
  `tensorrt_llm/_torch/models/__init__.py`. The model will be available
  automatically to all TensorRT-LLM users without any extra imports.
- **Out-of-tree registration:** Keep your model file anywhere you want and
  `import` it in your script **before** calling `LLM(...)`. The
  `@register_auto_model` decorator fires on import, so the LLM API can
  discover it. See the
  [out-of-tree model example](https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html#out-of-tree-models)
  for a working reference.

Both approaches give you the full LLM API and all TensorRT-LLM optimized
kernels. The only difference is where the code lives and whether an explicit
import is needed.

---

## LLM API Versions (1 and 2): Quick Reference

For versions 1 and 2, the model is served through TensorRT-LLM's `LLM` API.
Configuration is split between `config_llmapi.pbtxt` (Triton settings) and
`model.yaml` (TRT-LLM engine arguments).

### `model.yaml` Key Fields

```yaml
model: textattack/bert-base-uncased-yelp-polarity
backend: "pytorch"
disable_overlap_scheduler: true  # Required for encoder-only models
```

For version 2, additionally:

```yaml
torch_compile_config:
  enable_fullgraph: true
  enable_inductor: true
  enable_piecewise_cuda_graph: false

cuda_graph_config:
  max_batch_size: 64
  enable_padding: true
  num_tokens: [8, 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
  max_seq_len: 512
```

To switch from BERT to a different encoder model in versions 1/2:
1. Change the `model` field in `model.yaml` to your model name.
2. Make sure the model is registered in TensorRT-LLM (see Step 5 above).
3. Update the `_create_response` method in `model.py` if your model's output
   format differs from classification logits.
