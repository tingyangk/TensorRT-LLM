''' Version 1 '''
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, TorchCompileConfig)
from tensorrt_llm.sampling_params import SamplingParams
import torch

def main():
    # Cuda graph config
    cuda_graph_config = CudaGraphConfig(
        max_batch_size=64,
        enable_padding=True,
        num_tokens=[8, 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
        max_seq_len=512,
    )

    torch_compile_config=TorchCompileConfig(
        enable_fullgraph=True,
        enable_inductor=True,
        enable_piecewise_cuda_graph=False,
    )

    # backend default is pytorch
    # disable_overlap_scheduler=True is required for bert model
    # baseline:
    # llm = LLM(
    #     model="textattack/bert-base-uncased-yelp-polarity",
    #     disable_overlap_scheduler=True,
    #     # torch_compile_config=torch_compile_config,
    #     # cuda_graph_config=cuda_graph_config,
    #     cuda_graph_config=None,
    # )
    # optimized:
    llm = LLM(
        model="textattack/bert-base-uncased-yelp-polarity",
        disable_overlap_scheduler=True,
        torch_compile_config=torch_compile_config,
        cuda_graph_config=cuda_graph_config,
        # cuda_graph_config=None,
    )

    sampling_param = SamplingParams(return_context_logits=True)

    # Sample prompts.
    # prompts = [
    #     "This movie was absolutely fantastic!",
    #     "This movie was absolutely terrible!",
    #     "This movie was just okay."
    # ]

    # Around 10 tokens sequences
    prompts = [
        "The quick brown fox jumps over the lazy dog",
        "She walked to the park during the sunny afternoon.",
        "Learning new skills can greatly improve your future opportunities.",
        "Please remember to submit your assignment before the deadline.",
        "Technology evolves quickly, changing how society works every year.",
        "My favorite meal is pasta with a spicy tomato sauce.",
        "Reading books is a great way to expand your knowledge.",
        "The weather forecast predicts rain throughout the coming week.",
        "Healthy habits are built through daily practice and dedication.",
        "Our team will meet tomorrow to finalize the project details."# * 20 # Add a long string
    ]

    # Warmup
    for _ in range(3):
        llm.generate(prompts, sampling_params=sampling_param)

    # Generate
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate(prompts, sampling_params=sampling_param)
    torch.cuda.cudart().cudaProfilerStop()
    
    for output in outputs:
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()[0, :]

        prediction = torch.argmax(tllm_logit, dim=-1)
        probability = torch.softmax(tllm_logit, dim=-1)
        # 0 = negative review
        # 1 = positive review
        sentiment = "positive" if prediction.item() == 1 else "negative"
        print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}, Prediction: {prediction}, Probability: {probability}, Sentiment: {sentiment}")
        # print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}, Prediction: {prediction}, Probability: {probability}, Sentiment: {sentiment}")
        # print(f"Prompt: {prompt!r}, Prediction: {prediction}, Sentiment: {sentiment}")


if __name__ == '__main__':
    main()
