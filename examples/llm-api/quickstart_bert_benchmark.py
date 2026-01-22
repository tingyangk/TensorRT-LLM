#!/usr/bin/env python3
"""
BERT Performance Benchmark Script using TensorRT-LLM

Supports benchmarking with different optimization modes:
  - baseline:       No optimizations (eager execution)
  - torch-compile:  torch.compile with inductor backend
  - cuda-graph:     Encoder CUDA graphs only
  - optimized:      Both torch.compile and encoder CUDA graphs
  - all:            Run all modes sequentially and compare
"""

import argparse
import time
import random
import numpy as np
import torch
import warnings
import os
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, TorchCompileConfig

MODES = ['baseline', 'torch-compile', 'cuda-graph', 'optimized', 'all']


class LatencyPercentiles:
    """Track and calculate latency percentiles."""

    def __init__(self):
        self.timings = []

    def add_timing(self, time_ms):
        self.timings.append(time_ms)

    def calculate_percentiles(self):
        if not self.timings:
            return {}
        sorted_timings = np.sort(self.timings)
        return {
            'min': np.min(sorted_timings),
            'p50': np.percentile(sorted_timings, 50),
            'p90': np.percentile(sorted_timings, 90),
            'p95': np.percentile(sorted_timings, 95),
            'p99': np.percentile(sorted_timings, 99),
            'max': np.max(sorted_timings),
            'mean': np.mean(sorted_timings),
            'std': np.std(sorted_timings),
        }

    def print_stats(self, header=None):
        stats = self.calculate_percentiles()
        if not stats:
            print("No timings recorded")
            return
        if header:
            print(f"\n{'='*60}")
            print(header)
        print(f"{'='*60}")
        print(f"Samples: {len(self.timings)}")
        print(f"Min:     {stats['min']:.3f} ms")
        print(f"P50:     {stats['p50']:.3f} ms")
        print(f"P90:     {stats['p90']:.3f} ms")
        print(f"P95:     {stats['p95']:.3f} ms")
        print(f"P99:     {stats['p99']:.3f} ms")
        print(f"Max:     {stats['max']:.3f} ms")
        print(f"Mean:    {stats['mean']:.3f} ms")
        print(f"Std Dev: {stats['std']:.3f} ms")
        print(f"{'='*60}")


def suppress_warnings(level='partial'):
    if level == 'none':
        return
    warnings.filterwarnings("ignore")
    if level == 'all':
        os.environ['TLLM_LOG_LEVEL'] = 'ERROR'


def generate_random_prompts(num_queries, target_tokens=10, seed=0):
    random.seed(seed)
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work",
        "first", "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "was", "are", "been", "has", "had",
        "were", "said", "did", "getting", "made", "find", "where", "much", "too",
        "very", "still", "being", "going", "why", "before", "never", "here", "more",
        "out", "those", "down", "should", "call", "world",
    ]
    prompts = []
    for _ in range(num_queries):
        num_words = max(5, target_tokens + random.randint(-2, 2))
        words = random.choices(common_words, k=num_words)
        prompt = ' '.join(words)
        prompt = prompt[0].upper() + prompt[1:] + '.'
        prompts.append(prompt)
    return prompts


def build_llm_kwargs(mode, args):
    """Build LLM constructor kwargs for a given optimization mode."""
    kwargs = {
        'model': args.model,
        'disable_overlap_scheduler': True,
    }

    use_torch_compile = mode in ('torch-compile', 'optimized')
    use_cuda_graph = mode in ('cuda-graph', 'optimized')

    if use_torch_compile:
        kwargs['torch_compile_config'] = TorchCompileConfig(
            enable_fullgraph=True,
            enable_inductor=True,
            enable_piecewise_cuda_graph=False,
        )

    if use_cuda_graph:
        kwargs['cuda_graph_config'] = CudaGraphConfig(
            max_batch_size=args.cuda_graph_max_batch_size,
            enable_padding=True,
            num_tokens=args.cuda_graph_num_tokens,
            max_seq_len=args.cuda_graph_max_seq_len,
        )
    else:
        kwargs['cuda_graph_config'] = None

    return kwargs


def run_benchmark(mode, args, prompts, batched_prompts):
    """Run a single benchmark for the given mode. Returns LatencyPercentiles."""
    print(f"\n{'#'*60}")
    print(f"  MODE: {mode}")
    print(f"{'#'*60}")

    llm_kwargs = build_llm_kwargs(mode, args)

    tc = llm_kwargs.get('torch_compile_config')
    cg = llm_kwargs.get('cuda_graph_config')
    print(f"  torch_compile_config: {tc}")
    print(f"  cuda_graph_config:    {cg}")

    print("\n  Initializing model...")
    # llm = LLM(**llm_kwargs)
    llm = LLM(**llm_kwargs, load_format='dummy')

    sampling_params = SamplingParams(return_context_logits=True)

    # Warmup
    print(f"  Running {args.warmup_steps} warmup iterations...")
    warmup_batch = batched_prompts[0]
    for i in range(args.warmup_steps):
        llm.generate(warmup_batch, sampling_params=sampling_params, use_tqdm=False)
        if (i + 1) % 10 == 0:
            print(f"    Warmup {i + 1}/{args.warmup_steps}")

    # Measurement
    print("  Starting measurement...")
    if args.use_profiling:
        torch.cuda.cudart().cudaProfilerStart()

    tracker = LatencyPercentiles()
    for idx, batch in enumerate(batched_prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(batch, sampling_params=sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        tracker.add_timing(latency_ms)

        if (idx + 1) % 10 == 0 or args.verbose:
            print(f"    Iter {idx + 1}/{len(batched_prompts)}: {latency_ms:.3f} ms")

        if args.verbose:
            for output in outputs:
                logit = output.context_logits.cpu()[0, :]
                sentiment = "positive" if torch.argmax(logit, dim=-1).item() == 1 else "negative"
                print(f"      {output.prompt[:50]}... -> {sentiment}")

    if args.use_profiling:
        torch.cuda.cudart().cudaProfilerStop()

    tracker.print_stats(header=f"LATENCY STATISTICS — {mode} (ms)")

    stats = tracker.calculate_percentiles()
    if stats:
        total_time_s = sum(tracker.timings) / 1000.0
        throughput = len(prompts) / total_time_s
        print(f"  Throughput: {throughput:.2f} prompts/sec")
        print(f"  Total time: {sum(tracker.timings):.2f} ms")

    # Explicitly delete to free GPU memory before next mode
    del llm
    torch.cuda.empty_cache()

    return tracker


def print_comparison(results):
    """Print a side-by-side comparison table of all modes."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    header = f"{'Metric':<12}"
    for mode in results:
        header += f"  {mode:>16}"
    print(header)
    print("-" * len(header))

    metrics = ['min', 'p50', 'p90', 'p95', 'p99', 'max', 'mean', 'std']
    for metric in metrics:
        row = f"{metric:<12}"
        for mode, tracker in results.items():
            stats = tracker.calculate_percentiles()
            row += f"  {stats.get(metric, 0):>13.3f} ms"
        print(row)

    # Throughput row
    row = f"{'throughput':<12}"
    for mode, tracker in results.items():
        total_time_s = sum(tracker.timings) / 1000.0
        total_prompts = len(tracker.timings)  # each timing = one batch
        # We don't know total prompts here directly, so compute from timings
        if total_time_s > 0:
            stats = tracker.calculate_percentiles()
            row += f"  {1000.0 / stats['mean']:>10.2f} b/sec"
        else:
            row += f"  {'N/A':>13}"
    print(row)

    # Speedup relative to baseline
    if 'baseline' in results:
        baseline_mean = results['baseline'].calculate_percentiles().get('mean', 0)
        if baseline_mean > 0:
            print()
            row = f"{'speedup':<12}"
            for mode, tracker in results.items():
                mean = tracker.calculate_percentiles().get('mean', 0)
                speedup = baseline_mean / mean if mean > 0 else 0
                row += f"  {speedup:>14.2f}x"
            print(row)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='BERT Performance Benchmark using TensorRT-LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no optimizations)
  python quickstart_bert_benchmark.py --mode baseline

  # Encoder CUDA graphs only
  python quickstart_bert_benchmark.py --mode cuda-graph

  # torch.compile + encoder CUDA graphs
  python quickstart_bert_benchmark.py --mode optimized

  # Run all modes and compare
  python quickstart_bert_benchmark.py --mode all --num-queries 200 --batch-size 8
""")
    parser.add_argument('--mode', choices=MODES, default='all',
                        help='Optimization mode (default: all)')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='Number of queries to execute (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--target-tokens', type=int, default=10,
                        help='Target number of tokens per prompt (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--model', type=str,
                        default='textattack/bert-base-uncased-yelp-polarity',
                        help='Model name or path')
    parser.add_argument('--use-profiling', action='store_true',
                        help='Enable CUDA profiling')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output for each query')
    parser.add_argument('--suppress-warnings', choices=['none', 'partial', 'all'],
                        default='partial',
                        help='Level of warning suppression (default: partial)')

    # Encoder CUDA graph args
    parser.add_argument('--cuda-graph-max-batch-size', type=int, default=64,
                        help='Max batch size for encoder CUDA graphs (default: 64)')
    parser.add_argument('--cuda-graph-num-tokens', type=int, nargs='+',
                        default=[8, 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
                        help='Token counts for encoder CUDA graph capture')
    parser.add_argument('--cuda-graph-max-seq-len', type=int, default=512,
                        help='Max sequence length for encoder CUDA graphs (default: 512)')

    args = parser.parse_args()

    suppress_warnings(args.suppress_warnings)

    print("=" * 60)
    print("BERT PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Model:             {args.model}")
    print(f"Mode:              {args.mode}")
    print(f"Num queries:       {args.num_queries}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Warmup steps:      {args.warmup_steps}")
    print(f"Target tokens:     {args.target_tokens}")
    print(f"Seed:              {args.seed}")
    print("=" * 60)

    # Generate prompts (same across all modes for fair comparison)
    print(f"\nGenerating {args.num_queries} random prompts...")
    prompts = generate_random_prompts(
        args.num_queries, target_tokens=args.target_tokens, seed=args.seed)

    if args.verbose and prompts:
        print("\nSample prompts:")
        for i in range(min(5, len(prompts))):
            print(f"  [{i}]: {prompts[i]}")

    batched_prompts = [
        prompts[i:i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]

    # print(f"Batched prompts: {batched_prompts}")

    modes_to_run = MODES[:-1] if args.mode == 'all' else [args.mode]

    results = {}
    for mode in modes_to_run:
        tracker = run_benchmark(mode, args, prompts, batched_prompts)
        results[mode] = tracker

    if len(results) > 1:
        print_comparison(results)


if __name__ == '__main__':
    main()
