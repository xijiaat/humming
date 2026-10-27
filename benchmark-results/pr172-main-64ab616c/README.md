# Humming #172: latest-main comparison

These plots compare main `64ab616cc26e6d5e4000a8fc3022814d2abd90bb`
with that same main plus the four kernel-header changes from #172 at
`2e8e8972e8558ebc5cfa2e6f4f1d478c91d50575`.
The candidate is a test composition, not an assertion that the PR branch
already has this new base. #171 is present in both arms.

## Plots and data

- `figures/native_latency.png`: Dense controls and Grouped E96.
- `figures/expert_latency.png`: Grouped E8/E32/E96/E256, top-k 6.
- `figures/expert_reduction.png`: all measured expert-sweep reductions,
  including zero and negative points.
- `figures/tp_latency.png` and `figures/tp_reduction.png`: per-rank TP4/TP8
  kernel shapes, with and without EP. These are not multi-GPU serving runs.
- `measurements.csv`: individual timing passes, in microseconds.
- `comparison.csv`: arithmetic means and per-arm min/max.
- `tp_measurements.csv`, `tp_comparison.csv`, `tp_mapping.csv` and
  `tp_routing.json`: per-rank times, actual padding, and expert token histograms.
- `figures/model_latency.png`, `figures/model_reduction.png` and the
  `model_*.csv/json` files: GPT-OSS-20B/120B and Qwen3-30B-A3B MXFP4
  checkpoint-shaped GEMMs, not full-model serving or quality tests.

The reduction is `100 * (1 - mean(candidate) / mean(main))`.
Positive means lower latency. Bands are observed min/max, NOT confidence
intervals. The matrices are small screens, not all-shape guarantees.

## Comparison with earlier numbers

The earlier 8.11-10.20% routed-expert chain reduction compared pre-#171 main
with #171 + #172, using fixed tiles and CUDA Graph timing. The earlier
2.56-4.52% model-throughput gain also included #171. Those historical results
are not contradicted or replaced by these incremental measurements.

Here #171 is already in both arms; #172 alone is added to a newer main.
The primary native-tactic E96 screen shows W13 reductions of 1.58-4.85%
and W2 reductions of 1.40-3.49% at M8/32/128 (M1: 0.54%). Dense controls
are near noise and do not establish a stable gain. Pure TP4/TP8 W13 shapes
improve by 1.37-4.19%; their narrower W2 shapes are essentially unchanged.
The expert and TP extensions contain negative points; none are omitted.
Do not sum percentages or describe the older combined result as this PR's
remaining incremental gain over the latest main.

## Method

- H20, SM90; PyTorch 2.13.0+cu130 / CUDA 13.0.
- W13 N4608/K5120; W2 N5120/K2304.
- BF16 activations and output, MXFP4 weights, E8M0 scales, group32.
- Native upstream heuristics, Stream-K configuration and timing, unchanged.
- Source: `benchmarks/bench_humming.py`, SHA256
  `9552406fc68e74cb8c4583bbfe4c0c5c0e8f1ea0c1219d194c6964fffb8f419a`.
- The upstream function uses `triton.testing.do_bench`, warmup=100, rep=1000.
  JIT preparation is outside timing. This is not a forced CUDA-graph test.
- Seed 2026 is reset per projection/expert-count case; upstream synthetic
  routing is retained. This is not production routing or model quality data.
- Native matrix: M1/8/32/128, Dense and Grouped E96, ABBA twice, four passes/arm.
- Expert matrix: M1/4/8/16/32/64/128, E8/32/96/256, ABBA, two passes/arm.
- Each matrix compares both arms on the same GPU. The matrices ran on
  different idle GPUs on one host, with some overlap. They are not GPU-to-GPU
  comparisons. Clocks were not locked; host/thermal drift is not eliminated.
- Independent source and JIT-cache directories. No tactic or tolerance changes.
- M means input token rows, not serving concurrency. Top-k remains six while
  expert count changes, so average token density per expert also changes.
- Estimated bandwidth/TOPS from the upstream benchmark are not hardware
  counter measurements and are not used in these plots.

### TP/EP-shaped kernel extension

The checkpoint architecture was read directly: hidden=5120, intermediate=2304,
384 routed experts, top-k=6. The vLLM mapping was checked against
`FusedMoEParallelConfig` and the Humming MoE adapter. With EP, experts are
partitioned while each expert stays whole; without EP, intermediate width is
partitioned and all experts remain local.

| Layout | Local experts | W13 logical N/K | W2 logical N/K |
| --- | ---: | --- | --- |
| TP4 | 384 | 1152 / 5120 | 5120 / 576 |
| TP8 | 384 | 576 / 5120 | 5120 / 288 |
| TP4 + EP4 | 96 | 4608 / 5120 | 5120 / 2304 |
| TP8 + EP8 | 48 | 4608 / 5120 | 5120 / 2304 |

This extension uses a separate driver, not an unmodified call to the upstream
benchmark. It retains native heuristic selection and the same do_bench timer,
and requests N-to-256 / K-to-128 padding as in the inspected vLLM adapter.
The resulting Humming config is recorded rather than assumed.

For each global M, a seed-fixed synthetic top-6 routing over all 384 experts
is generated. For EP, the rank with the most routed rows is measured, using
its actual local histogram. For pure TP all routed rows remain local.
Histograms are checked identical between arms. Busiest by token count does
not prove slowest by latency. No average-M approximation is substituted for
the per-expert counts. Inputs/weights are synthetic, not a checkpoint replay.

The driver tests W13/W2 independently, excluding routing execution, activation,
combine, attention, all-to-all/all-reduce, scheduler and DSpark. It therefore
cannot establish TP4/TP8 model throughput or critical-path speedup. Its ABBA
screen has two passes/arm, with the same timing-only correctness limitation.

### Additional model shapes

The vLLM Humming GSM8K configuration list at
[`c3257547`](https://github.com/vllm-project/vllm/blob/c325754771a123330e55928094c8049d7b997db2/tests/evals/gsm8k/configs/humming/config.txt)
includes GPT-OSS-20B and a Qwen3-30B-A3B MXFP4 checkpoint. GPT-OSS-120B is
an additional same-architecture expert-count control, not an entry in that
specific configuration list. This evidence establishes integration/test
coverage, not deployment popularity or a claim that every vLLM model uses
Humming. The current SM90 priority helper puts Humming ahead of Marlin,
not ahead of every possible backend.

| Checkpoint shape (TP1) | Hidden | Expert intermediate | Experts | Top-k | Bias |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-OSS-20B | 2880 | 2880 | 32 | 4 | yes |
| GPT-OSS-120B | 2880 | 2880 | 128 | 4 | yes |
| Qwen3-30B-A3B-MXFP4A16 | 2048 | 768 | 128 | 8 | no |

The independent driver uses BF16 activations, MXFP4/E8M0 group32 weights,
the adapter's N256/K128 padding, native tactics, and seed-fixed synthetic
top-k routing with actual histograms. Each W13/W2 is measured independently
with ABBA, two passes/arm, M1/4/8/16/32/64/128. GPT-OSS bias is retained, so
the bias-disabled matrix-store path is intentionally not eligible there.
No model weights are loaded and no model accuracy claim is made. These are
supported-model-shaped kernel measurements, not results for all Humming
formats such as FP8, NVFP4 or GPTQ/AWQ.

| Shape | Median W13 reduction | Median W2 reduction |
| --- | ---: | ---: |
| GPT-OSS-20B | 2.58% | 0.90% |
| GPT-OSS-120B | 2.85% | 2.96% |
| Qwen3-30B-A3B-MXFP4 | 0.80% | 0.40% |

These are medians across the seven measured M values, not traffic-weighted
model speedups. Some individual cells are negative or effectively unchanged.

## Correctness boundary

The focused `test_mxfp4_bf16_wgmma_dataflow` regression passed seven
parametrizations on each arm (five M values per parametrization). It uses
the existing numerical criteria, including the existing BF16 Stream-K
outlier policy. It is NOT a numerical qualification of every native-tactic
cell in the timing matrices. Previously observed native Stream-K reference
failures remain a limitation; this experiment does not resolve them.

## Reproduction

Use a compatible H20/CUDA/PyTorch environment. Prepare separate source trees
for the pinned main and that main with `pr172-kernels.patch` applied. Check
that only the four intended kernel headers differ. Do not add #171 again.
Mount the two trees at `/sources/base172` and `/sources/post172`, and use a
separate writable `/work` for each matrix. Native/expert drivers invoke the
upstream benchmark function. TP/model drivers construct the mapped shapes
independently while retaining native heuristics and the same timer.

```bash
export OMP_NUM_THREADS=2
export HUMMING_MAX_PARALLEL_BUILD_WORKERS=2
python /work/native_benchmark.py
# In a separate work directory:
python /work/expert_benchmark.py
# In another separate work directory:
python /work/tp_benchmark.py
# In another separate work directory:
python /work/model_benchmark.py
```

The drivers set per-arm `PYTHONPATH`, `HUMMING_CACHE_DIR`, `HUMMING_TMP_DIR`
and `TRITON_CACHE_DIR`, and fail on any timed subprocess failure. The original
image is not distributed with this artifact. `provenance.json` and the patch
record source identity independently of that image.
