# PR #172: active paths and fallback diagnosis

This supplements, rather than replaces, the [complete unfiltered screen](https://github.com/xijiaat/humming/tree/f13bc5aa173398d01f1ebeeb10427129f37c6ec0/benchmark-results/pr172-main-64ab616c).
Base: main `64ab616cc26e6d5e4000a8fc3022814d2abd90bb`.
Candidate: that main plus the four-header #172 patch from
`2e8e8972e8558ebc5cfa2e6f4f1d478c91d50575`. Both contain #171; neither contains #173.
No heuristic, tile, serving configuration or PR code was changed for this supplement.

## Main figures: code-applicable cells only

Selection uses native tactic and compile-time applicability, **not the timing sign**.
All 98 TP/model cells are in `all_cells_with_scope.csv`. The main figures retain
all 55 active cells, including small gains, and exclude all 43 fallback cells,
including positive ones. Missing points are not zero gain and should not be
interpolated as measured results. Original values are unchanged.

For the SM90 BF16/MXFP4 E8M0 group32 cases in this dataset, the scale-reuse gate
requires WGMMA and WarpN64, aligned K32, FP32 accumulation, no zero point,
packed-K, fused-E8M0, integer scale or warp specialization. The other two new
paths have stricter conditions and do not admit any extra cells in this dataset.
`active_scope.py` records the native interval selected by actual routed rows:
`lower < routed_rows <= upper`. It checks the shared format constraints before
classifying by WGMMA/WarpN. This is not a general-purpose gate implementation for
other Humming formats.

| Suite | Active / all measured cells | Observed latency reduction on active cells |
| --- | ---: | ---: |
| V4.1 TP4/TP8/TP4+EP4/TP8+EP8 | 32 / 56 | 1.41-5.07% |
| GPT-OSS-20B/120B and Qwen3-30B-A3B-MXFP4 | 23 / 42 | 0.75-4.19% |

These are descriptive timing ranges, not confidence bounds. Kernel dimensions
match the checkpoint/rank mapping; weights and routing are synthetic. There is
no new full-model performance or quality claim. See the original artifact for
raw ABBA timings, mapping, histograms, exact compiler/kernel source and tests.

## Why the previous graphs had small negative cells

All 20 negative TP/model cells use a fallback path (WarpN32 or MMA rather than
WGMMA). A new same-process diagnostic recompiled both revisions with the exact
native tactic and identical input/weight/output/lock allocations:

- All 20 have identical disassembled SASS instruction hashes, register count,
  static shared memory, local memory, and maximum threads per block.
- Warm CUDA Graph paired median differences range from **-0.079% to +0.111%**.
- Native paired median differences range from **-0.330% to +0.469%**.
- Graph A/A medians range from **-0.051% to +0.068%**.
- The two active TP4+EP4 M32 controls retain native paired median gains of
  **4.26% W13 / 4.80% W2**, and warm-graph gains of **5.19% / 4.97%**.

This does not reproduce a stable code-caused fallback regression. It does not
prove every future shape cannot regress. The diagnosis used 12 balanced,
randomized AB/BA native pairs, 60 graph pairs and 60 graph A/A pairs per cell;
JIT was excluded. Native `do_bench` uses warmup25/rep100; each graph replays16
calls. These diagnostic timings use a different method from the original
screen and are not substituted into its graphs.

Full cubin hashes can differ due to non-instruction content even when the SASS
hash matches. ELF text-section hashes were NOT_MEASURED (pyelftools unavailable).
The raw cubins and disassembly remain in the local experiment archive.
Stream-K output comparisons show bitwise variability in both A/B and A/A for
some cases; no nonfinite values occurred. This diagnostic is not an independent
numerical-reference qualification; existing Stream-K limitations still apply.

Files: `fallback_diagnosis.csv`, `paired_samples.json`, `diagnosis_cases.json`,
and `paired_diagnosis.py`. Reproduce by mounting the two exact source snapshots
at `/sources/base172` and `/sources/post172`, the cases as `/work/cases.json`,
setting `PYTHONPATH=/sources/base172`, and running the script. Use distinct
working/JIT directories, compatible CUDA tools including `nvdisasm`, and an
isolated SM90 GPU. Do not run profilers or this benchmark on serving GPUs.

## Rebuild the scoped figures

Place the original artifact's `tp_comparison.csv` and `model_comparison.csv`
under `public/` and the original source-run JSON files under
`raw-tp/tp-bench/0-base172.json` and `raw-models/model-bench/0-base172.json`.
Run `diagnosis/active_scope.py` in that campaign layout. It writes a separate
`public-active/` directory and never overwrites the original measurements.

The native/expert sweep remains available in the unfiltered archive as coverage
data. It is not included in the active-only TP/model summary.
