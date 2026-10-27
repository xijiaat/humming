# PR #172 Stream-K component ablation

Scale-only and full #172 have identical disassembled SASS instruction hashes in 55/55 cases, and identical recorded register/static-shared/local-memory/max-thread attributes in 55/55. All four scale-containing arms have identical SASS in 55/55; matrix-only matches main in 55/55.

| Arm | Native median reduction | Warm-graph median reduction | Warm-graph range across 55 cases |
| --- | ---: | ---: | ---: |
| Main | 0.000% | 0.000% | 0.000% to 0.000% |
| Scale reuse only | 3.881% | 4.361% | 1.222% to 5.462% |
| Matrix-store only | -0.053% | 0.002% | -0.168% to 0.575% |
| Scale + matrix-store | 3.887% | 4.381% | 1.195% to 5.447% |
| Scale + dequant overlap | 3.896% | 4.374% | 1.218% to 5.597% |
| Full #172 | 3.903% | 4.331% | 1.228% to 5.480% |

All 55 code-active cells from the previously published native-tactic screen are
included: 32 V4.1 TP/EP cells and 23 GPT-OSS/Qwen-shaped cells. M is input token
rows, not serving concurrency. The 43 inactive cells were covered separately
in the preceding fallback study; this ablation does not claim all-shape speedups.

Every arm keeps Stream-K enabled, the exact original tactic, inputs, weights,
expert routing, output allocation and launcher. Main is `64ab616c` (includes
#171); full #172 uses the four headers from `2e8e8972`; #173 is absent. Ablations
only disable selected compile-time gates in copied headers. The current
matrix-store and dequant-overlap gates reject Stream-K, so enabling their source
changes does not force those paths to execute. Overlap structurally depends on
scale reuse; the scale+overlap arm is not an independent overlap-only path.

Same-process balanced rotated/reversed six-arm orders: 12 native timing cycles
(`do_bench` warmup15/rep40 ms), 60 warm CUDA Graph cycles (16 kernel calls/replay),
and 60 graph A/A cycles. JIT is excluded, clocks are not locked. The table uses
the median across 55 per-case paired-median reductions, not a workload-weighted
model average. Full per-case timings, A/A spread and native timings are retained;
these ranges are not confidence intervals. Small differences between identical
SASS arms are timing variability, not evidence of additional optimizations.

All measured outputs were finite. Bitwise A/B and A/A variability under
Stream-K is recorded, not hidden. This is attribution/performance testing, not
an independent accuracy qualification, quality test or full-model benchmark.
Static shared-memory attributes exclude dynamic shared memory. Full cubin
hashes may differ despite identical instruction hashes. ELF text-section hashes
remain NOT_MEASURED when pyelftools is unavailable.

## Reproduction

Use an isolated SM90 GPU and a compatible environment (recorded PyTorch/CUDA in
samples.json). Check out Humming main `64ab616cc26e6d5e4000a8fc3022814d2abd90bb`
for the Python package. Extract source-headers.tar.gz first. Mount this artifact
at `/work` (writeable) and its `sources/` at `/sources` (read-only). Set PYTHONPATH to that main checkout, use
fresh HUMMING_CACHE_DIR/HUMMING_TMP_DIR/TRITON_CACHE_DIR, and run
`python /work/run_ablation.py`. The run refuses to overwrite `/work/results`.
cuobjdump and nvdisasm must be available. `paired_diagnosis.py` supplies the
shared compilation/SASS helper. The tested header copies and SHA256 manifest
are supplied, so regeneration with prepare.py is not required.
