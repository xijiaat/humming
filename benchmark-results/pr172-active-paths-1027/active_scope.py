#!/usr/bin/env python3
"""Separate code-applicable timings from fallback controls, without sign filtering."""
import csv
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/humming-pr172-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public-active"
MS = [1, 4, 8, 16, 32, 64, 128]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify(layer, tuning):
    # This dataset is SM90 BF16/MXFP4 group32. Within it, either other new
    # path implies this scale-reuse gate; do not generalize to other formats.
    assert layer["sm_version"] == 90
    assert layer["a_dtype"] == layer["c_dtype"] == "bfloat16"
    assert layer["b_dtype"] == "float4e2m1" and layer["bs_dtype"] == "float8e8m0"
    assert layer["weight_scale_type"] == "group" and layer["weight_scale_group_size"] == 32
    assert not any(layer[k] for k in ("has_zero_point", "use_packed_k_layout",
                                    "use_fused_e8m0_scale", "use_int_weight_scale"))
    assert not tuning.get("use_warp_spec", False)
    assert not tuning.get("use_f16_accum", False)
    assert tuning["warp_shape"][2] % 32 == 0 and tuning["block_shape"][2] % 32 == 0
    if tuning["mma_type"] != "wgmma":
        return False, "MMA fallback; no WGMMA optimization"
    if tuning["warp_shape"][1] != 64:
        return False, "WarpN32 fallback; all three PR172 gates disabled"
    return True, "Aligned K32 scale reuse enabled; other gates may also apply"


def draw(suite, rows, groups):
    colors = ["#245B97", "#B85A19", "#23845B", "#984589"]
    for metric in ("reduction", "latency"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        active = [r for r in rows if r["optimization_active"]]
        for ax, shape in zip(axes, ("w13", "w2")):
            for (group, label), color in zip(groups, colors):
                selected = {int(r["m"]): r for r in active if r["group"] == group and r["shape"] == shape}
                if not selected:
                    continue
                if metric == "reduction":
                    ys = [float(selected[m]["reduction_pct"]) if m in selected else float("nan") for m in MS]
                    ax.plot(MS, ys, marker="o", color=color, label=label)
                else:
                    for field, style, arm in (("main_us", "--", "main"), ("pr_us", "-", "+#172")):
                        ys = [float(selected[m][field]) if m in selected else float("nan") for m in MS]
                        ax.plot(MS, ys, marker="o", markersize=4, color=color, linestyle=style, label=label + " " + arm)
            if metric == "reduction":
                ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(MS, [str(m) for m in MS])
            ax.grid(alpha=0.2)
            ax.set_title(shape.upper())
            ax.set_xlabel("Input token rows M (not serving concurrency)")
            ax.set_ylabel("Latency reduction vs main (%)" if metric == "reduction" else "Kernel latency (us)")
        handles, labels = [], []
        for ax in axes:
            h, lab = ax.get_legend_handles_labels()
            for handle, label in zip(h, lab):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.92), frameon=False, fontsize=9)
        title = "V4.1 TP/EP shapes" if suite == "tp" else "Other MXFP4 model shapes"
        fig.suptitle(title + " | #172-active native tactics only", fontsize=14, y=0.99)
        fig.text(0.5, 0.02,
                 f"main 64ab616c vs main+#172; {len(active)}/{len(rows)} cells active. Missing points = fallback, not zero gain.\n"
                 "Selection uses compile-time gates, not timing sign. All active cells retained. H20; per-rank kernels, not full models.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.10, 1, 0.82))
        fig.savefig(PUBLIC / "figures" / (suite + "_active_" + metric + ".png"), dpi=180)
        plt.close(fig)


def main():
    (PUBLIC / "figures").mkdir(parents=True, exist_ok=True)
    all_rows = []
    for suite, group_field, raw_name, groups in (
        ("tp", "mode", "raw-tp/tp-bench/0-base172.json", [("tp4", "TP4"), ("tp8", "TP8"),
          ("tp4_ep4", "TP4+EP4"), ("tp8_ep8", "TP8+EP8")]),
        ("model", "model", "raw-models/model-bench/0-base172.json", [("openai/gpt-oss-20b", "GPT-OSS-20B"),
          ("openai/gpt-oss-120b", "GPT-OSS-120B"), ("nm-testing/Qwen3-30B-A3B-MXFP4A16", "Qwen3-30B-A3B-MXFP4")]),
    ):
        cases = json.loads((ROOT / raw_name).read_text())["cases"]
        rows = read_csv(ROOT / "public" / (suite + "_comparison.csv"))
        records = []
        for row in rows:
            case = next(c for c in cases if c[group_field] == row[group_field] and c["shape"] == row["shape"])
            result = next(r for r in case["results"] if r["shape_m"] == int(row["m"]))
            count = result["local_routed_rows"]
            lower, upper, tuning = next(t for t in case["tuning_config"] if t[0] < count <= t[1])
            active, reason = classify(case["layer_config"], tuning)
            record = dict(suite=suite, group=row[group_field], shape=row["shape"], m=int(row["m"]),
                main_us=float(row["main_us"]), pr_us=float(row["pr_us"]), reduction_pct=float(row["reduction_pct"]),
                optimization_active=active, reason=reason, routed_rows=count,
                interval_lower=lower, interval_upper=upper, mma=tuning["mma_type"],
                block=json.dumps(tuning["block_shape"]), warp=json.dumps(tuning["warp_shape"]),
                stream_k=tuning["use_stream_k"])
            records.append(record)
        draw(suite, records, groups)
        all_rows.extend(records)
    write_csv(PUBLIC / "all_cells_with_scope.csv", all_rows)
    active = [r for r in all_rows if r["optimization_active"]]
    fallback = [r for r in all_rows if not r["optimization_active"]]
    write_csv(PUBLIC / "active_cells.csv", active)
    write_csv(PUBLIC / "fallback_controls.csv", fallback)
    shutil.copy2(__file__, PUBLIC / "active_scope.py")
    for suite in ("tp", "model"):
        kept = [r for r in active if r["suite"] == suite]
        skipped = [r for r in fallback if r["suite"] == suite]
        print(suite, "active", len(kept), "fallback", len(skipped), "negative active", sum(r["reduction_pct"] < 0 for r in kept))
        for group in sorted(set(r["group"] for r in kept)):
            vals = [r["reduction_pct"] for r in kept if r["group"] == group]
            print(group, min(vals), max(vals))


if __name__ == "__main__":
    main()
