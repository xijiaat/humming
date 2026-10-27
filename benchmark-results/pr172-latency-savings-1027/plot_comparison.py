#!/usr/bin/env python3
"""Plot the published measurements as zero-based latency reductions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/humming-pr172-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent
MODELS = (
    ("openai/gpt-oss-20b", "GPT-OSS-20B", "#2366A7"),
    ("openai/gpt-oss-120b", "GPT-OSS-120B", "#A64429"),
    ("nm-testing/Qwen3-30B-A3B-MXFP4A16", "Qwen3-30B-A3B-MXFP4", "#147D6B"),
)
LAYOUTS = (
    ("tp4", "TP4", "#2366A7"),
    ("tp8", "TP8", "#A64429"),
    ("tp4_ep4", "TP4 + EP4", "#147D6B"),
    ("tp8_ep8", "TP8 + EP8", "#7C4A91"),
)
MS = (1, 4, 8, 16, 32, 64, 128)


def draw(rows, groups, suite, title):
    assert len(rows) == len(groups) * 14
    active = [r for r in rows if r["optimization_active"] == "True"]
    assert len(active) == (32 if suite == "tp" else 23)
    for row in rows:
        calculated = 100 * (1 - float(row["pr_us"]) / float(row["main_us"]))
        assert abs(calculated - float(row["reduction_pct"])) < 1e-8
    assert all(r["stream_k"] == "True" for r in active)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#424953", "text.color": "#20272F",
        "xtick.color": "#626B75", "ytick.color": "#424953",
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(18, 3 * len(groups) + 2.8), facecolor="white")
    grid = fig.add_gridspec(len(groups), 4, width_ratios=[1, .64, 1, .64],
                           left=.045, right=.985, bottom=.14, top=.83,
                           wspace=.08, hspace=.60)
    lo, hi = min(float(r["reduction_pct"]) for r in active), max(float(r["reduction_pct"]) for r in active)
    fig.text(.045, .956, title, fontsize=25, weight="bold")
    fig.text(.045, .914, f"{lo:.2f}-{hi:.2f}% lower kernel latency across {len(active)} active cases", fontsize=16)
    fig.text(.045, .881, "H20 / SM90  |  Native tactics, Stream-K ON  |  main 64ab616c vs main + #172", fontsize=11, color="#626B75")

    for i, (model, label, color) in enumerate(groups):
        for j, shape in enumerate(("w13", "w2")):
            ax = fig.add_subplot(grid[i, j * 2])
            table = fig.add_subplot(grid[i, j * 2 + 1])
            table.axis("off")
            transform = blended_transform_factory(table.transAxes, table.transData)
            ax.set_title(f"{label} / {shape.upper()}", loc="left", pad=16, color=color)
            ax.set_ylim(6.65, -.65)
            table.set_ylim(6.65, -.65)
            ax.set_xlim(0, 6.1)
            ax.set_xticks(range(7), [f"{x}%" for x in range(7)])
            ax.set_yticks(range(7), [f"M={m}" for m in MS])
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E6E9ED", linewidth=.8)
            ax.axvline(0, color="#9CA5AF", linewidth=1)
            ax.tick_params(axis="both", length=0, pad=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == len(groups) - 1:
                ax.set_xlabel("Latency reduction vs main (0% baseline)", labelpad=10)
            columns = ((.01, "Main"), (.34, "+#172"), (.70, "Saved"))
            for x, heading in columns:
                table.text(x, 1.08, heading, transform=table.transAxes,
                           fontsize=9.5, color="#626B75", weight="bold")
            for y, m in enumerate(MS):
                row = next(r for r in rows if r["group"] == model and r["shape"] == shape and int(r["m"]) == m)
                if row["optimization_active"] != "True":
                    ax.text(.12, y, "fallback", va="center", color="#88919B", fontsize=10)
                    table.text(.01, y, "not on optimized path", transform=transform,
                               va="center", color="#9AA2AA", fontsize=8.5)
                    continue
                reduction = float(row["reduction_pct"])
                baseline, candidate = float(row["main_us"]), float(row["pr_us"])
                ax.barh(y, reduction, height=.58, color=color, alpha=.93, zorder=3)
                ax.text(reduction + .07, y, f"{reduction:.2f}%", va="center",
                        color=color, fontsize=10.5, weight="bold")
                for (x, _), value in zip(columns, (baseline, candidate, baseline - candidate)):
                    table.text(x, y, f"{value:.2f}", transform=transform, va="center",
                               fontsize=9.3, color=color if x == .70 else "#39424C")

    fig.text(.045, .077, "Bars: 100 x (main - #172) / main; zero-based axes, identical 0-6% tick scale. Table: microseconds (us).", fontsize=10.5)
    fig.text(.045, .051, f"All {len(active)} active cases retained; selection is by compile-time gates, not timing sign. The other {len(rows)-len(active)} cases use fallback paths.", fontsize=10.5, color="#626B75")
    fig.text(.045, .025, "M = input token rows, NOT serving concurrency. Per-rank synthetic kernel benchmarks, NOT full-model throughput. No new measurements.", fontsize=10.5, color="#626B75")
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"{suite}_latency_savings.{ext}", dpi=180)
    plt.close(fig)
    return dict(active_cases=len(active), total_cases=len(rows), reduction_range_pct=[lo, hi])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "comparison_cells.csv")
    args = parser.parse_args()
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 98
    with (ROOT / "comparison_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = {}
    for suite, groups, title in (
        ("tp", LAYOUTS, "DeepSeek-V4.1-Flash: #172 kernel latency savings"),
        ("model", MODELS, "Other MXFP4 models: #172 kernel latency savings"),
    ):
        results[suite] = draw([r for r in rows if r["suite"] == suite], groups, suite, title)
    metadata = dict(suites=results,
                    source_commit="013003a7821e5746fef5ee90d13de7c1b52395a2",
                    source_path="benchmark-results/pr172-active-paths-1027/all_cells_with_scope.csv",
                    csv_sha256=hashlib.sha256((ROOT / "comparison_cells.csv").read_bytes()).hexdigest(),
                    plot_scope="Existing native-tactic measurements only; not new warm-graph ablation data.")
    (ROOT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
