#!/usr/bin/env python3
"""Materialize six compile-time ablations without changing the PR or tactics."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
ARMS = {
    "base172": None,
    "scale_only": (True, False, False),
    "matrix_only": (False, True, False),
    "scale_matrix": (True, True, False),
    "scale_overlap": (True, False, True),
    "post172": (True, True, True),
}


def disable(path, anchor):
    source = path.read_text()
    assert source.count(anchor) == 1, (path, anchor)
    path.write_text(source.replace(anchor, anchor + " false &&"))


def main():
    sources = ROOT / "sources"
    sources.mkdir(exist_ok=False)
    for arm, flags in ARMS.items():
        origin = "base172" if flags is None else "post172"
        target = sources / arm / "humming/include"
        shutil.copytree(CAMPAIGN / "sources" / origin / "humming/include", target)
        if flags is None:
            continue
        scale, matrix, overlap = flags
        wgmma = target / "humming/mma/wgmma.cuh"
        if not scale:
            disable(wgmma, "static constexpr bool kReuseBScaleAcrossFragments =")
        if not overlap:
            disable(wgmma, "static constexpr bool kOverlapBDequant =")
        if not matrix:
            disable(target / "humming/epilogue/smem_writer.cuh", "constexpr bool kMatrixStore =")
    rows = list(csv.DictReader((CAMPAIGN / "public-active/active_cells.csv").open()))
    cases = []
    for row in rows:
        assert row["stream_k"] == "True"
        suite = row["suite"]
        key = "mode" if suite == "tp" else "model"
        raw = "raw-tp/tp-bench/0-base172.json" if suite == "tp" else "raw-models/model-bench/0-base172.json"
        original = json.loads((CAMPAIGN / raw).read_text())["cases"]
        case = next(c for c in original if c[key] == row["group"] and c["shape"] == row["shape"])
        result = next(r for r in case["results"] if r["shape_m"] == int(row["m"]))
        count = result["local_routed_rows"]
        lo, hi, tuning = next(t for t in case["tuning_config"] if t[0] < count <= t[1])
        assert tuning["use_stream_k"]
        label = row["group"].split("/")[-1] + "-" + row["shape"] + "-m" + row["m"]
        cases.append(dict(label=label, suite=suite, group=row["group"], shape=row["shape"],
            m=int(row["m"]), n=case["n"], k=case["k"], experts=case["experts"],
            bias=case.get("bias", False), layer_config=case["layer_config"],
            tuning=copy.deepcopy(tuning), interval=[lo, hi], routed_rows=count,
            counts=result.get("local_expert_counts", result.get("expert_counts"))))
    assert len(cases) == 55 and len({c["label"] for c in cases}) == 55
    (ROOT / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    shutil.copy2(CAMPAIGN / "diagnosis/paired_diagnosis.py", ROOT / "paired_diagnosis.py")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources.rglob("*")) if p.is_file()}
    provenance = dict(base="64ab616cc26e6d5e4000a8fc3022814d2abd90bb",
        patch_source="2e8e8972e8558ebc5cfa2e6f4f1d478c91d50575",
        arms=ARMS, header_sha256=hashes, cases=len(cases), stream_k=True,
        note="Both revisions contain #171; #173 absent. Disabled gates only; no forced enabling.")
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared", len(cases), "cases and", len(ARMS), "arms")


if __name__ == "__main__":
    main()
