#!/usr/bin/env python3
"""Use the unmodified upstream benchmark, native heuristics and timing method."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def once(arm, output):
    import torch
    from humming.config import GemmType
    from humming.kernel.humming import HummingKernel

    source = Path('/sources') / arm / 'benchmarks/bench_humming.py'
    spec = importlib.util.spec_from_file_location('upstream_benchmark', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(arm=arm, benchmark_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  method='upstream bench_humming, native heuristics, triton.testing.do_bench',
                  cases=[])
    for shape, n, k in [('w13', 4608, 5120), ('w2', 5120, 2304)]:
        for grouped in (False, True):
            torch.manual_seed(2026)
            result = module.bench_humming(
                shape_n=n, shape_k=k, a_dtype='bfloat16', b_dtype='float4e2m1',
                c_dtype='bfloat16', bs_dtype='float8e8m0', weight_scale_group_size=32,
                num_experts=96 if grouped else 0, top_k=6 if grouped else 0,
                shape_m_list=[1, 8, 32, 128],
                gemm_type=GemmType.GROUPED_CONTIGUOUS if grouped else GemmType.DENSE)
            record['cases'].append(dict(shape=shape, n=n, k=k, grouped=grouped, results=result))
            record['compiled_kernels'] = [json.loads(x.to_str()) for x in HummingKernel._id2kernel.values()]
            output.write_text(json.dumps(record, indent=2) + '\n')
            print(arm, shape, grouped, result, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.arm:
        once(args.arm, args.output)
        return
    root = Path('/work/native-bench')
    root.mkdir(exist_ok=False)
    records = []
    # Reverse the order twice. This is a bounded native-path screen,
    # not a confidence interval or a controlled full-model measurement.
    for index, arm in enumerate(['base172', 'post172', 'post172', 'base172'] * 2):
        env = dict(os.environ, PYTHONPATH='/sources/' + arm,
                   HUMMING_CACHE_DIR='/work/cache/native-' + arm,
                   HUMMING_TMP_DIR='/work/tmp/native-' + arm,
                   TRITON_CACHE_DIR='/work/cache/triton-native-' + arm)
        name = str(index) + '-' + arm
        cmd = [sys.executable, __file__, '--arm', arm, '--output', str(root / (name + '.json'))]
        start = time.time()
        with (root / (name + '.log')).open('x') as log:
            try:
                result = subprocess.run(cmd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=900)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        records.append(dict(name=name, command=cmd, returncode=code, seconds=time.time() - start))
        (root / 'status.json').write_text(json.dumps(records, indent=2) + '\n')
        print(records[-1], flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
