#!/usr/bin/env python3
"""Per-rank V4.1 shapes; excludes all-reduce/all-to-all and model execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def once(arm, output):
    import torch
    import triton
    from humming.config import GemmType
    from humming.layer import HummingLayer
    from humming.testing import random_fill_tensor
    from humming.tune import get_heuristics_config

    record = dict(arm=arm, method='per-rank grouped GEMM; native heuristic; do_bench100/1000',
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  global_experts=384, top_k=6, hidden_size=5120, intermediate_size=2304,
                  numerical_validation='timing only, not a numerical reference test', cases=[])
    for tp, use_ep in [(4, False), (8, False), (4, True), (8, True)]:
        experts = 384 // tp if use_ep else 384
        intermediate = 2304 if use_ep else 2304 // tp
        mode = 'tp' + str(tp) + ('_ep' + str(tp) if use_ep else '')
        for shape, n, k in [('w13', 2 * intermediate, 5120), ('w2', 5120, intermediate)]:
            torch.manual_seed(2026)
            layer = HummingLayer(shape_n=n, shape_k=k, num_experts=experts,
                pad_n_to_multiple=256, pad_k_to_multiple=128,
                weight_config=dict(dtype='float4e2m1', group_size=32, scale_dtype='float8e8m0'),
                input_config=dict(dtype='bfloat16', group_size=0), torch_dtype=torch.bfloat16).to('cuda:0')
            for tensor in layer.parameters():
                random_fill_tensor(tensor)
            layer.transform()
            config = layer.humming_config
            tuning = get_heuristics_config(layer_config=config, gemm_type=GemmType.GROUPED_CONTIGUOUS)
            tuning_json = json.dumps(tuning)
            compute_json = json.dumps(dict(gemm_type=GemmType.GROUPED_CONTIGUOUS.value))
            case = dict(mode=mode, tp=tp, ep=tp if use_ep else 1, shape=shape, n=n, k=k,
                        experts=experts, layer_config=json.loads(config.to_str()),
                        tuning_config=tuning, results=[])
            for m in (1, 4, 8, 16, 32, 64, 128):
                generator = torch.Generator(device='cpu').manual_seed(2026 + m)
                ids = torch.randn((m, 384), generator=generator).topk(6, dim=1).indices
                global_counts = ids.flatten().bincount(minlength=384)
                if use_ep:
                    rank_counts = global_counts.reshape(tp, experts)
                    rank = int(rank_counts.sum(1).argmax())
                    counts = rank_counts[rank]
                else:
                    rank = 0
                    counts = global_counts
                routed_rows = int(counts.sum())
                assert routed_rows > 0 and int(global_counts.sum()) == m * 6
                offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to('cuda:0')
                torch.manual_seed(2026 + m)
                inputs = torch.randn((routed_rows, k), dtype=torch.bfloat16, device='cuda:0')

                def run():
                    return layer(inputs=inputs, expert_layout=offsets, top_k=1,
                                 compute_config=compute_json, tuning_config=tuning_json)

                result = run()
                torch.cuda.synchronize()
                latency = triton.testing.do_bench(run, warmup=100, rep=1000)
                case['results'].append(dict(shape_m=m, local_routed_rows=routed_rows, rank=rank,
                    global_expert_counts=global_counts.tolist(), local_expert_counts=counts.tolist(),
                    time=latency, output_shape=list(result.shape)))
            record['cases'].append(case)
            output.write_text(json.dumps(record, indent=2) + '\n')
            print(arm, mode, shape, [(r['shape_m'], r['local_routed_rows'], r['time']) for r in case['results']], flush=True)
            del layer, inputs, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.arm:
        once(args.arm, args.output)
        return
    root = Path('/work/tp-bench')
    root.mkdir(exist_ok=False)
    status = []
    for index, arm in enumerate(['base172', 'post172', 'post172', 'base172']):
        env = dict(os.environ, PYTHONPATH='/sources/' + arm,
                   HUMMING_CACHE_DIR='/work/cache/tp-' + arm,
                   HUMMING_TMP_DIR='/work/tmp/tp-' + arm,
                   TRITON_CACHE_DIR='/work/cache/triton-tp-' + arm)
        name = str(index) + '-' + arm
        command = [sys.executable, __file__, '--arm', arm, '--output', str(root / (name + '.json'))]
        started = time.time()
        with (root / (name + '.log')).open('x') as log:
            try:
                code = subprocess.run(command, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=1800).returncode
            except subprocess.TimeoutExpired:
                code = 124
        status.append(dict(name=name, returncode=code, seconds=time.time() - started))
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(status[-1], flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
