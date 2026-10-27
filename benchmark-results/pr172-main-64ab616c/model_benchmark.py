#!/usr/bin/env python3
"""Representative checkpoint shapes, not checkpoint or full-model execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MODELS = [
    dict(model='openai/gpt-oss-20b', hidden=2880, intermediate=2880, experts=32,
         topk=4, bias=True,
         source='https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json'),
    dict(model='openai/gpt-oss-120b', hidden=2880, intermediate=2880, experts=128,
         topk=4, bias=True,
         source='https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json'),
    dict(model='nm-testing/Qwen3-30B-A3B-MXFP4A16', hidden=2048, intermediate=768,
         experts=128, topk=8, bias=False,
         source='https://huggingface.co/nm-testing/Qwen3-30B-A3B-MXFP4A16/blob/45e10a0/config.json'),
]


def once(arm, output):
    import torch
    import triton
    from humming.config import GemmType
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingLayer
    from humming.testing import random_fill_tensor
    from humming.tune import get_heuristics_config

    record = dict(arm=arm, method='TP1 grouped GEMM; native heuristic; do_bench100/1000',
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  numerical_validation='timing only, not a numerical reference test', cases=[])
    for model in MODELS:
        for shape, n, k in [('w13', 2 * model['intermediate'], model['hidden']),
                             ('w2', model['hidden'], model['intermediate'])]:
            torch.manual_seed(2026)
            layer = HummingLayer(shape_n=n, shape_k=k, num_experts=model['experts'],
                pad_n_to_multiple=256, pad_k_to_multiple=128, has_bias=model['bias'],
                weight_config=dict(dtype='float4e2m1', group_size=32, scale_dtype='float8e8m0'),
                input_config=dict(dtype='bfloat16', group_size=0), torch_dtype=torch.bfloat16).to('cuda:0')
            for tensor in layer.parameters():
                random_fill_tensor(tensor)
            layer.transform()
            config = layer.humming_config
            tuning = get_heuristics_config(layer_config=config, gemm_type=GemmType.GROUPED_CONTIGUOUS)
            tuning_json = json.dumps(tuning)
            compute_json = json.dumps(dict(gemm_type=GemmType.GROUPED_CONTIGUOUS.value))
            case = dict(model, shape=shape, n=n, k=k, tp=1, ep=1,
                        layer_config=json.loads(config.to_str()), tuning_config=tuning, results=[])
            for m in (1, 4, 8, 16, 32, 64, 128):
                generator = torch.Generator(device='cpu').manual_seed(2026 + m)
                ids = torch.randn((m, model['experts']), generator=generator).topk(model['topk'], dim=1).indices
                counts = ids.flatten().bincount(minlength=model['experts'])
                offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to('cuda:0')
                routed_rows = int(counts.sum())
                assert routed_rows == m * model['topk']
                torch.manual_seed(2026 + m)
                inputs = torch.randn((routed_rows, k), dtype=torch.bfloat16, device='cuda:0')

                def run():
                    return layer(inputs=inputs, expert_layout=offsets, top_k=1,
                                 compute_config=compute_json, tuning_config=tuning_json)

                result = run()
                torch.cuda.synchronize()
                latency = triton.testing.do_bench(run, warmup=100, rep=1000)
                case['results'].append(dict(shape_m=m, local_routed_rows=routed_rows,
                    expert_counts=counts.tolist(), time=latency, output_shape=list(result.shape)))
            record['cases'].append(case)
            record['compiled_kernels'] = [json.loads(x.to_str()) for x in HummingKernel._id2kernel.values()]
            output.write_text(json.dumps(record, indent=2) + '\n')
            print(arm, model['model'], shape, [(r['shape_m'], r['time']) for r in case['results']], flush=True)
            del layer, inputs, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.arm:
        once(args.arm, args.output)
        return
    root = Path('/work/model-bench')
    root.mkdir(exist_ok=False)
    status = []
    for index, arm in enumerate(['base172', 'post172', 'post172', 'base172']):
        env = dict(os.environ, PYTHONPATH='/sources/' + arm,
                   HUMMING_CACHE_DIR='/work/cache/model-' + arm,
                   HUMMING_TMP_DIR='/work/tmp/model-' + arm,
                   TRITON_CACHE_DIR='/work/cache/triton-model-' + arm)
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
