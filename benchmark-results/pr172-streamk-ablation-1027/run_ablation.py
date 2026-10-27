#!/usr/bin/env python3
"""Six-arm native Stream-K ablation on one idle GPU; never calls a model."""
import gc
import json
import random
import statistics
import time

import paired_diagnosis as h

ARMS = ("base172", "scale_only", "matrix_only", "scale_matrix", "scale_overlap", "post172")
h.ARMS = ARMS
torch = h.torch


def balanced(functions, measure, repeats, seed):
    orders = [ARMS[i:] + ARMS[:i] for i in range(len(ARMS))]
    orders += [tuple(reversed(o)) for o in orders]
    orders *= repeats
    random.Random(seed).shuffle(orders)
    results = []
    for index, order in enumerate(orders):
        row = dict(cycle=index, order=order)
        for arm in order:
            row[arm] = measure(functions[arm])
        results.append(row)
    return results


def main():
    cases = json.loads((h.ROOT / "cases.json").read_text())
    manifest = dict(torch=torch.__version__, cuda=torch.version.cuda, arms=ARMS,
        method="Same process/tensors/native tactic; all Stream-K; balanced arm order; JIT excluded.",
        native="12 cycles, do_bench warmup15/rep40 ms; cache behavior is Triton's native do_bench.",
        warm_graph="60 cycles, 16 launches per replay, event timing after one warm replay.",
        numerical_scope="A/B and A/A only; not an independent reference or Stream-K accuracy qualification.",
        cases=[])
    for index, case in enumerate(cases):
        start_time = time.time()
        print("START", index + 1, "/", len(cases), case["label"], flush=True)
        assert case["tuning"]["use_stream_k"]
        h.Compiler.humming_include_dir = staticmethod(h.ORIGINAL_INCLUDE)
        torch.manual_seed(2026)
        layer = h.HummingLayer(shape_n=case["n"], shape_k=case["k"], num_experts=case["experts"],
            pad_n_to_multiple=256, pad_k_to_multiple=128, has_bias=case["bias"],
            weight_config=dict(dtype="float4e2m1", group_size=32, scale_dtype="float8e8m0"),
            input_config=dict(dtype="bfloat16", group_size=0), torch_dtype=torch.bfloat16).to("cuda:0")
        for tensor in layer.parameters():
            h.random_fill_tensor(tensor)
        del tensor
        layer.transform()
        assert json.loads(layer.humming_config.to_str()) == case["layer_config"]
        counts = torch.tensor(case["counts"], dtype=torch.int64)
        assert int(counts.sum()) == case["routed_rows"]
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to("cuda:0")
        torch.manual_seed(2026 + case["m"])
        inputs = torch.randn((case["routed_rows"], case["k"]), dtype=torch.bfloat16, device="cuda:0")
        outputs = torch.empty((case["routed_rows"], case["n"]), dtype=torch.bfloat16, device="cuda:0")
        configs, kernels = h.prepare_pair(case, layer)
        meta = h.HummingLayerMethod._get_meta(layer)
        kwargs = dict(inputs=inputs, outputs=outputs, expert_layout=offsets, top_k=1, locks=layer.locks)
        for key, attribute in (("weight", meta.weight_name), ("weight_scale", meta.weight_scale_name),
                               ("weight_scale_2", meta.weight_scale_2_name), ("bias", meta.bias_name)):
            kwargs[key] = getattr(layer, attribute, None)
        functions = {arm: (lambda config=configs[arm]: h.launch_kernel(configs=config, **kwargs)) for arm in ARMS}
        record = dict(case=case, kernels=kernels, telemetry_before=h.telemetry())
        a = functions["base172"]().clone()
        record["numerical"] = {}
        for arm in ARMS:
            b = functions[arm]().clone()
            torch.cuda.synchronize()
            record["numerical"][arm] = dict(bit_mismatches=int((a.view(torch.int16) != b.view(torch.int16)).sum()),
                max_abs=float((a.float() - b.float()).abs().max()), nonfinite=int((~torch.isfinite(b)).sum()))
            assert not record["numerical"][arm]["nonfinite"]
        del a, b
        for _ in range(100):
            for fn in functions.values():
                fn()
        torch.cuda.synchronize()
        record["native_cycles"] = balanced(functions,
            lambda fn: float(h.triton.testing.do_bench(fn, warmup=15, rep=40)) * 1000, 1, 3000 + index)
        graphs = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for arm, fn in functions.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(16):
                        fn()
                graphs[arm] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def graph_us(fn):
            fn()
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) * 1000 / 16

        replays = {arm: g.replay for arm, g in graphs.items()}
        record["warm_graph_cycles"] = balanced(replays, graph_us, 5, 4000 + index)
        record["aa_graph_cycles"] = balanced({arm: replays["base172"] for arm in ARMS}, graph_us, 5, 5000 + index)
        record["telemetry_after"] = h.telemetry()
        record["seconds"] = time.time() - start_time
        manifest["cases"].append(record)
        (h.OUT / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        timing = {arm: statistics.median(r[arm] for r in record["warm_graph_cycles"]) for arm in ARMS}
        print("DONE", case["label"], "us", timing, "seconds", record["seconds"], flush=True)
        del graphs, graph, replays, functions, fn, kwargs, inputs, outputs, offsets, layer
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
