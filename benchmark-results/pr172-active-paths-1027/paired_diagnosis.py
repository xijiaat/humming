#!/usr/bin/env python3
"""Same-process paired native-kernel diagnosis; not a model benchmark."""
import hashlib
import json
import random
from pathlib import Path
import re
import shutil
import subprocess
import time

import cuda.bindings.driver as cbd
import torch
import triton
from humming.config import GemmType
from humming.jit.compiler import Compiler
from humming.jit.runtime import KernelRuntime
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayer, HummingLayerMethod
from humming.ops.launcher import launch_kernel
from humming.testing import random_fill_tensor

ROOT = Path("/work")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=False)
ARMS = ("base172", "post172")
ORIGINAL_INCLUDE = Compiler.humming_include_dir


def telemetry():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    return result.stdout.strip()


def kernel_record(kernel, folder):
    folder.mkdir(parents=True, exist_ok=False)
    cubin = Path(kernel.kernel_filename)
    shutil.copy2(cubin, folder / "kernel.cubin")
    shutil.copy2(cubin.parent / "kernel.cu", folder / "kernel.cu")
    record = dict(config=json.loads(kernel.to_str()),
                  cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest())
    function = kernel.load_cubin()
    for name in ("NUM_REGS", "SHARED_SIZE_BYTES", "LOCAL_SIZE_BYTES", "MAX_THREADS_PER_BLOCK"):
        result, value = cbd.cuFuncGetAttribute(getattr(cbd.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name), function)
        assert result == 0, str(result)
        record[name.lower()] = value
    executable = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    for option, name in (("--dump-sass", "sass"), ("--dump-resource-usage", "resources")):
        result = subprocess.run([executable, option, str(cubin)], capture_output=True, text=True, timeout=30)
        (folder / (name + ".txt")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0
        if name == "sass":
            lines = [line.strip() for line in result.stdout.splitlines() if re.search(r"/\*[0-9a-fA-F]+\*/", line)]
            record["sass_instructions"] = len(lines)
            record["sass_sha256"] = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    try:
        from elftools.elf.elffile import ELFFile
        with cubin.open("rb") as stream:
            record["text_sections"] = {s.name: hashlib.sha256(s.data()).hexdigest()
                                       for s in ELFFile(stream).iter_sections() if s.name.startswith(".text.")}
    except ImportError:
        record["text_sections"] = "NOT_MEASURED: pyelftools unavailable"
    return record


def prepare_pair(case, layer):
    compute = dict(gemm_type=GemmType.GROUPED_CONTIGUOUS.value)
    pair, metadata = {}, {}
    for arm in ARMS:
        include = "/sources/" + arm + "/humming/include"
        Compiler.humming_include_dir = staticmethod(lambda path=include: path)
        # Only headers differ; retain one shared Python launcher and input tensors.
        KernelRuntime._instances.clear()
        HummingKernel._str2kernel_cache.clear()
        configs = HummingKernel.prepare_kernels(layer.humming_config.to_str(), compute, case["tuning"])
        kernel = HummingKernel._id2kernel[int(configs[2])]
        pair[arm] = configs
        metadata[arm] = kernel_record(kernel, OUT / case["label"] / arm)
    Compiler.humming_include_dir = staticmethod(ORIGINAL_INCLUDE)
    return pair, metadata


def paired_times(functions, measure, pairs, seed):
    orders = [ARMS, ARMS[::-1]] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    samples = []
    for index, order in enumerate(orders):
        sample = dict(pair=index, order=order)
        for arm in order:
            sample[arm] = measure(functions[arm])
        samples.append(sample)
    return samples


def main():
    cases = json.loads((ROOT / "cases.json").read_text())
    manifest = dict(torch=torch.__version__, cuda=torch.version.cuda,
                    method="Same process, exact selected native tactic, shared weights/inputs/output/locks. No forced tactic optimization.",
                    numerical_scope="A/B output comparison only; not an independent accuracy reference.", cases=[])
    for index, case in enumerate(cases):
        started = time.time()
        print("START", case["label"], flush=True)
        Compiler.humming_include_dir = staticmethod(ORIGINAL_INCLUDE)
        torch.manual_seed(2026)
        layer = HummingLayer(shape_n=case["n"], shape_k=case["k"], num_experts=case["experts"],
            pad_n_to_multiple=256, pad_k_to_multiple=128, has_bias=case["bias"],
            weight_config=dict(dtype="float4e2m1", group_size=32, scale_dtype="float8e8m0"),
            input_config=dict(dtype="bfloat16", group_size=0), torch_dtype=torch.bfloat16).to("cuda:0")
        for tensor in layer.parameters():
            random_fill_tensor(tensor)
        layer.transform()
        assert json.loads(layer.humming_config.to_str()) == case["layer_config"]
        counts = torch.tensor(case["counts"], dtype=torch.int64)
        assert int(counts.sum()) == case["routed_rows"]
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to("cuda:0")
        torch.manual_seed(2026 + case["m"])
        inputs = torch.randn((case["routed_rows"], case["k"]), dtype=torch.bfloat16, device="cuda:0")
        outputs = torch.empty((case["routed_rows"], case["n"]), dtype=torch.bfloat16, device="cuda:0")
        configs, metadata = prepare_pair(case, layer)
        meta = HummingLayerMethod._get_meta(layer)
        kwargs = dict(inputs=inputs, outputs=outputs, expert_layout=offsets, top_k=1, locks=layer.locks)
        for key, attribute in (("weight", meta.weight_name), ("weight_scale", meta.weight_scale_name),
                               ("weight_scale_2", meta.weight_scale_2_name), ("bias", meta.bias_name)):
            kwargs[key] = getattr(layer, attribute, None)
        functions = {arm: (lambda config=configs[arm]: launch_kernel(configs=config, **kwargs)) for arm in ARMS}
        result = dict(case=case, kernels=metadata, telemetry_before=telemetry())
        result["same_sass"] = metadata[ARMS[0]]["sass_sha256"] == metadata[ARMS[1]]["sass_sha256"]
        result["same_cubin"] = metadata[ARMS[0]]["cubin_sha256"] == metadata[ARMS[1]]["cubin_sha256"]
        a = functions["base172"]().clone()
        b = functions["post172"]().clone()
        aa = functions["base172"]().clone()
        torch.cuda.synchronize()
        result["output_mismatches_ab"] = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        result["output_mismatches_aa"] = int((a.view(torch.int16) != aa.view(torch.int16)).sum())
        result["nonfinite"] = [int((~torch.isfinite(t)).sum()) for t in (a, b, aa)]
        assert not any(result["nonfinite"])
        for _ in range(100):
            for fn in functions.values():
                fn()
        torch.cuda.synchronize()
        result["native_pairs"] = paired_times(functions,
            lambda fn: float(triton.testing.do_bench(fn, warmup=25, rep=100)) * 1000, 12, 3000 + index)
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

        replays = {arm: graph.replay for arm, graph in graphs.items()}
        result["warm_graph_pairs"] = paired_times(replays, graph_us, 60, 4000 + index)
        aa_functions = {arm: replays["base172"] for arm in ARMS}
        result["aa_graph_pairs"] = paired_times(aa_functions, graph_us, 60, 5000 + index)
        result["telemetry_after"] = telemetry()
        result["seconds"] = time.time() - started
        manifest["cases"].append(result)
        (OUT / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("DONE", case["label"], "same_sass", result["same_sass"], "seconds", result["seconds"], flush=True)
        del graphs, replays, functions, aa_functions, kwargs, inputs, outputs, layer, a, b, aa
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
