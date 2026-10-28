"""Correctness-checked GEMM/BMM baselines with first-call and warm timings."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

from .workloads import load_workloads


def make_backend(torch, name, op):
    # Distinct Python code objects for GEMM and BMM avoid sharing Dynamo guards.
    def gemm(a, b):
        return torch.mm(a, b)

    def bmm(a, b):
        return torch.bmm(a, b)

    function = {"gemm": gemm, "bmm": bmm}[op]
    if name == "eager":
        return function
    mode = {"inductor": "default", "inductor-max-autotune": "max-autotune"}[name]
    return torch.compile(function, backend="inductor", dynamic=True, fullgraph=True, mode=mode)


def measure_latency(torch, function, a, b, warmup, repeats):
    is_cuda = a.device.type == "cuda"
    for _ in range(warmup):
        function(a, b)
    if is_cuda:
        torch.cuda.synchronize(a.device)
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(repeats)]
        for start, stop in events:
            start.record()
            function(a, b)
            stop.record()
        torch.cuda.synchronize(a.device)
        return [start.elapsed_time(stop) for start, stop in events]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(a, b)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def run(args, workloads):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Follow README.md, or use --dry-run.") from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu --backend eager for a smoke test.")
    if device.type == "cuda":
        # PyTorch 2.5 requires an explicit index for set_device and metadata APIs.
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "triton": triton_version,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "backend": args.backend, "dtype": args.dtype, "seed": args.seed,
        "warmup": args.warmup, "repeats": args.repeats, "cpu_threads": args.cpu_threads,
        "dynamic": args.backend != "eager", "tf32": False,
        "latency_metric": "cuda_event_elapsed" if device.type == "cuda" else "cpu_wall_time",
        "cache_state": "uncontrolled; use fresh processes and separate compiler cache dirs for cold experiments",
        "first_call_note": "includes execution and any compilation/autotuning; not isolated tuning time",
    }
    # Reuse each compiled callable across shapes: dynamic-shape behavior matters.
    functions = {}
    rows = []
    with torch.inference_mode():
        for workload in workloads:
            problem = workload.problem
            a_shape, b_shape = problem.input_shapes
            a = torch.randn(a_shape, device=device, dtype=dtype)
            b = torch.randn(b_shape, device=device, dtype=dtype)
            reference = torch.matmul(a, b)
            setup_ms = 0.0
            if problem.op not in functions:
                start = time.perf_counter()
                functions[problem.op] = make_backend(torch, args.backend, problem.op)
                setup_ms = (time.perf_counter() - start) * 1000
            function = functions[problem.op]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            actual = function(a, b)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - start) * 1000
            tolerance = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 3e-2}[args.dtype]
            torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
            timings = measure_latency(torch, function, a, b, args.warmup, args.repeats)
            median_ms = statistics.median(timings)
            rows.append({
                "name": workload.name, "split": workload.split, **asdict(problem),
                "setup_ms": setup_ms, "first_call_ms": first_call_ms,
                "latency_median_ms": median_ms, "latency_min_ms": min(timings),
                "latency_max_ms": max(timings), "samples_ms": timings,
                "tflops": problem.flops / (median_ms * 1e9) if median_ms > 0 else None,
                "correct": True,
            })
    return {"schema_version": 1, "metadata": metadata, "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=Path("configs/workloads.json"))
    parser.add_argument("--backend", choices=["eager", "inductor", "inductor-max-autotune"], default="eager")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--split", choices=["all", "train", "unseen"], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate/list workloads without PyTorch")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 2 or args.cpu_threads < 1:
        parser.error("warmup must be >= 0, repeats >= 2 and cpu-threads >= 1")
    try:
        workloads = load_workloads(args.workloads, args.dtype, args.split)
        if args.dry_run:
            result = {"dry_run": True, "workloads": [asdict(w) for w in workloads]}
        else:
            result = run(args, workloads)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Saved {args.output}")
    else:
        sys.stdout.write(payload)
    return 0
