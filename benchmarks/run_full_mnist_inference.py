#!/usr/bin/env python3
import json
import os
import statistics
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = int(os.getenv("TP_FULL_MNIST_RUNS", "3"))


def run_process(command, env):
    output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("missing json payload")


def average_payload(items):
    timings = {}
    for key in items[0]["timings"]:
        timings[key] = statistics.mean(item["timings"][key] for item in items)
    return {
        "runtime": items[0]["runtime"],
        "device": items[0]["device"],
        "test_path": items[0]["test_path"],
        "samples": items[0]["samples"],
        "shape": items[0]["shape"],
        "timings": timings,
    }


def collect(command, device):
    env = os.environ.copy()
    env["TP_FULL_MNIST_DEVICE"] = device
    items = []
    for _ in range(RUNS):
        items.append(run_process(command, env))
    return average_payload(items)


def main():
    report = {
        "tensorpy_cpu": collect(["./tensorpy", "benchmarks/full_mnist_inference_tensorpy.py"], "cpu"),
        "pytorch_cpu": collect(["python3", "benchmarks/full_mnist_inference_pytorch.py"], "cpu"),
    }

    mps_available = subprocess.run(
        ["python3", "-c", "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)"],
        cwd=ROOT,
    ).returncode == 0

    if mps_available:
        report["tensorpy_metal"] = collect(["./tensorpy", "benchmarks/full_mnist_inference_tensorpy.py"], "metal")
        report["pytorch_mps"] = collect(["python3", "benchmarks/full_mnist_inference_pytorch.py"], "mps")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
