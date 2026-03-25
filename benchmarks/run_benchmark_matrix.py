#!/usr/bin/env python3
import json
import os
import statistics
import subprocess
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = 3
SCENARIOS = [
    {"name": "single_step", "train_limit": 1, "test_limit": 1, "epochs": 1},
    {"name": "tiny_run", "train_limit": 8, "test_limit": 8, "epochs": 1},
    {"name": "small_run", "train_limit": 32, "test_limit": 32, "epochs": 1},
    {"name": "full_run", "train_limit": 200, "test_limit": 50, "epochs": 1},
]


def run_process(command, env=None):
    started = time.perf_counter()
    output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
    wall = time.perf_counter() - started
    payload = None
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            break
    return wall, payload


def average(values):
    return statistics.mean(values)


def collect_runtime(runtime_name, command, scenario):
    results = []
    for _ in range(RUNS):
        env = os.environ.copy()
        env["TP_BENCH_TRAIN_LIMIT"] = str(scenario["train_limit"])
        env["TP_BENCH_TEST_LIMIT"] = str(scenario["test_limit"])
        env["TP_BENCH_EPOCHS"] = str(scenario["epochs"])
        wall, payload = run_process(command, env)
        results.append((wall, payload))

    if runtime_name == "tensorpy":
        get_summary = lambda item, key: item["t"][key]
        get_breakdown = lambda item, key: item["t"]["tb"][key]
        get_config = lambda item, key: item["c"][key]
    else:
        get_summary = lambda item, key: item["timings"][key]
        get_breakdown = lambda item, key: item["timings"]["train_breakdown"][key]
        get_config = lambda item, key: item["config"][key]

    return {
        "wall": average([wall for wall, _ in results]),
        "load": average([get_summary(payload, "load" if runtime_name == "tensorpy" else "load_dataset") for _, payload in results]),
        "init": average([get_summary(payload, "init" if runtime_name == "tensorpy" else "init_model") for _, payload in results]),
        "train": average([get_summary(payload, "train" if runtime_name == "tensorpy" else "train_total") for _, payload in results]),
        "eval": average([get_summary(payload, "eval" if runtime_name == "tensorpy" else "eval_total") for _, payload in results]),
        "overall": average([get_summary(payload, "all" if runtime_name == "tensorpy" else "overall") for _, payload in results]),
        "tensorize": average([get_breakdown(payload, "tz" if runtime_name == "tensorpy" else "tensorize") for _, payload in results]),
        "forward": average([get_breakdown(payload, "fw" if runtime_name == "tensorpy" else "forward") for _, payload in results]),
        "loss": average([get_breakdown(payload, "ls" if runtime_name == "tensorpy" else "loss") for _, payload in results]),
        "backward": average([get_breakdown(payload, "bw" if runtime_name == "tensorpy" else "backward") for _, payload in results]),
        "step": average([get_breakdown(payload, "st" if runtime_name == "tensorpy" else "step") for _, payload in results]),
        "train_samples": int(get_config(results[0][1], "train_n" if runtime_name == "tensorpy" else "train_samples")),
        "test_samples": int(get_config(results[0][1], "test_n" if runtime_name == "tensorpy" else "test_samples")),
    }


def collect_startup():
    scenarios = {
        "startup_empty": {
            "tensorpy": ["./tensorpy", "-c", "print(0)"],
            "pytorch": ["python3", "-c", "print(0)"],
        },
        "startup_ai_import": {
            "tensorpy": ["./tensorpy", "-c", "import ml\nimport nn\nprint(0)"],
            "pytorch": ["python3", "-c", "import torch\nimport torch.nn as nn\nprint(0)"],
        },
    }
    out = {}
    for name, commands in scenarios.items():
        out[name] = {}
        for runtime_name, command in commands.items():
            times = []
            for _ in range(RUNS):
                wall, _ = run_process(command, os.environ.copy())
                times.append(wall)
            out[name][runtime_name] = average(times)
    return out


def speedup(faster, slower):
    if faster <= 0:
        return None
    return slower / faster


def main():
    report = {"startup": collect_startup(), "scenarios": {}}

    tensorpy_command = ["./tensorpy", "benchmarks/mnist_cnn_tensorpy.py"]
    pytorch_command = ["python3", "benchmarks/mnist_cnn_pytorch.py"]

    for scenario in SCENARIOS:
        report["scenarios"][scenario["name"]] = {
            "tensorpy": collect_runtime("tensorpy", tensorpy_command, scenario),
            "pytorch": collect_runtime("pytorch", pytorch_command, scenario),
        }

    print(json.dumps(report, indent=2))

    print("\nSummary")
    for name, values in report["startup"].items():
        tp = values["tensorpy"]
        pt = values["pytorch"]
        winner = "TensorPy" if tp < pt else "PyTorch"
        ratio = speedup(min(tp, pt), max(tp, pt))
        print(f"{name}: {winner} wins, tensorpy={tp:.4f}s pytorch={pt:.4f}s ratio={ratio:.2f}x")

    for name, values in report["scenarios"].items():
        tp = values["tensorpy"]
        pt = values["pytorch"]
        winner = "TensorPy" if tp["train"] < pt["train"] else "PyTorch"
        ratio = speedup(min(tp['train'], pt['train']), max(tp['train'], pt['train']))
        print(
            f"{name}: train winner={winner}, "
            f"tensorpy_train={tp['train']:.4f}s pytorch_train={pt['train']:.4f}s "
            f"tensorpy_overall={tp['overall']:.4f}s pytorch_overall={pt['overall']:.4f}s "
            f"ratio={ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
