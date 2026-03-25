import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


TEST_PATH = os.getenv("TP_FULL_MNIST_TEST_PATH", "data/mnist/mnist_test_full.csv")
DEVICE = os.getenv("TP_FULL_MNIST_DEVICE", "cpu")
CONV_CHANNELS = int(os.getenv("TP_FULL_MNIST_CONV_CHANNELS", "8"))


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, CONV_CHANNELS, 3)
        self.fc = nn.Linear(CONV_CHANNELS * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_dataset(csv_path):
    images = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            pixels = [float(value.strip()) / 255.0 for value in row[1:]]
            image = []
            for y in range(28):
                start = y * 28
                image.append(pixels[start:start + 28])
            images.append([image])
    return images


device = torch.device(DEVICE)
if device.type == "cpu":
    torch.set_num_threads(1)

load_t0 = time.perf_counter()
images = load_dataset(TEST_PATH)
load_t1 = time.perf_counter()

tensorize_t0 = time.perf_counter()
x = torch.tensor(images, dtype=torch.float32, device=device)
tensorize_t1 = time.perf_counter()

init_t0 = time.perf_counter()
model = SimpleCNN().eval().to(device)
init_t1 = time.perf_counter()

forward_t0 = time.perf_counter()
with torch.no_grad():
    y = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
forward_t1 = time.perf_counter()

print(json.dumps({
    "runtime": "pytorch",
    "device": device.type,
    "test_path": TEST_PATH,
    "samples": len(images),
    "timings": {
        "load_dataset": load_t1 - load_t0,
        "tensorize": tensorize_t1 - tensorize_t0,
        "init_model": init_t1 - init_t0,
        "forward": forward_t1 - forward_t0,
        "overall": forward_t1 - load_t0,
    },
    "shape": list(y.shape),
}))
