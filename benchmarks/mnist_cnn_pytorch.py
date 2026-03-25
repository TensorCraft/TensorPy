import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_PATH = "data/mnist/mnist_train_200.csv"
TEST_PATH = "data/mnist/mnist_test_50.csv"
EPOCHS = 1
LR = 0.001
NUM_CLASSES = 10
TRAIN_LIMIT = os.getenv("TP_BENCH_TRAIN_LIMIT")
TEST_LIMIT = os.getenv("TP_BENCH_TEST_LIMIT")
EPOCHS_OVERRIDE = os.getenv("TP_BENCH_EPOCHS")
LR_OVERRIDE = os.getenv("TP_BENCH_LR")


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.fc = nn.Linear(8 * 26 * 26, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_dataset(csv_path, limit=None):
    images = []
    labels = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if limit is not None and len(images) >= limit:
                break
            labels.append(int(row[0].strip()))
            pixels = [float(value.strip()) / 255.0 for value in row[1:]]
            image = []
            for y in range(28):
                start = y * 28
                image.append(pixels[start:start + 28])
            images.append([image])
    return images, labels


def one_hot(label, classes):
    out = [0.0] * classes
    out[label] = 1.0
    return [out]


def empty_timings():
    return {
        "tensorize": 0.0,
        "zero_grad": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "step": 0.0,
        "metrics": 0.0,
    }


def evaluate(model, images, labels):
    timings = empty_timings()
    total_loss = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for image, label in zip(images, labels):
            t0 = time.perf_counter()
            x = torch.tensor([image], dtype=torch.float32)
            target = torch.tensor(one_hot(label, NUM_CLASSES), dtype=torch.float32)
            timings["tensorize"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            logits = model(x)
            timings["forward"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            loss = F.mse_loss(logits, target)
            total_loss += loss.item()
            timings["loss"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if torch.argmax(logits, dim=1).item() == label:
                correct += 1
            timings["metrics"] += time.perf_counter() - t0

    return {
        "loss": total_loss / len(images),
        "acc": correct / len(images),
        "timings": timings,
    }

if TRAIN_LIMIT is not None:
    TRAIN_LIMIT = int(TRAIN_LIMIT)
if TEST_LIMIT is not None:
    TEST_LIMIT = int(TEST_LIMIT)
if EPOCHS_OVERRIDE is not None:
    EPOCHS = int(EPOCHS_OVERRIDE)
if LR_OVERRIDE is not None:
    LR = float(LR_OVERRIDE)

torch.set_num_threads(1)
overall_t0 = time.perf_counter()

load_t0 = time.perf_counter()
train_images, train_labels = load_dataset(TRAIN_PATH, TRAIN_LIMIT)
test_images, test_labels = load_dataset(TEST_PATH, TEST_LIMIT)
load_time = time.perf_counter() - load_t0

init_t0 = time.perf_counter()
model = SimpleCNN()
optim = torch.optim.Adam(model.parameters(), lr=LR)
init_time = time.perf_counter() - init_t0

train_timings = empty_timings()
train_total_t0 = time.perf_counter()
last_train_loss = 0.0
last_train_acc = 0.0
for _ in range(EPOCHS):
    total_loss = 0.0
    correct = 0
    model.train()
    for image, label in zip(train_images, train_labels):
        t0 = time.perf_counter()
        x = torch.tensor([image], dtype=torch.float32)
        target = torch.tensor(one_hot(label, NUM_CLASSES), dtype=torch.float32)
        train_timings["tensorize"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        train_timings["zero_grad"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = model(x)
        train_timings["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        loss = F.mse_loss(logits, target)
        train_timings["loss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        train_timings["backward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        optim.step()
        train_timings["step"] += time.perf_counter() - t0

        total_loss += loss.item()

        t0 = time.perf_counter()
        if torch.argmax(logits, dim=1).item() == label:
            correct += 1
        train_timings["metrics"] += time.perf_counter() - t0

    last_train_loss = total_loss / len(train_images)
    last_train_acc = correct / len(train_images)

train_total_time = time.perf_counter() - train_total_t0

eval_total_t0 = time.perf_counter()
eval_metrics = evaluate(model, test_images, test_labels)
eval_total_time = time.perf_counter() - eval_total_t0

report = {
    "runtime": "pytorch",
    "config": {
        "train_path": TRAIN_PATH,
        "test_path": TEST_PATH,
        "epochs": EPOCHS,
        "lr": LR,
        "num_classes": NUM_CLASSES,
        "train_samples": len(train_images),
        "test_samples": len(test_images),
    },
    "timings": {
        "load_dataset": load_time,
        "init_model": init_time,
        "train_total": train_total_time,
        "train_breakdown": train_timings,
        "eval_total": eval_total_time,
        "eval_breakdown": eval_metrics["timings"],
        "overall": time.perf_counter() - overall_t0,
    },
    "metrics": {
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "test_loss": eval_metrics["loss"],
        "test_acc": eval_metrics["acc"],
    },
}

print(json.dumps(report))
