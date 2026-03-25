import json
import os
from benchmarks.tp_bench_util import run_benchmark


TRAIN_PATH = "data/mnist/mnist_train_200.csv"
TEST_PATH = "data/mnist/mnist_test_50.csv"
EPOCHS = 1
LR = 0.001
TRAIN_LIMIT = os.getenv("TP_BENCH_TRAIN_LIMIT")
TEST_LIMIT = os.getenv("TP_BENCH_TEST_LIMIT")
EPOCHS_OVERRIDE = os.getenv("TP_BENCH_EPOCHS")
LR_OVERRIDE = os.getenv("TP_BENCH_LR")


if not os.exists(TRAIN_PATH) or not os.exists(TEST_PATH):
    raise RuntimeError("missing dataset")

if TRAIN_LIMIT != None:
    TRAIN_LIMIT = int(TRAIN_LIMIT)
if TEST_LIMIT != None:
    TEST_LIMIT = int(TEST_LIMIT)
if EPOCHS_OVERRIDE != None:
    EPOCHS = int(EPOCHS_OVERRIDE)
if LR_OVERRIDE != None:
    LR = float(LR_OVERRIDE)

print(json.dumps(run_benchmark(TRAIN_PATH, TEST_PATH, EPOCHS, LR, TRAIN_LIMIT, TEST_LIMIT)))
