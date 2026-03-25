import json
import os
import time

import ml
import nn


TEST_PATH = os.getenv("TP_FULL_MNIST_TEST_PATH", "data/mnist/mnist_test_full.csv")
DEVICE = os.getenv("TP_FULL_MNIST_DEVICE", "cpu")
CONV_CHANNELS = int(os.getenv("TP_FULL_MNIST_CONV_CHANNELS", "8"))


load_t0 = time.time()
test = ml.load_mnist_csv(TEST_PATH)
load_t1 = time.time()

tensorize_t0 = time.time()
x = ml.tensor(test[0], ml.float32, ml.device(DEVICE))
tensorize_t1 = time.time()

init_t0 = time.time()
model = nn.SimpleCNN(28, 1, CONV_CHANNELS, 3, 10)
model.to_inference(DEVICE)
init_t1 = time.time()

forward_t0 = time.time()
y = model.forward(x)
forward_t1 = time.time()

print(json.dumps({
    "runtime": "tensorpy",
    "device": DEVICE,
    "test_path": TEST_PATH,
    "samples": len(test[0]),
    "timings": {
        "load_dataset": load_t1 - load_t0,
        "tensorize": tensorize_t1 - tensorize_t0,
        "init_model": init_t1 - init_t0,
        "forward": forward_t1 - forward_t0,
        "overall": forward_t1 - load_t0,
    },
    "shape": y.shape,
}))
