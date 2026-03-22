import csv
import ml
import nn
import os
import path


TRAIN_PATH = "data/mnist/mnist_train_1000.csv"
TEST_PATH = "data/mnist/mnist_test_200.csv"
ARTIFACT_DIR = "artifacts/mnist_cnn"
MODEL_PATH = path.join(ARTIFACT_DIR, "model.json")
OPTIM_PATH = path.join(ARTIFACT_DIR, "adam.json")
EPOCHS = 3
LR = 0.001
NUM_CLASSES = 10


def load_dataset(csv_path):
    rows = csv.read_rows(csv_path)
    images = []
    labels = []

    i = 0
    while i < len(rows):
        row = rows[i]
        label = int(row[0].strip())
        image = []
        pixel = 1
        y = 0
        while y < 28:
            line = []
            x = 0
            while x < 28:
                line.append(float(row[pixel].strip()) / 255.0)
                pixel = pixel + 1
                x = x + 1
            image.append(line)
            y = y + 1
        images.append([image])
        labels.append(label)
        i = i + 1

    return [images, labels]


def one_hot(label, classes):
    out = []
    i = 0
    while i < classes:
        if i == label:
            out.append(1.0)
        else:
            out.append(0.0)
        i = i + 1
    return [out]


def argmax(values):
    best_index = 0
    best_value = values[0]
    i = 1
    while i < len(values):
        if values[i] > best_value:
            best_value = values[i]
            best_index = i
        i = i + 1
    return best_index


def build_model():
    return nn.SimpleCNN(28, 1, 8, 3, NUM_CLASSES)


def evaluate(model, images, labels):
    total_loss = 0.0
    correct = 0
    i = 0
    while i < len(images):
        x = ml.tensor([images[i]])
        target = ml.tensor(one_hot(labels[i], NUM_CLASSES))
        logits = model.forward(x)
        total_loss = total_loss + ml.mse_loss(logits, target).item()
        if argmax(logits.tolist()[0]) == labels[i]:
            correct = correct + 1
        i = i + 1

    return {
        "loss": total_loss / len(images),
        "acc": correct / len(images),
    }


def ensure_parent_dir(file_path):
    directory = path.dirname(file_path)
    if directory != "." and not os.exists(directory):
        os.makedirs(directory, exist_ok=True)


def print_metrics(prefix, metrics):
    print(prefix, "loss", metrics["loss"], "acc", metrics["acc"])


if not os.exists(TRAIN_PATH) or not os.exists(TEST_PATH):
    raise RuntimeError("missing dataset: run scripts/prepare_mnist_csv.py first")

train_images, train_labels = load_dataset(TRAIN_PATH)
test_images, test_labels = load_dataset(TEST_PATH)

model = build_model()
optim = nn.Adam(model.parameters(), lr=LR)

epoch = 0
while epoch < EPOCHS:
    total_loss = 0.0
    correct = 0
    i = 0
    while i < len(train_images):
        x = ml.tensor([train_images[i]])
        target = ml.tensor(one_hot(train_labels[i], NUM_CLASSES))

        optim.zero_grad()
        logits = model.forward(x)
        loss = ml.mse_loss(logits, target)
        loss.backward()
        optim.step()

        total_loss = total_loss + loss.item()
        if argmax(logits.tolist()[0]) == train_labels[i]:
            correct = correct + 1
        i = i + 1

    train_metrics = {
        "loss": total_loss / len(train_images),
        "acc": correct / len(train_images),
    }
    test_metrics = evaluate(model, test_images, test_labels)

    print("epoch", epoch + 1)
    print_metrics("train", train_metrics)
    print_metrics("test", test_metrics)
    epoch = epoch + 1

ensure_parent_dir(MODEL_PATH)
ensure_parent_dir(OPTIM_PATH)
model.save(MODEL_PATH)
optim.save(OPTIM_PATH)

saved_metrics = evaluate(model, test_images, test_labels)
print("saved", MODEL_PATH)
print("saved", OPTIM_PATH)
print_metrics("test_before_reload", saved_metrics)

reloaded_model = build_model()
reloaded_optim = nn.Adam(reloaded_model.parameters(), lr=LR)
reloaded_model.load(MODEL_PATH)
reloaded_optim.load(OPTIM_PATH)

reloaded_metrics = evaluate(reloaded_model, test_images, test_labels)
print_metrics("test_after_reload", reloaded_metrics)

if abs(saved_metrics["acc"] - reloaded_metrics["acc"]) > 0.000001:
    raise RuntimeError("reloaded model accuracy mismatch")

print("mnist-cnn-save-load-ok")
