import csv
import ml
import nn
import os


TRAIN_PATH = "data/mnist/mnist_train_200.csv"
TEST_PATH = "data/mnist/mnist_test_50.csv"
EPOCHS = 3
LR = 0.01


def load_dataset(path):
    rows = csv.read_rows(path)
    images = []
    labels = []

    i = 0
    while i < len(rows):
        row = rows[i]
        label = int(row[0].strip())
        image = []
        pixel_index = 1
        y = 0
        while y < 28:
            line = []
            x = 0
            while x < 28:
                line.append(float(row[pixel_index].strip()) / 255.0)
                pixel_index = pixel_index + 1
                x = x + 1
            image.append(line)
            y = y + 1
        images.append([image])
        labels.append(label)
        i = i + 1

    return [images, labels]


def one_hot(label, classes):
    values = []
    i = 0
    while i < classes:
        if i == label:
            values.append(1.0)
        else:
            values.append(0.0)
        i = i + 1
    return [values]


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


def evaluate(model, images, labels):
    correct = 0
    i = 0
    while i < len(images):
        logits = model.forward(ml.tensor([images[i]]))
        predicted = argmax(logits.tolist()[0])
        if predicted == labels[i]:
            correct = correct + 1
        i = i + 1
    return correct / len(images)


if not os.exists(TRAIN_PATH):
    raise RuntimeError("missing dataset: run scripts/prepare_mnist_csv.py first")

train_images, train_labels = load_dataset(TRAIN_PATH)
test_images, test_labels = load_dataset(TEST_PATH)

model = nn.SimpleCNN(image_size=28, in_channels=1, conv_channels=4, kernel_size=3, num_classes=10)

epoch = 0
while epoch < EPOCHS:
    total_loss = 0.0
    correct = 0
    i = 0
    while i < len(train_images):
        x = ml.tensor([train_images[i]])
        y = ml.tensor(one_hot(train_labels[i], 10))

        model.zero_grad()
        logits = model.forward(x)
        loss = ml.mse_loss(logits, y)
        loss.backward()
        ml.sgd_step(model.parameters(), LR)

        total_loss = total_loss + loss.item()
        if argmax(logits.tolist()[0]) == train_labels[i]:
            correct = correct + 1
        i = i + 1

    train_loss = total_loss / len(train_images)
    train_acc = correct / len(train_images)
    test_acc = evaluate(model, test_images, test_labels)

    print("epoch", epoch + 1, "loss", train_loss, "train_acc", train_acc, "test_acc", test_acc)
    epoch = epoch + 1
