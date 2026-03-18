#!/usr/bin/env python3
import csv
import gzip
import os
import struct
import urllib.request


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download(url, path):
    if os.path.exists(path):
        return
    urllib.request.urlretrieve(url, path)


def read_images(path):
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic: {magic}")
        payload = handle.read()
    image_size = rows * cols
    images = []
    for index in range(count):
        start = index * image_size
        end = start + image_size
        images.append(list(payload[start:end]))
    return images


def read_labels(path):
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic: {magic}")
        payload = handle.read()
    return list(payload[:count])


def write_csv(path, images, labels, limit):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for index, (label, image) in enumerate(zip(labels, images)):
            if limit is not None and index >= limit:
                break
            writer.writerow([label] + image)


def main():
    root = os.path.join(os.getcwd(), "data", "mnist")
    raw = os.path.join(root, "raw")
    ensure_dir(raw)

    local_paths = {}
    for name, filename in FILES.items():
        path = os.path.join(raw, filename)
        download(BASE_URL + filename, path)
        local_paths[name] = path

    train_images = read_images(local_paths["train_images"])
    train_labels = read_labels(local_paths["train_labels"])
    test_images = read_images(local_paths["test_images"])
    test_labels = read_labels(local_paths["test_labels"])

    write_csv(os.path.join(root, "mnist_train_200.csv"), train_images, train_labels, 200)
    write_csv(os.path.join(root, "mnist_test_50.csv"), test_images, test_labels, 50)
    write_csv(os.path.join(root, "mnist_train_1000.csv"), train_images, train_labels, 1000)
    write_csv(os.path.join(root, "mnist_test_200.csv"), test_images, test_labels, 200)

    print("prepared:")
    print(os.path.join(root, "mnist_train_200.csv"))
    print(os.path.join(root, "mnist_test_50.csv"))
    print(os.path.join(root, "mnist_train_1000.csv"))
    print(os.path.join(root, "mnist_test_200.csv"))


if __name__ == "__main__":
    main()
