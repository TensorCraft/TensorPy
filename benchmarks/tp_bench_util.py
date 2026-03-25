import ml
import nn
import time


NUM_CLASSES = 10


def load_dataset(csv_path, limit=None):
    return ml.load_mnist_csv(csv_path, limit)


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


def empty_timings():
    return {
        "tz": 0.0,
        "zg": 0.0,
        "fw": 0.0,
        "ls": 0.0,
        "bw": 0.0,
        "st": 0.0,
        "mt": 0.0,
    }


def evaluate(model, images, labels):
    timings = empty_timings()
    total_loss = 0.0
    correct = 0

    i = 0
    while i < len(images):
        t0 = time.time()
        x = ml.tensor([images[i]])
        target = ml.tensor(one_hot(labels[i], NUM_CLASSES))
        timings["tz"] = timings["tz"] + (time.time() - t0)

        t0 = time.time()
        logits = model.forward(x)
        timings["fw"] = timings["fw"] + (time.time() - t0)

        t0 = time.time()
        loss = ml.mse_loss(logits, target)
        total_loss = total_loss + loss.item()
        timings["ls"] = timings["ls"] + (time.time() - t0)

        t0 = time.time()
        if argmax(logits.tolist()[0]) == labels[i]:
            correct = correct + 1
        timings["mt"] = timings["mt"] + (time.time() - t0)
        i = i + 1

    return {
        "loss": total_loss / len(images),
        "acc": correct / len(images),
        "timings": timings,
    }


def run_benchmark(train_path, test_path, epochs, lr, train_limit=None, test_limit=None):
    overall_t0 = time.time()

    load_t0 = time.time()
    train_images, train_labels = load_dataset(train_path, train_limit)
    test_images, test_labels = load_dataset(test_path, test_limit)
    load_time = time.time() - load_t0

    init_t0 = time.time()
    model = build_model()
    optim = nn.Adam(model.parameters(), lr=lr)
    init_time = time.time() - init_t0

    train_timings = empty_timings()
    train_total_t0 = time.time()
    epoch = 0
    last_train_loss = 0.0
    last_train_acc = 0.0
    while epoch < epochs:
        total_loss = 0.0
        correct = 0
        i = 0
        while i < len(train_images):
            t0 = time.time()
            x = ml.tensor([train_images[i]])
            target = ml.tensor(one_hot(train_labels[i], NUM_CLASSES))
            train_timings["tz"] = train_timings["tz"] + (time.time() - t0)

            t0 = time.time()
            optim.zero_grad()
            train_timings["zg"] = train_timings["zg"] + (time.time() - t0)

            t0 = time.time()
            logits = model.forward(x)
            train_timings["fw"] = train_timings["fw"] + (time.time() - t0)

            t0 = time.time()
            loss = ml.mse_loss(logits, target)
            train_timings["ls"] = train_timings["ls"] + (time.time() - t0)

            t0 = time.time()
            loss.backward()
            train_timings["bw"] = train_timings["bw"] + (time.time() - t0)

            t0 = time.time()
            optim.step()
            train_timings["st"] = train_timings["st"] + (time.time() - t0)

            total_loss = total_loss + loss.item()

            t0 = time.time()
            if argmax(logits.tolist()[0]) == train_labels[i]:
                correct = correct + 1
            train_timings["mt"] = train_timings["mt"] + (time.time() - t0)
            i = i + 1

        last_train_loss = total_loss / len(train_images)
        last_train_acc = correct / len(train_images)
        epoch = epoch + 1

    train_total_time = time.time() - train_total_t0

    eval_total_t0 = time.time()
    eval_metrics = evaluate(model, test_images, test_labels)
    eval_total_time = time.time() - eval_total_t0

    report = {}
    report["r"] = "tensorpy"

    cfg = {}
    cfg["train"] = train_path
    cfg["test"] = test_path
    cfg["epochs"] = epochs
    cfg["lr"] = lr
    cfg["train_n"] = len(train_images)
    cfg["test_n"] = len(test_images)
    report["c"] = cfg

    timings = {}
    timings["load"] = load_time
    timings["init"] = init_time
    timings["train"] = train_total_time
    timings["tb"] = train_timings
    timings["eval"] = eval_total_time
    timings["eb"] = eval_metrics["timings"]
    timings["all"] = time.time() - overall_t0
    report["t"] = timings

    metrics = {}
    metrics["train_loss"] = last_train_loss
    metrics["train_acc"] = last_train_acc
    metrics["test_loss"] = eval_metrics["loss"]
    metrics["test_acc"] = eval_metrics["acc"]
    report["m"] = metrics

    return report
