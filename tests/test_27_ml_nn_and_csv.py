import csv
import io
import ml
import nn


def ok(x, m):
    if not x:
        raise RuntimeError(m)


csv_text = "label,p0,p1\n\"zero\",1,2\n\"with,comma\",3,\"4\"\n"
rows = csv.loads(csv_text)
ok(len(rows) == 3, "a")
ok(rows[1][0] == "zero", "b")
ok(rows[2][0] == "with,comma", "c")
ok(rows[2][2] == "4", "d")

path = "/tmp/tensorpy_csv_test.csv"
io.write_text(path, csv_text)
dict_rows = csv.read_dicts(path)
ok(len(dict_rows) == 2, "e")
ok(dict_rows[0]["label"] == "zero", "f")
ok(dict_rows[1]["p0"] == "3", "g")

linear = nn.Linear(2, 3)
out = linear.forward(ml.tensor([[1, 2]]))
ok(out.rank == 2, "h1")
ok(out.shape[0] == 1 and out.shape[1] == 3, "h2")
ok(isinstance(linear, nn.Module), "h3")
ok(type(out.tolist()) == "list", "h4")

seq = nn.Sequential([
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
])
x = ml.tensor([[1], [2], [3], [4]])
y = ml.tensor([[2], [4], [6], [8]])
initial = ml.mse_loss(seq.forward(x), y).item()
for _ in range(120):
    seq.zero_grad()
    loss = ml.mse_loss(seq.forward(x), y)
    loss.backward()
    ml.sgd_step(seq.parameters(), 0.03)
final = ml.mse_loss(seq.forward(x), y).item()
ok(final < initial, "i")

cnn = nn.SimpleCNN(image_size=3, in_channels=1, conv_channels=1, kernel_size=2, num_classes=1)
image = ml.tensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]])
target = ml.tensor([[1]])
initial_cnn = ml.mse_loss(cnn.forward(image), target).item()
for _ in range(40):
    cnn.zero_grad()
    loss = ml.mse_loss(cnn.forward(image), target)
    loss.backward()
    ml.sgd_step(cnn.parameters(), 0.02)
final_cnn = ml.mse_loss(cnn.forward(image), target).item()
ok(final_cnn < initial_cnn, "j")

print("ml-nn-csv-ok")
