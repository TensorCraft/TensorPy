import ml
import nn


def ok(x, m):
    if not x:
        raise RuntimeError(m)


x = ml.tensor([[1], [2], [3], [4]])
y = ml.tensor([[3], [5], [7], [9]])
model = nn.Linear(1, 1)
optim = nn.Adam(model.parameters(), lr=0.05)
initial = ml.mse_loss(model.forward(x), y).item()

for _ in range(120):
    optim.zero_grad()
    loss = ml.mse_loss(model.forward(x), y)
    loss.backward()
    optim.step()

final = ml.mse_loss(model.forward(x), y).item()
ok(final < initial, "a")
ok(final < 0.1, "b")
ok(model.weight.item() > 1.5, "c")

print("nn-adam-ok")
