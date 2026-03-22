import ml
import nn
import os


def ok(x, m):
    if not x:
        raise RuntimeError(m)


x = ml.tensor([[1], [2], [3], [4]])
y = ml.tensor([[3], [5], [7], [9]])
model = nn.Linear(1, 1)
optim = nn.Adam(model.parameters(), lr=0.05)
initial = ml.mse_loss(model.forward(x), y).item()

for _ in range(180):
    optim.zero_grad()
    loss = ml.mse_loss(model.forward(x), y)
    loss.backward()
    optim.step()

final = ml.mse_loss(model.forward(x), y).item()
ok(final < initial, "a")
ok(final < 0.1, "b")
ok(model.weight.item() > 1.5, "c")

state = optim.state_dict()
ok(state["step_count"] == 180, "d")
ok(len(state["m"]) == len(model.parameters()), "e")
ok(len(state["v"]) == len(model.parameters()), "f")

path = "/tmp/tensorpy_adam_state.json"
if os.exists(path):
    os.remove(path)

ok(nn.save(optim, path) == path, "g")
ok(os.exists(path), "h")

fresh = nn.Adam(model.parameters(), lr=0.001)
ok(fresh.lr == 0.001, "i")
info = nn.load(fresh, path)
ok(len(info["missing_keys"]) == 0, "j")
ok(len(info["unexpected_keys"]) == 0, "k")
ok(fresh.lr == optim.lr, "l")
ok(fresh.beta1 == optim.beta1, "m")
ok(fresh.beta2 == optim.beta2, "n")
ok(fresh.eps == optim.eps, "o")
ok(fresh.step_count == optim.step_count, "p")
ok(len(fresh.m) == len(optim.m), "q")
ok(len(fresh.v) == len(optim.v), "r")
ok(abs(fresh.m[0].sum() - optim.m[0].sum()) < 0.0001, "s")
ok(abs(fresh.v[0].sum() - optim.v[0].sum()) < 0.0001, "t")

print("nn-adam-ok")
