import ml


def ok(x, m):
    if not x:
        raise RuntimeError(m)


def close(a, b, eps=0.2):
    return abs(a - b) < eps


x = ml.tensor([[1], [2], [3], [4]])
y = ml.tensor([[3], [5], [7], [9]])

w = ml.Parameter([[0]])
b = ml.Parameter([0])

initial = ml.mse_loss(ml.add(ml.matmul(x, w), b), y).item()

for _ in range(200):
    ml.zero_grad([w, b])
    pred = ml.add(ml.matmul(x, w), b)
    loss = ml.mse_loss(pred, y)
    loss.backward()
    ml.sgd_step([w, b], 0.05)

final = ml.mse_loss(ml.add(ml.matmul(x, w), b), y).item()

ok(w.requires_grad, "a")
ok(b.requires_grad, "b")
ok(w.grad != None, "c")
ok(b.grad != None, "d")
ok(final < initial * 0.05, "e")
ok(close(w.item(), 2.0), "f")
ok(close(b.item(), 1.0), "g")

print("ml-train-ok")
