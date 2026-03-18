import ml


def ok(x, m):
    if not x:
        raise RuntimeError(m)


def close(a, b):
    return abs(a - b) < 0.0001


a = ml.tensor([[1, 2], [3, 4]])
b = ml.tensor([[10, 20], [30, 40]])

c = ml.add(a, b)
ok(close(c.sum(), 110), "a")
ok(close(ml.mul(a, 2).sum(), 20), "b")
ok(close(ml.sub(10, a).sum(), 30), "c")
ok(close(ml.div(b, 10).mean(), 2.5), "d")

g = ml.add(ml.tensor([[1], [2]]), ml.tensor([[10, 20]]))
ok(len(g.shape) == 2, "e")
ok(close(g.sum(), 66), "f")

m = ml.matmul(a, b)
ok(close(m.sum(), 540), "g")
ok(close(m.max(), 220), "h")
ok(close(ml.matmul(ml.tensor([1, 2, 3]), ml.tensor([4, 5, 6])), 32), "i")

v = ml.matmul(a, ml.tensor([1, 2]))
ok(close(v.sum(), 16), "j")

ok(close(ml.relu(ml.tensor([-1, 0, 2])).sum(), 2), "k")
ok(close(ml.tensor([-1, 0, 2]).relu().sum(), 2), "l")
ok(close(ml.sigmoid(ml.tensor([0, 0])).mean(), 0.5), "m")

ge = ml.gelu(ml.tensor([1]))
ok(ge.max() > 0.8, "n")
ok(ge.max() < 0.9, "o")

s = ml.softmax(ml.tensor([[1, 1], [2, 2]]))
ok(close(s.sum(), 2), "p")
ok(close(s.max(), 0.5), "q")

ln = ml.layernorm(ml.tensor([[1, 2], [3, 4]]))
ok(close(ln.mean(), 0), "r")
ok(ln.max() > 0.9, "s")
ok(ln.max() < 1.1, "t")
ok(close(ml.tensor([[1, 2], [3, 4]]).layernorm().mean(), 0), "u")

print("ml-eager-ok")
