import io
import ml
import nn
import os


def ok(x, m):
    if not x:
        raise RuntimeError(m)


def close(a, b, eps=0.0001):
    return abs(a - b) < eps


def same_nested(left, right, eps=0.0001):
    if type(left) != type(right):
        return False
    if type(left) == "list":
        if len(left) != len(right):
            return False
        i = 0
        while i < len(left):
            if not same_nested(left[i], right[i], eps):
                return False
            i = i + 1
        return True
    if type(left) == "float" or type(left) == "int":
        return close(left, right, eps)
    return left == right


model = nn.Linear(2, 2)
x = ml.tensor([[1, 2], [3, 4]])
baseline = model.forward(x).tolist()

state = model.state_dict()
ok("weight" in state, "a")
ok("bias" in state, "b")
ok(state["weight"]["device"] == "cpu", "c")
ok(state["weight"]["requires_grad"], "d")

path = "/tmp/tensorpy_nn_state.json"
if os.exists(path):
    os.remove(path)

ok(nn.save(model, path) == path, "e")
ok(os.exists(path), "f")

model.weight = ml.Parameter([[0, 0], [0, 0]])
model.bias = ml.Parameter([0, 0])
changed = model.forward(x).tolist()
ok(not same_nested(changed, baseline), "g")

load_info = nn.load(model, path)
ok(len(load_info["missing_keys"]) == 0, "h")
ok(len(load_info["unexpected_keys"]) == 0, "i")
restored = model.forward(x).tolist()
ok(same_nested(restored, baseline), "j")

reloaded = nn.Linear(2, 2)
reloaded.load(path)
ok(same_nested(reloaded.forward(x).tolist(), baseline), "k")

if ml.metal_available():
    try:
        model.to("metal")
        raise RuntimeError("expected-metal-param-error")
    except RuntimeError as exc:
        ok("only CPU devices" in exc.message, "l")

print("nn-save-load-ok")
