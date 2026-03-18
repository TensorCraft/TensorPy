import ml


def ok(x, m):
    if not x:
        raise RuntimeError(m)


def close(a, b):
    return abs(a - b) < 0.0001


cpu_tensor = ml.ones(4)
ok(cpu_tensor.device.name == "cpu", "a")

if not ml.metal_available():
    print("ml-metal-skip")
else:
    metal_tensor = ml.ones(4, ml.float32, ml.metal)
    ok(metal_tensor.device.name == "metal", "b")

    moved = cpu_tensor.to("metal")
    ok(moved.device.name == "metal", "c")

    added = ml.add(metal_tensor, moved)
    ok(added.device.name == "metal", "d")
    ok(close(added.sum(), 8), "e")

    multiplied = ml.mul(metal_tensor, 3)
    ok(multiplied.device.name == "metal", "f")
    ok(close(multiplied.sum(), 12), "g")

    relu = ml.relu(ml.tensor([-1, 2], ml.float32, ml.metal))
    ok(relu.device.name == "metal", "h")
    ok(close(relu.sum(), 2), "i")

    print("ml-metal-ok")
