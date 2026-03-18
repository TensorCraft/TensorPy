import ml
import types

x = ml.tensor([[1, 2], [3, 4]])
print(type(x))
print(x.shape)
print(x.rank)
print(x.size)
print(x.dtype)
print(x.device)
print(x.contiguous)
print(x.strides)
print(len(x))
print(types.is_tensor(x))
print(types.is_dtype(x.dtype))
print(types.is_device(x.device))
print(getattr(x, "shape"))
print(hasattr(x, "dtype"))
print("reshape" in dir(x))

y = ml.zeros([2, 3])
print(y)
print(y.shape)

z = ml.ones(4)
print(z.shape)
print(z)

w = ml.full((3,), 2)
print(w)

r = ml.arange(1, 6, 2)
print(r)

reshaped = ml.reshape(x, [4])
print(reshaped.shape)
print(reshaped)

print(x.reshape([4]).shape)
print(x.to("cpu").device)
print(ml.cast(x, ml.float32).dtype)
print(x.astype("float32").dtype)
print(ml.device("cpu"))
print(ml.dtype("float32"))
