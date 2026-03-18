import array
import inspect
import path
import types


def add3(a, b, c):
    return a + b + c


print(array.zeros(4))
print(array.full(3, 7))
print(array.shape([[1, 2], [3, 4]]))
print(array.add([1, 2, 3], 10))
print(array.mul([2, 3, 4], [5, 6, 7]))
print(array.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))

print(types.type_name([1]))
print(types.is_number(1.5))
print(types.is_module(path))

print(callable(add3))
print(inspect.is_callable(add3))
print(inspect.is_function(add3))
print(inspect.is_module(path))
print(inspect.type_name(add3))
print(len(inspect.members(path)) > 0)
