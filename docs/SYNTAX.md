# TensorPy Syntax Guide

TensorPy is a Python-like language, but it currently implements a practical
subset rather than full CPython compatibility. This guide focuses on the syntax
that is available in the current repository.

## Basics

TensorPy supports:

- numbers, booleans, strings, bytes, and `None`
- arithmetic and comparison operators
- variable assignment
- indexing and slicing
- list, tuple, dict, and set literals

Examples:

```python
x = 7
name = "tensorpy"
nums = [1, 2, 3]
pair = (1, 2)
mapping = {"ok": True}
unique = {1, 2, 3}

print(x + 5)
print(name.upper())
print(nums[1])
print(nums[0:2])
```

## Control Flow

Conditionals and loops are supported:

```python
if x > 3:
    print("big")
else:
    print("small")

for item in [1, 2, 3]:
    print(item)

i = 0
while i < 3:
    print(i)
    i = i + 1
```

Supported statements include:

- `if` / `elif` / `else`
- `for`
- `while`
- `break`
- `continue`
- `pass`

## Functions And Lambdas

TensorPy supports:

- named functions
- default arguments
- `*args`
- lambdas
- nested functions and closure capture

```python
def add(a, b=1):
    return a + b

def wrap(prefix):
    def inner(value):
        return prefix + value
    return inner

tag = wrap("v1:")
print(add(4))
print(tag("ok"))
print((lambda x: x * 2)(6))
```

## Classes

Class definitions, instance fields, and methods are supported:

```python
class Box:
    def __init__(self, value):
        self.value = value

    def double(self):
        return self.value * 2

box = Box(9)
print(box.value)
print(box.double())
```

## Exceptions

TensorPy supports typed exceptions with `try` / `except` and `raise`:

```python
try:
    raise ValueError("boom")
except ValueError as e:
    print(type(e))
    print(e.message)
```

Builtin exception classes currently include:

- `Exception`
- `RuntimeError`
- `OSError`
- `TypeError`
- `ValueError`
- `KeyError`
- `IndexError`
- `AttributeError`
- `ZeroDivisionError`

## Imports

The module system supports:

- `import module`
- `import package.module`
- `from module import name`
- `from module import name as alias`
- `from package.module import name`

TensorPy looks for modules in:

- `modules/<name>.py`
- `modules/<pkg>/__init__.py`
- `lib/<name>.py`
- `./<name>.py`

## Notes And Current Gaps

Important differences from full Python today:

- compatibility is intentionally partial
- many standard-library modules are small V1 subsets
- some advanced language features are still missing
- runtime and C extension support are still evolving beyond the scalar-only ABI

For builtin functions and the current standard-library surface, see
`docs/BUILTINS_AND_STDLIB.md`.
