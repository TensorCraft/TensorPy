# TensorPy Builtins And Standard Library

This document summarizes the builtin functions and the V1 standard-library
surface that are available in the current workspace.

## Core Builtins

Common builtins include:

- `print(...)`
- `len(value)`
- `type(value)`
- `callable(value)`
- `str(value)`
- `abs(x)`
- `min(...)`
- `max(...)`
- `sum(...)`
- `all(...)`
- `any(...)`
- `round(x)`
- `ord(char)`
- `chr(code)`
- `dir(value)`

Container construction helpers:

- `list(value)`
- `tuple(value)`
- `dict(...)`
- `set(value)`

Reflection helpers:

- `isinstance(value, type_or_name)`
- `getattr(obj, name, default=None)`
- `setattr(obj, name, value)`
- `hasattr(obj, name)`

Sequence helpers:

- `sorted(list_value)`
- `enumerate(iterable)`
- `range(stop)`
- `range(start, stop)`
- `range(start, stop, step)`
- `reversed(seq)`
- `zip(a, b)`
- `map(func, seq)`
- `filter(func, seq)`

## Builtin Type Methods

TensorPy includes a practical set of methods on:

- `list`
- `dict`
- `set`
- `tuple`
- `str`
- `bytes`

Examples already supported include methods such as:

- `append`, `extend`, `pop`, `remove`, `insert`, `reverse`, `copy`, `count`, `index`
- `get`, `keys`, `values`, `items`, `setdefault`, `popitem`, `update`
- `add`, `discard`, `union`, `intersection`, `difference`, `issubset`
- `lower`, `upper`, `strip`, `split`, `join`, `replace`, `startswith`, `endswith`
- `encode`, `decode`, `hex`

## Standard Library Modules

### Core Data / Utility

- `json`
- `re`
- `math`
- `time`
- `random`

### Filesystem And Runtime

- `os`
  - `name`, `sep`
  - `getcwd()`
  - `listdir(path=".")`
  - `exists(path)`
  - `isdir(path)`
  - `isfile(path)`
  - `mkdir(path, exist_ok=False)`
  - `makedirs(path, exist_ok=False)`
  - `remove(path)`
  - `rmdir(path)`
  - `rename(src, dst)`
- `io`
  - `read_text(path)`
  - `write_text(path, text)`
  - `read_bytes(path)`
  - `write_bytes(path, data)`
- `path`
  - `join(...)`
  - `normpath(path)`
  - `abspath(path)`
  - `basename(path)`
  - `dirname(path)`
  - `splitext(path)`
- `sys`
  - `implementation`
  - `version`
  - `version_info`
  - `platform`
  - `argv`

### Logging / Error Reporting

- `logging`
  - `DEBUG`, `INFO`, `WARN`, `ERROR`
  - `set_level(level)`
  - `get_level()`
  - `debug/info/warn/error(message)`
  - `getLogger(name)`
- `traceback`
  - `format_exception(exc)`
  - `print_exception(exc)`

### Collection / Functional Helpers

- `collections`
  - `Counter`
  - `defaultdict`
  - `flatten(items)`
  - `chunked(items, size)`
- `itertools`
  - `chain(*iterables)`
  - `repeat(value, count)`
  - `take(count, iterable)`
  - `batched(iterable, size)`
- `functools`
  - `partial(func, *bound_args)`
  - `compose(f, g)`

### Embedding-Friendly Helpers

- `env`
  - `get(name, default=None)`
  - `exists(name)`
  - `require(name)`
- `config`
  - `loads(text)`
  - `load(path)`
  - `get(mapping, key, default=None)`
  - `require(mapping, key)`
  - `merge(base, override)`
- `host`
  - `set(name, value)`
  - `get(name, default=None)`
  - `has(name)`
  - `call(name, *args)`
  - `keys()`

### Array / Introspection Helpers

- `array`
  - `zeros(count)`
  - `full(count, value)`
  - `shape(value)`
  - `add(a, b)`
  - `mul(a, b)`
  - `matmul(a, b)`
- `types`
  - type-name constants such as `ListType`, `DictType`, `ModuleType`
  - `type_name(value)`
  - `is_number`, `is_string`, `is_list`, `is_dict`, `is_tuple`, `is_bytes`, `is_module`
- `inspect`
  - `type_name(value)`
  - `is_callable(value)`
  - `is_function(value)`
  - `is_class(value)`
  - `is_module(value)`
  - `members(value)`

## Example

```python
import io
import logging
import os
import path

base = path.join("tmp", "demo")
os.makedirs(base, exist_ok=True)
io.write_text(path.join(base, "hello.txt"), "hi")
logging.info(io.read_text(path.join(base, "hello.txt")))
```
