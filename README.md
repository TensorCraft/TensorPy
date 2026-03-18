# TensorPy

TensorPy is an AI Native Python interpreter written in C. It is built around a practical Python-like language core plus native support for tensors, devices, autograd, atomics, SIMD compute paths, threading primitives, and explicit Metal acceleration on Apple Silicon.

This project is still in an active build-out phase. The goal is not full CPython compatibility. The goal is a compact, hackable, high-performance interpreter that feels Pythonic while treating ML and systems primitives as first-class runtime features instead of bolted-on libraries.

## Positioning

TensorPy aims to be:

- an AI Native Python interpreter, not just a Python-like scripting language
- native-first for `tensor`, `device`, and training workflows
- systems-aware, with built-in concurrency primitives, atomics, SIMD, and portability layers
- small enough to understand, modify, and extend without a giant runtime stack

## Current Features

- Python-like syntax for expressions, functions, classes, conditionals, loops, slicing, and container literals
- Core data structures: `list`, `tuple`, `dict`, `set`, `str`, `bytes`
- Typed exceptions with `try` / `except`, including `except Exception as e`
- Improved function call semantics for keyword arguments, duplicate parameter checks, missing-argument reporting, and `*args`
- Basic nested-function closure capture for outer bindings
- REPL plus file execution
- Basic module imports:
  - `import module`
  - `import package.module`
  - `from module import name`
  - `from package import module`
  - `from package.module import name`
  - `from module import name as alias`
- Module lookup currently checks:
  - `modules/<name>.py`
  - `modules/<pkg>/__init__.py`
  - `lib/<name>.py`
  - `./<name>.py`
- Builtin-like modules written in TensorPy:
  - `json`
  - `re`
  - `math`
  - `time`
  - `random`
  - `os`
  - `io`
  - `path`
  - `logging`
  - `traceback`
  - `sys`
  - `collections`
  - `itertools`
  - `functools`
  - `env`
  - `config`
  - `host`
  - `array`
  - `ml` (builtin module)
  - `types`
  - `inspect`
- Platform-facing runtime operations are routed through a portability layer in `src/platform.c`
- Native ML runtime with `tensor`, `dtype`, `device`, autograd primitives, and eager ops
- Runtime concurrency layer with threads, mutexes, condition variables, atomics, thread pool, and `parallel_for`
- CPU compute backend with scalar, SIMD, and threaded execution paths for core float32 kernels
- Apple Silicon Metal backend with explicit opt-in execution and a portable no-Metal build mode
- CPU remains the default execution device for compatibility; Metal is explicit opt-in

## Milestone Status

See [ROADMAP.md](/Users/tensorcraft/Projects/TensorPy/ROADMAP.md) for the tracked checklist of completed work, in-flight work, and next priorities.

## Implemented Library Surface

The runtime already includes a practical set of methods for:

- `list`
- `dict`
- `set`
- `tuple`
- `str`
- `bytes`

Examples include methods such as `append`, `pop`, `remove`, `sort`, `setdefault`, `popitem`, `union`, `intersection`, `count`, `index`, `split`, `join`, `encode`, `decode`, and `hex`.

Common builtins now also include:

- `isinstance`
- `getattr`
- `setattr`
- `hasattr`
- `reversed`
- `zip`
- `map`
- `filter`
- `round`
- `ord`
- `chr`
- `dir`

Functions, lambdas, and methods now support `*args` collection.

The `json` module supports:

- `json.loads(...)`
- `json.dumps(...)`
- `json.JSON(...).parse()`
- string escapes including `\n`, `\t`, `\r`, `\b`, `\f`, and `\uXXXX`
- exponent numbers like `1.5e2`
- stricter invalid-literal and invalid-escape errors

The `os` module currently supports:

- `os.name`
- `os.sep`
- `os.getcwd()`
- `os.listdir(path=".")`
- `os.exists(path)`
- `os.isdir(path)`
- `os.isfile(path)`
- `os.mkdir(path, exist_ok=False)`
- `os.makedirs(path, exist_ok=False)`
- `os.remove(path)`
- `os.rmdir(path)`
- `os.rename(src, dst)`
- `os.replace(src, dst)`
- `os.removedirs(path)`
- `os.getenv(name, default=None)`
- `os.system(command)`

The V1 utility modules also include:

- `io.read_text(path)` / `io.write_text(path, text)`
- `io.read_bytes(path)` / `io.write_bytes(path, bytes)`
- `io.append_text(path, text)` / `io.append_bytes(path, bytes)`
- `io.read_lines(path)` / `io.write_lines(path, lines)`
- `path.join(...)`, `path.normpath(...)`, `path.abspath(...)`
- `path.basename(...)`, `path.dirname(...)`, `path.splitext(...)`
- `path.exists(...)`, `path.isdir(...)`, `path.isfile(...)`
- `path.split(...)`, `path.relpath(path, start=".")`
- `logging.debug/info/warn/error(...)`
- `logging.basicConfig(level=...)`, `logging.exception(exc)`
- `logging.getLogger(name)`
- `traceback.format_exception(exc)` / `traceback.print_exception(exc)`
- `traceback.format_exception_only(exc)` / `traceback.as_dict(exc)`
- minimal `sys` metadata: `implementation`, `version`, `version_info`, `platform`, `argv`, `path`, `byteorder`, `executable`
- `collections.Counter`, `collections.defaultdict`, `collections.flatten`, `collections.chunked`
- `itertools.chain`, `itertools.repeat`, `itertools.take`, `itertools.batched`
- `functools.partial`, `functools.compose`
- `env.get`, `env.exists`, `env.require`
- `config.load`, `config.loads`, `config.get`, `config.require`, `config.merge`
- `host.set`, `host.get`, `host.has`, `host.call`
- `array.zeros`, `array.full`, `array.shape`, `array.add`, `array.mul`, `array.matmul`
- `ml.metal_available()`
- `ml.device(name)`, `ml.dtype(name)`
- `ml.tensor(data, dtype=None, device=None)`
- `ml.Parameter(data, dtype=None, device=None)`
- `ml.zeros(shape, dtype=None, device=None)`
- `ml.ones(shape, dtype=None, device=None)`
- `ml.full(shape, value, dtype=None, device=None)`
- `ml.arange(start, stop=None, step=None)`
- `ml.reshape(tensor, shape)`, `ml.cast(tensor, dtype)`
- `ml.add/sub/mul/div`, `ml.sum/mean/max`, `ml.matmul`
- `ml.relu`, `ml.sigmoid`, `ml.gelu`, `ml.softmax`, `ml.layernorm`
- `ml.mse_loss(pred, target)`
- `ml.backward(tensor)`, `ml.zero_grad(params)`, `ml.sgd_step(params, lr)`
- `types.type_name` plus basic `is_*` helpers
- `inspect.type_name`, `inspect.is_callable`, `inspect.is_function`, `inspect.is_class`, `inspect.is_module`

Tensor objects currently expose:

- `.shape`, `.rank`, `.size`, `.dtype`, `.device`, `.contiguous`, `.strides`
- `.requires_grad`, `.grad`
- `.reshape(...)`, `.to(...)`, `.astype(...)`
- `.item()`, `.backward()`, `.zero_grad()`
- `.sum()`, `.mean()`, `.max()`, `.matmul(...)`
- `.relu()`, `.sigmoid()`, `.gelu()`, `.softmax()`, `.layernorm()`

The `re` module currently supports a useful regex subset:

- `re.compile(...)`
- `re.match(...)`
- `re.search(...)`
- `re.fullmatch(...)`
- `re.findall(...)`
- `re.split(...)`
- `re.sub(...)`
- `re.subn(...)`

Supported regex constructs currently include:

- literals
- `.`
- `^` and `$`
- top-level alternation with `|`
- `*`, `+`, `?`
- grouping, including grouped quantifiers like `(ab)+`
- character classes like `[abc]`
- ranges like `[a-z]`
- negated classes like `[^0-9]`
- `\d`, `\w`, `\s`
- captured-group access via `group(1)`, `start(1)`, `end(1)`, and `span(1)`

## C API Status

TensorPy now includes a minimal public embedding header:

- `include/tensorpy/api.h`

It exposes an opaque `TPContext` plus small helpers for:

- creating a context
- interpreting source in that context
- reading and writing simple globals as public values
- retrieving the last runtime error as a public value
- registering a simple host-native module with scalar values and functions
- checking the public API / extension ABI version
- destroying the context

Module registration handles are lightweight setup helpers; the VM owns the
registered module object after creation.

This is enough to begin embedding TensorPy from C without including `vm.h` or
other internal runtime headers. The detailed readiness notes live in:

- `docs/C_API_READINESS.md`

## Build

```bash
make
```

To build without Metal support:

```bash
make METAL=0
```

This produces the interpreter binary:

```bash
./tensorpy
```

## Usage

Run a script:

```bash
./tensorpy path/to/script.py
```

Run one command:

```bash
./tensorpy -c "print(1 + 2)"
```

Start the interactive REPL:

```bash
./tensorpy
```

In the REPL, expression-like input is automatically echoed:

```text
> 1 + 2
3
> x = 7
> x
7
```

### ML Runtime

```python
import ml

x = ml.tensor([[1, 2], [3, 4]])
print(x.shape)          # [2, 2]
print(ml.add(x, x).sum())

# CPU is still the default
print(ml.ones(4).device.name)   # cpu

# Metal is explicit opt-in
if ml.metal_available():
    y = ml.ones(4, ml.float32, ml.metal)
    print(y.device.name)        # metal
    print(ml.mul(y, 3).sum())
```

### CPU Training Loop

```python
import ml

x = ml.tensor([[1], [2], [3], [4]])
y = ml.tensor([[3], [5], [7], [9]])

w = ml.Parameter([[0]])
b = ml.Parameter([0])

for _ in range(200):
    ml.zero_grad([w, b])
    pred = ml.add(ml.matmul(x, w), b)
    loss = ml.mse_loss(pred, y)
    loss.backward()
    ml.sgd_step([w, b], 0.05)

print(w.item(), b.item())
```

## Examples

### Importing Modules

```python
import json
from json import loads as jl

print(json.dumps({"ok": True, "nums": [1, 2]}))
print(jl("[1,2,3]")[2])
```

### Regex

```python
import re

print(re.findall("\\d+", "a12 b34 c5"))
print(re.sub("\\d+", "#", "a12 b34 c5"))
```

## Testing

Run the full test suite:

```bash
python3 run_tests.py
```

At the time of writing, the suite contains `29` organized test files and passes in the current workspace.

## Project Layout

- [src/main.c](/Users/tensorcraft/Projects/TensorPy/src/main.c): CLI entry point and REPL
- [src/compiler.c](/Users/tensorcraft/Projects/TensorPy/src/compiler.c): parser and bytecode compiler
- [src/vm.c](/Users/tensorcraft/Projects/TensorPy/src/vm.c): virtual machine and runtime behavior
- [src/platform.c](/Users/tensorcraft/Projects/TensorPy/src/platform.c): portability layer for filesystem, time, random, and OS-facing helpers
- [modules](/Users/tensorcraft/Projects/TensorPy/modules): TensorPy standard-library-style modules
- [modules/json.py](/Users/tensorcraft/Projects/TensorPy/modules/json.py): TensorPy JSON module
- [modules/re.py](/Users/tensorcraft/Projects/TensorPy/modules/re.py): TensorPy regex module
- [modules/math.py](/Users/tensorcraft/Projects/TensorPy/modules/math.py): TensorPy math module
- [modules/time.py](/Users/tensorcraft/Projects/TensorPy/modules/time.py): TensorPy time module
- [modules/random.py](/Users/tensorcraft/Projects/TensorPy/modules/random.py): TensorPy random module
- [modules/os.py](/Users/tensorcraft/Projects/TensorPy/modules/os.py): TensorPy os module
- [tests](/Users/tensorcraft/Projects/TensorPy/tests): ordered regression and feature tests

## Limitations

TensorPy is not yet a full Python implementation. Notable gaps still include:

- incomplete builtin and standard library coverage
- partial regex compatibility

## C API Status

TensorPy now has a minimal public embedding header:

- [include/tensorpy/api.h](/Users/tensorcraft/Projects/TensorPy/include/tensorpy/api.h)

That header is intentionally small and now exposes a frozen scalar-only Phase 1
embedding and host-extension surface.

The current readiness call is:

- third-party C extensions can target the scalar-only ABI in `api.h`
- public values and host-native callbacks are limited to `nil`, `bool`,
  `number`, `string`, and typed errors
- containers, objects, and callables are still intentionally out of scope for
  the public ABI until ownership rules are expanded and frozen

More detail is documented in:

- [docs/C_API_READINESS.md](/Users/tensorcraft/Projects/TensorPy/docs/C_API_READINESS.md)
- [docs/SYNTAX.md](/Users/tensorcraft/Projects/TensorPy/docs/SYNTAX.md)
- [docs/BUILTINS_AND_STDLIB.md](/Users/tensorcraft/Projects/TensorPy/docs/BUILTINS_AND_STDLIB.md)
- missing GC work for later phases
- incomplete Python compatibility for many edge cases and advanced syntax forms

## TODO

- expand data-structure method coverage, especially remaining `set`, `tuple`, `str`, and `bytes` behavior gaps
- broaden general builtins beyond the current reflection and iterable helpers
- extend module loading with richer search rules and better error reporting
- continue improving package semantics and nested module behavior
- strengthen exception compatibility, including richer exception objects and more Python-like error messages
- improve REPL behavior for multi-line input, block handling, and interactive error display
- expand `json` compatibility and validation behavior
- extend `re` support with groups, alternation, counted repetition, and flags
- add more large end-to-end language stress tests
- revisit garbage collection in a later phase

## Status

TensorPy is already useful for language and runtime experimentation, feature prototyping, and growing a test-backed Python-like interpreter. It is not yet a drop-in replacement for CPython or MicroPython.
