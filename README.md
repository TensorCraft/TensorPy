# TensorPy

TensorPy is a small Python-like interpreter written in C. It currently supports a useful subset of Python syntax, core data structures, exceptions, a simple module system, and a growing standard-library-style layer implemented in TensorPy itself.

This project is still in an active build-out phase. The goal right now is practical language coverage and runtime stability, not full CPython or MicroPython compatibility.

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
  - `types`
  - `inspect`
- Platform-facing runtime operations are routed through a portability layer in `src/platform.c`

## Milestone Status

- `Phase 1`: completed
  - common builtins expanded
  - reflection helpers added
  - `math`, `time`, `random`, and `os` modules added
  - platform abstraction layer added for system-facing operations
- `Phase 2`: completed
  - module globals are isolated from script globals
  - module cache is active across repeated imports
  - package imports now support `pkg`, `pkg.mod`, and `from pkg.mod import name`
  - package submodules are attached back onto parent package objects
- `Phase 3`: completed
  - exception matching and exception objects were made more Python-like
  - function call argument checking and keyword handling were tightened
  - nested functions now capture outer bindings
  - `json` validation and `re` alternation support received a second pass
  - REPL multi-line block input was improved
- `Phase 4`: completed
  - focus: GC readiness and memory model stabilization
  - object environment and closure ownership have been normalized ahead of GC
  - a non-collecting mark walker now traverses VM roots and heap object edges
  - object finalization and VM teardown now free heap-owned buffers and objects
  - an explicit mark-and-sweep collector now reclaims unreachable heap objects
  - a conservative automatic GC trigger now runs from object allocation thresholds
  - next step: return to language and compatibility work on top of the stabilized runtime

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

The V1 utility modules also include:

- `io.read_text(path)` / `io.write_text(path, text)`
- `io.read_bytes(path)` / `io.write_bytes(path, bytes)`
- `path.join(...)`, `path.normpath(...)`, `path.abspath(...)`
- `path.basename(...)`, `path.dirname(...)`, `path.splitext(...)`
- `logging.debug/info/warn/error(...)`
- `logging.getLogger(name)`
- `traceback.format_exception(exc)` / `traceback.print_exception(exc)`
- minimal `sys` metadata: `implementation`, `version`, `version_info`, `platform`, `argv`
- `collections.Counter`, `collections.defaultdict`, `collections.flatten`, `collections.chunked`
- `itertools.chain`, `itertools.repeat`, `itertools.take`, `itertools.batched`
- `functools.partial`, `functools.compose`
- `env.get`, `env.exists`, `env.require`
- `config.load`, `config.loads`, `config.get`, `config.require`, `config.merge`
- `host.set`, `host.get`, `host.has`, `host.call`
- `array.zeros`, `array.full`, `array.shape`, `array.add`, `array.mul`, `array.matmul`
- `types.type_name` plus basic `is_*` helpers
- `inspect.type_name`, `inspect.is_callable`, `inspect.is_function`, `inspect.is_class`, `inspect.is_module`

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
