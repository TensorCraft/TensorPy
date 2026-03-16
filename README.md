# TensorPy

TensorPy is a small Python-like interpreter written in C. It currently supports a useful subset of Python syntax, core data structures, exceptions, a simple module system, and a growing standard-library-style layer implemented in TensorPy itself.

This project is still in an active build-out phase. The goal right now is practical language coverage and runtime stability, not full CPython or MicroPython compatibility.

## Current Features

- Python-like syntax for expressions, functions, classes, conditionals, loops, slicing, and container literals
- Core data structures: `list`, `tuple`, `dict`, `set`, `str`, `bytes`
- Typed exceptions with `try` / `except`
- REPL plus file execution
- Basic module imports:
  - `import module`
  - `from module import name`
  - `from module import name as alias`
- Module lookup currently checks:
  - `modules/<name>.py`
  - `lib/<name>.py`
  - `./<name>.py`
- Builtin-like modules written in TensorPy:
  - `json`
  - `re`

## Implemented Library Surface

The runtime already includes a practical set of methods for:

- `list`
- `dict`
- `set`
- `tuple`
- `str`
- `bytes`

Examples include methods such as `append`, `pop`, `remove`, `sort`, `setdefault`, `popitem`, `union`, `intersection`, `count`, `index`, `split`, `join`, `encode`, `decode`, and `hex`.

The `json` module supports:

- `json.loads(...)`
- `json.dumps(...)`
- `json.JSON(...).parse()`

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
- `*`, `+`, `?`
- character classes like `[abc]`
- ranges like `[a-z]`
- negated classes like `[^0-9]`
- `\d`, `\w`, `\s`

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

At the time of writing, the suite contains `20` organized test files and passes in the current workspace.

## Project Layout

- [src/main.c](/Users/tensorcraft/Projects/TensorPy/src/main.c): CLI entry point and REPL
- [src/compiler.c](/Users/tensorcraft/Projects/TensorPy/src/compiler.c): parser and bytecode compiler
- [src/vm.c](/Users/tensorcraft/Projects/TensorPy/src/vm.c): virtual machine and runtime behavior
- [modules](/Users/tensorcraft/Projects/TensorPy/modules): TensorPy standard-library-style modules
- [modules/json.py](/Users/tensorcraft/Projects/TensorPy/modules/json.py): TensorPy JSON module
- [modules/re.py](/Users/tensorcraft/Projects/TensorPy/modules/re.py): TensorPy regex module
- [tests](/Users/tensorcraft/Projects/TensorPy/tests): ordered regression and feature tests

## Limitations

TensorPy is not yet a full Python implementation. Notable gaps still include:

- incomplete import/package semantics
- incomplete builtin and standard library coverage
- partial regex compatibility
- missing GC work for later phases
- incomplete Python compatibility for many edge cases and advanced syntax forms

## TODO

- expand data-structure method coverage, especially remaining `set`, `tuple`, `str`, and `bytes` behavior gaps
- add more general builtins such as `zip`, `map`, `filter`, `reversed`, `isinstance`, `getattr`, `setattr`, and `hasattr`
- improve module loading beyond the current minimal `import` / `from ... import ... as ...` support
- add package semantics and better module path resolution
- strengthen exception compatibility, including richer exception objects and more Python-like error messages
- improve REPL behavior for multi-line input, block handling, and interactive error display
- expand `json` compatibility and validation behavior
- extend `re` support with groups, alternation, counted repetition, and flags
- add more large end-to-end language stress tests
- revisit garbage collection in a later phase

## Status

TensorPy is already useful for language and runtime experimentation, feature prototyping, and growing a test-backed Python-like interpreter. It is not yet a drop-in replacement for CPython or MicroPython.
