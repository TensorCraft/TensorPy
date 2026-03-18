# C API Readiness

This note separates TensorPy's current internal runtime from the small public
surface that is stable enough to expose to embedders.

## Public Surface Today

The new public header is:

- `include/tensorpy/api.h`

It currently exposes:

- `TPContext`
- `tpContextCreate()`
- `tpContextDestroy()`
- `tpContextInterpret(context, source, filename)`
- `tpInit()`
- `tpFree()`
- `tpInterpret(source, filename)`
- `TPResult`

This is intentionally small. It gives embedders a stable "create context / run
code / destroy context" entry point without exposing VM internals.

The new context handle is still a Phase 0 embedding surface:

- `TPContext` is opaque to callers
- only one active context is supported at a time
- the old `tpInit/tpFree/tpInterpret` helpers remain as compatibility wrappers
  over an internal default context

## Still Internal

These headers remain internal implementation detail for now:

- `vm.h`
- `object.h`
- `value.h`
- `table.h`
- `chunk.h`
- `compiler.h`
- `builtins.h`
- `scanner.h`
- `debug.h`

Reasons they are not ready for a public extension ABI yet:

- object layouts are still evolving
- GC behavior is new and may still need policy changes
- module/runtime ownership rules are not frozen as a public contract
- native call conventions are designed for builtins, not third-party modules

## Readiness Assessment

### Embedding API

TensorPy is now close enough to begin designing an embedding API.

Why:

- initialization and shutdown are explicit
- code execution has a stable top-level entry point
- runtime teardown and GC exist
- the interpreter no longer requires callers to know about `VM` directly
- callers can now hold an opaque runtime handle instead of binding to global VM
  details

### Extension Module ABI

TensorPy is not yet ready for a public C extension ABI.

What is still missing:

- a stable public object/value model
- reference/ownership rules for public values
- a native module registration interface
- a way to construct and return exceptions through a public API
- a stable story for containers, strings, and callable invocation from C

## Recommended Next Step

If we want to move toward external C integration, the next milestone should be:

1. keep `api.h` minimal and stable
2. define a tiny public value API for strings, numbers, booleans, and errors
3. decide whether Phase 1 embedding should support multiple simultaneous
   contexts or keep a single-active-context rule
4. only then sketch an extension-module registration ABI

That means:

- yes, we can begin designing an embedding layer
- no, we should not yet freeze a third-party C extension ABI
