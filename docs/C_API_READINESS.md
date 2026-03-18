# C API Readiness

This note separates TensorPy's internal runtime from the public surface that is
stable enough to expose to embedders and early third-party native extensions.

## Public Surface Today

The new public header is:

- `include/tensorpy/api.h`

It currently exposes:

- `TPContext`
- `TPValue`
- `TENSORPY_API_VERSION`
- `TENSORPY_EXTENSION_ABI_VERSION`
- `tpContextCreate()`
- `tpContextDestroy()`
- `tpContextInterpret(context, source, filename)`
- `tpContextGetGlobal(context, name, out)`
- `tpContextSetGlobal(context, name, value)`
- `tpContextGetLastError(context, out)`
- `tpContextCreateModule(context, name)`
- `tpModuleDestroy(module)`
- `tpModuleAddValue(module, name, value)`
- `tpModuleAddFunction(module, name, function, userData)`
- `tpApiVersion()`
- `tpExtensionAbiVersion()`
- `tpInit()`
- `tpFree()`
- `tpInterpret(source, filename)`
- `TPResult`

This is intentionally small. It gives embedders a stable "create context / run
code / destroy context" entry point without exposing VM internals.

The current contract is now a documented Phase 1 scalar-only embedding and
extension surface:

- `TPContext` is opaque to callers
- only one active context is supported at a time
- `TPValue` currently supports only nil, booleans, numbers, strings, and errors
- `TPValue` errors can carry an explicit exception type plus message
- host-native modules can currently expose only scalar values and scalar-only
  native functions
- `TPModule` is a registration handle; the VM owns the module object after
  registration, and embedders can free the handle after setup
- the public header now exposes explicit API / extension ABI version markers
- the old `tpInit/tpFree/tpInterpret` helpers remain as compatibility wrappers
  over an internal default context

## Phase 1 Scalar Contract

TensorPy can now treat the scalar-only boundary as the first stable
third-party-facing C extension milestone.

The frozen scope for `TENSORPY_EXTENSION_ABI_VERSION == 1` is:

- one active `TPContext` at a time
- host-visible values limited to `nil`, `bool`, `number`, `string`, and error
- host-native functions receiving only scalar arguments
- host-native functions returning only scalar values or typed errors
- module registration through `tpContextCreateModule()`,
  `tpModuleAddValue()`, and `tpModuleAddFunction()`

Ownership rules for this scope are:

- every `TPValue` must be initialized with `tpValueInit()` before use and
  released with `tpValueFree()` when the caller no longer needs it
- `tpValueSetString()`, `tpValueSetError()`, and `tpValueSetException()`
  deep-copy caller-provided strings
- `tpContextGetGlobal()` and `tpContextGetLastError()` deep-copy runtime scalar
  data into the caller-provided `TPValue`
- `tpContextSetGlobal()` and `tpModuleAddValue()` copy the supplied scalar data
  into the runtime; callers may immediately reuse or free the input `TPValue`
- `TPNativeFn` arguments are borrowed read-only views that are valid only for
  the duration of the callback
- the `TPNativeFn` result slot is owned and cleaned up by TensorPy after the
  callback returns

Non-goals for this frozen scope:

- passing lists, dicts, tuples, sets, bytes, instances, classes, modules, or
  closures across the public boundary
- retaining borrowed callback arguments after the callback returns
- any guarantee about simultaneous multiple contexts

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
- embedders can exchange simple globals and inspect the last runtime error
  without including internal headers

### Extension Module ABI

TensorPy is now ready for a first frozen scalar-only third-party native module
ABI.

What is frozen in this first milestone:

- the API and ABI version markers in `api.h`
- the opaque `TPContext` lifecycle
- scalar-only `TPValue` exchange
- typed error return through `TPValue.exceptionType`
- scalar-only host-module registration and callback flow

What is intentionally still out of scope:

- any public object/value model beyond scalars
- container and callable ownership across the public boundary
- container construction and iteration helpers
- invoking TensorPy callables from C
- exposing runtime object layouts to embedders

What is still missing:

- a stable public object/value model beyond scalars
- reference/ownership rules for container and callable values
- richer public exception construction beyond `(type, message)` helpers
- a stable story for containers and callable invocation from C
- future versioning rules once non-scalar exchange is introduced

## Recommended Next Step

If we want to move beyond this first scalar-only ABI, the next milestone should
be:

1. keep `api.h` minimal and stable
2. decide whether Phase 1 embedding should support multiple simultaneous
   contexts or keep a single-active-context rule
3. expand the public value model beyond scalars if embedders need containers or
   callable exchange
4. expand native-module registration from Phase 0 helpers into a documented
   frozen ABI contract only after those value rules settle
5. only then freeze an extension-module registration ABI

That means:

- yes, TensorPy is ready for a first push as a scalar-only third-party C
  extension target
- no, the extension surface is not yet broad enough to promise container or
  callable exchange across the C boundary
