# GC Readiness Notes

This document records the current object graph and intended GC roots for
TensorPy after the environment/closure refactor.

## Current State

- Function objects (`ObjFunction`) no longer own execution environments.
- Closures (`ObjClosure`) now carry the pair:
  - executable function
  - captured environment object
- Execution environments are first-class objects (`ObjEnvironment`).
- Module globals can be wrapped by an environment object without copying the
  backing table.

This removes the previous "clone function and attach raw Table*" model and
gives the runtime a much clearer ownership story for closures.

## Intended GC Roots

The following values/objects must be treated as roots during marking:

- `vm.stack[0..stackTop)`
- `vm.frames[i].closure`
- `vm.frames[i].function`
- `vm.frames[i].env`
- `vm.handlers[i].expectedType`
- `vm.globalEnv`
- `vm.initString`
- `vm.moduleClass`
- values reachable from `vm.modules`
- keys reachable from `vm.strings`

## Object Graph Edges

### `ObjClosure`

- `function`
- `env`

### `ObjEnvironment`

- `parent`
- all keys and values reachable from `table`

If `ownsTable` is false, the table storage itself is externally owned, but the
GC still needs to trace all object references stored inside it.

### `ObjFunction`

- `defaults`
- `paramNames`
- `localNames`
- `chunk.constants`
- `name`

`ObjFunction` no longer points at any environment object.

### `ObjClass`

- `name`
- `superClass`
- all keys and values in `methods`

### `ObjInstance`

- `klass`
- all keys and values in `fields`

### `ObjBoundMethod`

- `receiver`
- `method`

The `method` value may be a closure or another callable object.

### Container Objects

- `ObjList`: all items
- `ObjTuple`: all items
- `ObjDict`: all keys and values
- `ObjSet`: all keys
- `ObjIterator`: `iterable`
- `ObjSlice`: `start`, `stop`, `step`

### Bytes and Strings

- `ObjString`: no object children
- `ObjBytes`: no object children

## Remaining Non-Object Storage To Account For

The runtime now has explicit teardown for the main raw allocations owned by
heap objects and chunks:

- `Chunk` code and line arrays are freed through `freeChunk()`
- `ValueArray` buffers owned by functions, lists, and tuples are freed
- byte buffers for `ObjBytes` are freed
- character buffers for `ObjString` are freed
- owned `Table` storage inside objects and environments is freed

There are still a few VM-level raw allocations that a future collector must
manage carefully:

- `vm.modules` table storage
- `vm.strings` table storage

These are no longer blockers for sweep, but they still define the boundaries
that collection must preserve while reclaiming unreachable objects.

## Current Mark Walker

The runtime now includes a non-collecting mark walker that:

1. marks roots from the VM
2. traces object edges using the rules above
3. can report reachable object counts for the full VM root set or a single
   starting value

This is intentionally a verification/debugging step only. It does not reclaim
memory.

## Current Collector State

TensorPy now has a mark-and-sweep collector with both explicit and conservative
automatic triggering:

1. mark from VM roots
2. remove white interned strings from `vm.strings`
3. sweep `vm.objects` and free unmarked heap objects
4. reschedule the next collection threshold based on surviving heap objects

Automatic collection is paused during compilation/bootstrap so unrooted
compiler-owned objects are not collected before they enter VM roots.

## Recommended Next GC Step

Tune policy and ergonomics:

1. evaluate whether the current threshold heuristic is aggressive enough
2. consider byte-based accounting in addition to object-count thresholds
3. decide whether the current behavior is sufficient to close Phase 4

With mark traversal, finalization, sweeping, and safe automatic triggering now
in place, the remaining work is policy refinement rather than core GC
mechanics.
