# TensorPy Roadmap

This file tracks the current project state as a practical checklist. Checked items are already landed on `main`. Unchecked items are the next planned steps, not vague ideas.

## Language Core

- [x] Python-like expressions, statements, loops, slicing, and container literals
- [x] Functions, lambdas, closures, and `*args`
- [x] Classes, inheritance, and `super()` method calls
- [x] Exceptions with `try` / `except`
- [x] Module imports, package imports, and module caching
- [x] REPL and script execution
- [ ] More complete keyword-argument and star-argument parity with Python
- [ ] Deeper compatibility coverage for advanced class and descriptor behavior

## Runtime And Memory

- [x] Mark-and-sweep garbage collector
- [x] VM root walking and heap edge traversal
- [x] Platform abstraction layer for filesystem, threads, timing, and process-facing operations
- [x] Opaque platform handles instead of leaking pthread types in public headers
- [x] Optional Metal build via `make METAL=0`
- [x] Initial shared memory abstraction via `tpMemAlloc` / `tpMemFree` / friends
- [ ] Finish routing remaining raw allocations through the memory layer
- [ ] Split time APIs into clearer wall-clock vs CPU-time abstractions
- [ ] Harden portability for non-Apple platforms beyond the current POSIX-first baseline

## Concurrency And Systems Primitives

- [x] Threads
- [x] Mutexes
- [x] Condition variables
- [x] Atomics
- [x] Thread pool
- [x] `parallel_for`
- [ ] Add more stress and race-oriented regression tests
- [ ] Expose more concurrency primitives ergonomically at the TensorPy layer

## Tensor And Compute Runtime

- [x] Native `tensor`, `dtype`, and `device` objects
- [x] CPU float32 eager ops
- [x] CPU scalar, SIMD, and threaded compute paths for core kernels
- [x] Tensor reshape, cast, transfer, reduction, activation, and `matmul`
- [x] CPU autograd subset for practical training loops
- [x] `sgd_step` and `adam_step`
- [x] Metal tensor allocation and transfer
- [x] Metal elementwise `fill`, `add`, `mul`, scalar `add`, scalar `mul`
- [x] Metal `matmul` for 2D float32 tensors
- [ ] Metal reductions
- [ ] Metal `softmax`
- [ ] Metal `layernorm`
- [ ] Better async scheduling to reduce `waitUntilCompleted` overhead
- [ ] More aggressive CPU `matmul` optimization, likely via Accelerate / BLAS on macOS

## NN And Training Stack

- [x] `Module` base class
- [x] Recursive parameter registration
- [x] `Linear`, `Conv2d`, `ReLU`, `Sigmoid`, `Tanh`, `Flatten`, `Sequential`
- [x] `Embedding`
- [x] `RNNCell`, `RNN`
- [x] `LSTMCell`, `LSTM`
- [x] `GRUCell`, `GRU`
- [x] `LogisticRegression`, `MLP`, `SimpleCNN`
- [x] `Adam`
- [ ] `CrossEntropyLoss`
- [ ] `Dropout`
- [ ] `LayerNorm` module wrapper
- [ ] `train()` / `eval()`
- [ ] More end-to-end recurrent training tests on real datasets

## Standard Library Surface

- [x] `json`
- [x] `re`
- [x] `math`
- [x] `time`
- [x] `random`
- [x] `os`
- [x] `io`
- [x] `path`
- [x] `logging`
- [x] `traceback`
- [x] `sys`
- [x] `collections`
- [x] `itertools`
- [x] `functools`
- [x] `env`
- [x] `config`
- [x] `host`
- [x] `array`
- [x] `ml`
- [x] `types`
- [x] `inspect`
- [ ] Continue filling Python-adjacent utility gaps where they meaningfully improve usability

## Near-Term Priorities

- [ ] Benchmark and improve CPU `matmul`
- [ ] Reduce Metal synchronization overhead
- [ ] Add more true Metal kernels for training-critical ops
- [ ] Expand training regression coverage, including MNIST-style smoke tests
- [ ] Keep the repository clean of generated datasets and local test binaries
