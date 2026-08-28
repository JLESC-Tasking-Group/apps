# MNMGDatalog TC - OpenMP target-task port

An OpenMP port of `MNMGDatalog-reference/tc_benchmark` (CUDA). It computes the
Datalog-style semi-naive **Transitive Closure** fixpoint

```
path(a, c) :- path(a, b), edge(b, c).
```

on fixed pre-allocated device buffers, with all per-iteration **sizes resident in
device memory** (`d_frontier_size` / `d_new_count`). The kernels read those sizes
on the device, so the per-iteration kernel sequence is identical every round --
which is exactly what lets one recorded task graph be replayed.

## The two versions are compile-time toggles

Unlike the CUDA reference (separate `v1_baseline/` and `v2_cudagraph/` sources),
this is **one source** (`tc.cpp`) whose backend is chosen by the shared toggles in
`../tasking.h` / `../common.mk`. The two CUDA versions map to `USE_TASKGRAPH`:

| CUDA reference        | Build here                | Fixpoint driver                                   |
|-----------------------|---------------------------|---------------------------------------------------|
| `v1_baseline`         | `make USE_TASKGRAPH=0`    | host loop spawns the kernels every iteration      |
| `v2_cudagraph`        | `make USE_TASKGRAPH=1`    | per-iteration body recorded once, then replayed   |

`v3_conditional` (the on-GPU conditional WHILE loop) is **not** ported: CGIR/XKOMP
has no conditional-node support yet. The remaining toggles are orthogonal:

- `USE_TARGET=1` GPU offload (target tasks) vs `USE_TARGET=0` host CPU tasks.
- `USE_SYNC=1` synchronous blocking schedule (each kernel runs to completion).

The per-iteration body `reset -> expand -> promote -> set_sizes` is wrapped in
`TASKGRAPH_BEGIN/END` together with a `k_writeback` async D2H of `new_count`; the
host then tests convergence on that (pinned) host scalar after a `taskwait` -- the
same host round-trip the CUDA `v2_cudagraph` pays. Eliminating that round-trip is
what `v3_conditional` would do on the GPU.

## Design notes

- **Device memory:** the large buffers (`edges`, `edge_table`, `result_set`, the
  two frontiers) are device-only via `omp_target_alloc`, reached through
  `is_device_ptr(...)` -- the direct analog of the reference's `cudaMalloc`, with
  no host mirror of the (possibly multi-GB) result set. On the CPU backend the
  same pointers are `malloc` and the kernels become host tasks.
- **`new_count` is the exception:** it is the one scalar the host reads *every*
  iteration, so it is pinned host memory from the shared allocator
  (`../alloc.h`, `host_alloc`/`host_free`) with a device copy created by
  `map(alloc:)` and reached from the kernels with `map(present:)` -- exactly how
  the Krylov solvers handle their scalars. `k_writeback` refreshes the host side
  with an `omp target update from(...) nowait` recorded *inside* the taskgraph, so
  the timed loop contains no blocking `omp_target_memcpy`. The remaining scalars
  (`frontier_size`, `result_count`, `overflow`) stay device-only and are read once
  after the fixpoint.
- **The `taskwait` after `TASKGRAPH_END` is required.** Under `USE_TASKGRAPH=1`
  the region is already effectively blocking (xkomp does an implicit taskwait
  while recording, and replay is synchronous), so it is free; but with
  `USE_TASKGRAPH=0` the macros vanish and the `nowait` tasks would still be in
  flight, so without it the loop would read a stale `new_count` and stop early.
- **Atomics:** the open-addressing result set and edge table are built with a
  compare-and-swap (`#pragma omp atomic compare capture`, OpenMP 5.1); the append
  counters use fetch-add (`#pragma omp atomic capture`). One portable code path
  for host and device.
  **Caveat:** on-device `omp atomic compare capture` codegen must be confirmed on
  your XKOMP/clang + NVPTX toolchain (this is the one construct I could not verify
  without a GPU). A reduction cannot replace it -- an open-addressing insert is a
  *slot claim* ("insert iff empty, report novelty"), not an associative combine.

## Build & run

```shell
make USE_TARGET=1 USE_TASKGRAPH=1        # GPU, replayed graph (v2_cudagraph)
make USE_TARGET=1 USE_TASKGRAPH=0        # GPU, host loop      (v1_baseline)
make                                     # CPU tasks (default), for correctness

./tc.x <data.bin> [capacity_mult] [repeats] [frontier_slots]
make test                                # data_7035.bin (TC=146120, 64 iters)
```

Environment: `TC_WRITE=1` writes `<input>_<version>_tc.bin` (off by default so
sweeps stay clean); `TC_DUMP=<f>` writes a `src dst` text dump; `TC_CSV=<f>` writes
the reference's 15-column metric row.

## Capacity

The result set is sized `next_pow2(n_edges * capacity_mult)` and must be
**>= ~2x the TC size**, or the run aborts fast with an overflow message (raise
`capacity_mult`, arg 2). Frontier buffers are decoupled (`min(result_cap, 2^28)`
slots each), so only the set grows with TC.

## Correctness

Known reference sizes (from the reference README): `data_10` -> TC 18 / 3 iters,
`data_7035` -> 146120 / 64, `data_23874` -> 481121 / 58. A build with
`USE_TASKGRAPH=0` and one with `USE_TASKGRAPH=1` must produce the identical TC
size, iteration count, and (via `TC_DUMP`) tuple set.

## Evaluation harness

Registered as app `mnmg` in `../scripts/appspecs.py`; datasets are `data_<N>.bin`
where `N` = edge count, so `evaluate.py --apps mnmg --sizes 7035,23874` maps sizes
to files. `evaluate.py` sweeps the synchronous / no-taskgraph / taskgraph configs
just like the other apps.
