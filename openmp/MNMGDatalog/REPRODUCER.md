# `-O0` OpenMP GPU offload faults with `CUDA_ERROR_ILLEGAL_ADDRESS`

A `#pragma omp target teams distribute parallel for` whose body makes **device
function calls** faults with `CUDA_ERROR_ILLEGAL_ADDRESS` (700) once roughly
10^5 threads are concurrently inside the call chain — but **only when compiled
`-O0`**. The identical source at `-O3`, where the callees are inlined, runs
clean at every size tested.

Reproducer: [`tc_repro.cpp`](tc_repro.cpp) — ~150 lines, no atomics required, no
out-of-bounds access, no tasking, no taskgraph, no dependences, plain blocking
`omp target` regions.

## Environment

| | |
|---|---|
| host | JLSE `grace01`, NVIDIA GH200 480 GB (102 GB device memory), aarch64 |
| driver | `nvidia-compute-G07-610.43.02` |
| CUDA | 13.3.1 |
| compiler | `clang++` fork (`llvm-project`, `21.0git`), `-fopenmp-version=60`, `--offload-arch=sm_90` |
| runtime | `libxkomp` (replaces `libomp`) + `xkrt`, custom `libomptarget` (`offload/libomptarget/xktarget.cpp`) |

Not yet checked against a stock `libomp` / `libomptarget`; the fault may or may
not be specific to this runtime. The kernel bodies and the launch path
(`cuLaunchKernel` with the geometry the plugin's `getEffectiveNumBlocks`
computes) are otherwise standard.

## Build and run

```shell
cd apps/openmp/MNMGDatalog

# failing build
make OPT="-O0 -g" USE_TARGET=1 USE_SYNC=1 USE_TASKGRAPH=0 repro
./tc_repro.x 1000000 4194304 2097152 0        # keys slots workers mode

# passing build -- identical source
make OPT="-O3"    USE_TARGET=1 USE_SYNC=1 USE_TASKGRAPH=0 repro
./tc_repro.x 1000000 4194304 2097152 0
```

Failure:

```
round 0: fill reset expand [0.67] [LOGGER] [FATAL] `cuStreamSynchronize(stream)`
   failed with `an illegal memory access was encountered` (700)
```

`fill` and `reset` complete; the fault is always in the third kernel (`expand`).

## The kernel

```c
RP_TEAMS_LOOP                            /* omp target teams distribute parallel for
                                            is_device_ptr(set, out, cnt, flg, dn) */
for (int t = 0; t < workers; t++) {
    const int n = dn[0];                 /* device-resident loop size */
    for (int i = t; i < n; i += workers)
        repro_one(i, set, slots, out, ocap, cnt, flg, mode);
}
```

With `mode = 0`, `repro_one` performs no atomics at all:

```c
u64 key = ((u64)(unsigned)i * 0x9e3779b97f4a7c15ULL) | 1ULL;
set[repro_hash(key, cap)] = key;         /* cap is a power of two; hash masks with cap-1 */
if ((unsigned)i < (unsigned)ocap) out[i] = key;
```

Every index is provably in range: `repro_hash` returns `k & (cap - 1)`, and the
`out` store is explicitly guarded. `workers > n`, so each `t < n` executes the
inner body exactly once and each `t >= n` executes it zero times.

## Matrix

`keys = 1000000`, `slots = 4194304` (32 MiB table), `mode = 0` unless stated.

| `-O` | workers | threads/block | result |
|---|---:|---:|---|
| `-O0 -g` | 4 096 | 32 | pass (8 rounds) |
| `-O0 -g` | 65 536 | 32 | pass (8 rounds) |
| `-O0 -g` | 1 048 576 | 128 | **fault, round 0** |
| `-O0 -g` | 2 097 152 | 128 | **fault, round 0** |
| `-O3` | 4 096 / 65 536 / 1 048 576 / 2 097 152 | 32 / 32 / 128 / 128 | pass |
| `-O3` | 102 400 / 262 144 / 409 600 / 524 288 / 786 432 | — | pass |

Mode sweep at `-O0 -g`, `workers = 2097152`:

| mode | insert | append | result |
|---|---|---|---|
| 0 | plain store | indexed store | **fault** |
| 1 | `omp atomic compare capture` | indexed store | **fault** |
| 2 | plain store | `omp atomic capture` | **fault** |
| 3 | `omp atomic compare capture` | `omp atomic capture` | **fault** |

`LIBOMPTARGET_STACK_SIZE` at `-O0 -g`, `workers = 2097152`, `mode = 0`:

| value | result |
|---|---|
| 8 192 | fault |
| 16 384 | fault |
| 65 536 | fault |
| 262 144 | fault |

`LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=128` with `workers = 4096`
(forcing 128 threads/block at only 32 blocks) passes at `-O3`; not retested at
`-O0`.

## Ruled out

- **Atomics** — `mode 0` uses none and still faults.
- **Out-of-bounds indexing** — see above; every subscript is masked or guarded.
- **Buffer size** — the same 32 MiB table passes at 65 536 workers and faults at
  1 048 576. In the parent application, an 8 MiB configuration faults while a
  16 MiB one passes.
- **Device stack** — `LIBOMPTARGET_STACK_SIZE` up to 256 KiB changes nothing.
- **Tasking, taskgraph, `depend`, the async command queue** — the reproducer has
  none of them; the parent application also faults with `USE_SYNC=1`.
- **Allocator** — a 16 GiB `omp_target_alloc` buffer is filled and read back
  correctly by the parent application in the same run.
- **Launch geometry alone** — the `fill` kernel in the same program runs with the
  same 3 200 × 128 geometry and never faults (see below).

## The correlation

Only kernels whose body makes **device calls** fault. In the parent application
(`tc.cpp`, `TC_TRACE=1` output) at 1 048 576 workers:

| kernel | device call chain at `-O0` | result |
|---|---|---|
| `fill_edge_table`, `fill_result_set` | none (body inline) | always pass |
| `k_promote` (`fr[i] = nf[i]`) | none (body inline) | pass |
| `k_expand` | `tc_expand_one` → `tc_set_insert` → `tc_cas_u64` | **fault** |
| `repro_one` mode 0 | `repro_one` → `repro_insert` → `repro_hash` | **fault** |

and the trigger is the *concurrency* of callers, not their number. In `tc.cpp`
the frontier grows each round and the fault appears at a threshold:

```
round 0: frontier ~49 152   -> pass
round 1: frontier  81 668   -> pass
round 2: frontier 113 978   -> FAULT
```

while the reproducer processes the same 1 000 000 keys fine at 65 536 workers
(32 threads/block) and faults at 1 048 576 (128 threads/block).

## Workaround

Build device code at `-O3`, or force the helpers inline. Both `tc.cpp` and
`tc_repro.cpp` now mark every `declare target` helper
`static inline __attribute__((always_inline))` (`TC_DEVFN` / `RP_DEVFN`) so the
call chain cannot survive to codegen at any `-O` level.
`apps/openmp/common.mk` defaults to `-O3` and warns when `-O0` is combined with
`USE_TARGET=1`.

## Application-level manifestation

`apps/openmp/MNMGDatalog/tc.cpp` (Datalog transitive closure) faulted on every
input above ~24 000 edges for the entire investigation, at `-O0 -g`. At `-O3` all
of it passes and reproduces the published closure sizes:

| dataset | edges | rounds | TC | matches reference |
|---|---:|---:|---:|---|
| OL.cedge | 7 035 | 64 | 146 120 | yes |
| TG.cedge | 23 874 | 58 | 481 121 | yes |
| p2p-Gnutella31 | 147 892 | 31 | 884 179 859 | yes |

with `USE_TASKGRAPH=0` and `USE_TASKGRAPH=1` producing identical results.
