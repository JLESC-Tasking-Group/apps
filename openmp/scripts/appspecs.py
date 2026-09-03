"""
Application registry for the apps/openmp evaluation harness.

Each AppSpec describes, for one benchmark app, how to (i) build a given
configuration, (ii) run one problem size, and (iii) parse the metrics it prints.
scripts/evaluate.py drives the sweep (app x variant x config x size) from these.

Configurations
--------------
Every app is always compared across these reference configurations plus one per
user-supplied CGIR optimization combo:

  * synchronous   : USE_SYNC=1, USE_TASKGRAPH=0, one task/kernel per loop
                    -- the "current practice" blocking baseline.
  * no-taskgraph  : USE_SYNC=0, USE_TASKGRAPH=0
                    -- plain OpenMP tasks/target (taskgraph overhead reference).
  * taskgraph:none: USE_TASKGRAPH=1, OMP_TASKGRAPH_OPT=""  (no CGIR pass).
  * taskgraph:<opt>: USE_TASKGRAPH=1, OMP_TASKGRAPH_OPT="<opt>" for each combo.

The synchronous / no-taskgraph / taskgraph split is a *compile-time* choice
(USE_SYNC / USE_TASKGRAPH), so evaluate.py rebuilds per configuration; the CGIR
pass within a taskgraph build is a *run-time* choice (OMP_TASKGRAPH_OPT).
The GPU vs CPU backend (USE_TARGET) is orthogonal and chosen once (--target).
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# CGIR optimization combos compared on top of the references (each -> one
# taskgraph:<opt> configuration). Names are XKOMP OMP_TASKGRAPH_OPT passes
# (comma or space separated): copy-normalize copy-fuse reduce-node transitive-reduction
# jit prog-fuse sequence batch.
DEFAULT_OPTS = [
    "reduce-node,transitive-reduction",
    "reduce-node,transitive-reduction,batch",
]


@dataclass
class Config:
    label: str                 # legend label, e.g. "taskgraph:reduce-node,transitive-reduction"
    build: Dict[str, str]      # make variables, e.g. {"USE_SYNC": "1", ...}
    opt: Optional[str]         # OMP_TASKGRAPH_OPT value (None if no taskgraph)
    grain1: bool = False       # run with one task/kernel per loop (sync baseline)


def default_configs(opts: List[str]) -> List[Config]:
    cfgs = [
        Config("synchronous",    {"USE_SYNC": "1", "USE_TASKGRAPH": "0"}, None, grain1=True),
        Config("no-taskgraph",   {"USE_SYNC": "0", "USE_TASKGRAPH": "0"}, None),
        Config("taskgraph:none", {"USE_SYNC": "0", "USE_TASKGRAPH": "1"}, ""),
    ]
    for o in opts:
        cfgs.append(Config(f"taskgraph:{o}", {"USE_SYNC": "0", "USE_TASKGRAPH": "1"}, o))
    return cfgs


# --------------------------------------------------------------------------- #
# Metric parsing helpers.
# --------------------------------------------------------------------------- #
_F = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return (m, s)


def _parse_krylov(text):
    return {
        "avg_ms":    _grab(text, r"(?:iterations|restarts) 2\.\..*\(avg\)\s*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"(?:iterations|restarts) 2\.\..*\(stddev\)\s*:\s*" + _F + r"\s*ms"),
        "iter0_ms":  _grab(text, r"(?:iteration|restart) 0.*:\s*" + _F + r"\s*ms"),
        "elapsed_s": _grab(text, r"total solve time\s*:\s*" + _F),
        "flops":     _grab(text, r"theoretical flops\s*:\s*" + _F),
        "gflops":    _grab(text, r"performance\s*:\s*" + _F),
        "residual":  _grab(text, r"relative residual\s*:\s*" + _F),
        "error":     _grab(text, r"relative error\s*:\s*" + _F),
    }


def _parse_lulesh(text):
    return {
        "avg_ms":    _grab(text, r"iterations 2\.\..*\(avg\)\s*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"iterations 2\.\..*\(stddev\)\s*:\s*" + _F + r"\s*ms"),
        "iter0_ms":  _grab(text, r"iteration 0\s*:\s*" + _F + r"\s*ms"),
        "elapsed_s": _grab(text, r"Elapsed time\s*=\s*" + _F),
        "fom":       _grab(text, r"FOM\s*=\s*" + _F + r"\s*\(z/s\)"),
    }


def _parse_mnmg(text):
    # tc.cpp reports per-ROUND times in the same shape as the Krylov drivers (see
    # MNMGDatalog/tc.cpp): round 0 records the task graph, round 1 is the first
    # replay (and where the command graph is built), rounds 2.. are steady state.
    # "total time (end-to-end)" is the MNMGDatalog paper's metric (file IO + H2D +
    # setup + compute + D2H) and is printed in ms, unlike the other apps' seconds.
    avg = _grab(text, r"rounds \d+\.\.\d+ \(avg\)\s*:\s*" + _F + r"\s*ms")
    std = _grab(text, r"rounds \d+\.\.\d+ \(stddev\)\s*:\s*" + _F + r"\s*ms")
    it0 = _grab(text, r"round 0[^:]*:\s*" + _F + r"\s*ms")
    if avg is None:
        # fewer than 3 rounds: no steady-state window, fall back to the last
        # round that was reported.
        avg = _grab(text, r"round 1[^:]*:\s*" + _F + r"\s*ms")
        if avg is None:
            avg = it0
        std = 0.0 if avg is not None else None
    total_ms = _grab(text, r"total time \(end-to-end\)\s*:\s*" + _F + r"\s*ms")
    return {
        "avg_ms":    avg,
        "stddev_ms": std,
        "iter0_ms":  it0,
        "elapsed_s": (total_ms / 1000.0) if total_ms is not None else None,
    }


def _parse_llmc(text):
    # llm.c prints one "Iteration runtime : X ms" per step (to stderr); compute
    # the avg/stddev over the steady steps (drop step 0 = warmup/first build).
    runs = [float(x) for x in re.findall(r"Iteration runtime\s*:\s*" + _F + r"\s*ms", text)]
    steady = runs[1:] if len(runs) > 1 else runs
    avg, std = _mean_std(steady)
    return {
        "avg_ms":    avg,
        "stddev_ms": std,
        "iter0_ms":  runs[0] if runs else None,
        "elapsed_s": _grab(text, r"Took\s+" + _F + r"\s*s"),
    }


# --------------------------------------------------------------------------- #
# App registry.
# --------------------------------------------------------------------------- #
@dataclass
class AppSpec:
    name: str
    directory: str                       # relative to apps/openmp/
    variants: List[str]                  # [""] when a single binary
    make_target: Callable[[str], str]    # variant -> make target
    binary: Callable[[str], str]         # variant -> ./binary (run cwd = directory)
    run_args: Callable                   # (variant, size, iters, cfg) -> [args]
    parse: Callable[[str], dict]         # stdout+stderr -> metrics dict
    work: Callable[[int], tuple]         # size -> (value, label) for the top axis
    sizes: List[int]
    iters: int
    rebuild_per_size: bool = False       # llm.c: size is a compile-time macro
    llmc_defs: Optional[Callable] = None # (size, iters, batch) -> LLMC_DEFS string
    batch: int = 4                       # llm.c BATCH_SIZE (for tokens = B*T)


# ---- krylov: grid n, matrix N=n^3, work ~ n^3; -t/-s = task counts (0=threads) --
# grain is a positional list [T1, T2]: T1 -> -t (tasks per vector op), T2 -> -s
# (SpMV sub-tasks per block, default 1 so every loop has T1 tasks). None -> the
# app default (-t 0 -s 0 = auto, i.e. omp threads). Sync is always 1/loop (-t 1 -s 1).
def _krylov_run(variant, size, iters, cfg, grain):
    if cfg.grain1:
        t, s = "1", "1"
    elif grain:
        t = str(grain[0])
        s = str(grain[1]) if len(grain) > 1 else "1"
    else:
        t, s = "0", "0"
    return ["-n", str(size), "-i", str(iters), "-t", t, "-s", s, "-S", "27"]

KRYLOV = AppSpec(
    name="krylov",
    directory="krylov",
    variants=["cg", "cr", "bicgstab", "minres", "gmres"],
    make_target=lambda v: v,
    binary=lambda v: f"./{v}.x",
    run_args=_krylov_run,
    parse=_parse_krylov,
    work=lambda n: (float(n) ** 3, "n\u00b3 (\u221d FLOPs)"),
    sizes=[32, 48, 64],
    iters=50,
)

# ---- lulesh: mesh side s, zones = s^3; -nb = tasks per loop ---------------------
# grain is a positional list [nb]: nb -> -nb (tasks per loop). None -> the app
# default (-nb 32). The synchronous config is always 1 task/loop (-nb 1).
def _lulesh_run(variant, size, iters, cfg, grain):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-s", str(size), "-r", "11", "-b", "1", "-c", "1", "-nb", nb]

LULESH = AppSpec(
    name="lulesh",
    directory="lulesh",
    variants=[""],
    make_target=lambda v: "lulesh",
    binary=lambda v: "./lulesh",
    run_args=_lulesh_run,
    parse=_parse_lulesh,
    work=lambda s: (float(s) ** 3, "zones (s\u00b3)"),
    sizes=[16, 32, 48, 64],
    iters=30,
)

# ---- llm.c: SEQUENCE_SIZE T is a compile-time macro -> rebuild per size ---------
# grain is a positional list mapping to the compile-time granularity macros
# [GRAN_TMP, OC_SPLIT, OC_BACK_SPLIT] (extras ignored); injected at build time
# rather than as run arguments. None -> the source defaults.
_LLMC_GRAIN_MACROS = ["GRAN_TMP", "OC_SPLIT", "OC_BACK_SPLIT"]

def _llmc_run(variant, size, iters, cfg, grain):
    return []  # no runtime args; size/steps/grain are compiled in

def _llmc_defs(size, iters, batch, grain):
    defs = f"-DSEQUENCE_SIZE={size} -DNB_STEPS={iters} -DBATCH_SIZE={batch}"
    for name, val in zip(_LLMC_GRAIN_MACROS, grain or []):
        defs += f" -D{name}={val}"
    return defs

LLMC = AppSpec(
    name="llm.c",
    directory="llm.c",
    variants=[""],
    make_target=lambda v: "train_gpt2",
    binary=lambda v: "./train_gpt2",
    run_args=_llmc_run,
    parse=_parse_llmc,
    work=lambda t: (4.0 * float(t), "tokens (B\u00b7T)"),  # batch default 4
    sizes=[64, 128, 256],
    iters=10,
    rebuild_per_size=True,
    llmc_defs=_llmc_defs,
)

# ---- mnmg: Datalog transitive closure; dataset data_<N>.bin, N = #edges --------
# The "size" selects the input MNMGDatalog-reference/data/data_<size>.bin
# (size = edge count) and the x-axis work is that edge count. capacity_mult sizes
# the result set (next_pow2(edges * mult); must be >= ~2x TC or the run aborts
# with an overflow message).
# `iters` is NOT a knob here: the fixpoint runs until it converges, so the number
# of rounds is determined by the dataset (data_7035 -> 64, data_23874 -> 58) and
# each round is timed individually. Likewise TC has no task-count knob, so grain /
# the synchronous 1-task/loop are irrelevant. (The untimed warm-up rounds before
# round 0 are set with the TC_WARMUP env var; the default of 3 is used here.)
_MNMG_DATA = "MNMGDatalog-reference/data"
_MNMG_MULT = {7035: 64, 23874: 64}     # verified small graphs (TC 146120 / 481121)
_MNMG_MULT_DEFAULT = 4096              # generous default; raise via a larger set

def _mnmg_run(variant, size, iters, cfg, grain):
    mult = _MNMG_MULT.get(size, _MNMG_MULT_DEFAULT)
    return [f"{_MNMG_DATA}/data_{size}.bin", str(mult)]

MNMG = AppSpec(
    name="mnmg",
    directory="MNMGDatalog",
    variants=[""],
    make_target=lambda v: "tc",
    binary=lambda v: "./tc.x",
    run_args=_mnmg_run,
    parse=_parse_mnmg,
    work=lambda n: (float(n), "edges"),
    sizes=[7035, 23874],
    iters=0,                           # unused: round count comes from the data
)

APPS: Dict[str, AppSpec] = {a.name: a for a in (KRYLOV, LULESH, LLMC, MNMG)}
