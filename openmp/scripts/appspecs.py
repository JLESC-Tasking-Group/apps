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
pass within a taskgraph build is a *run-time* choice (OMP_TASKGRAPH_OPT for
XKOMP, NODES_TASKITER_CGIR_OPT for NODES). The backend is orthogonal and chosen
once (--target cpu | gpu | ompss); see BACKENDS below.

Unrolling
---------
A taskgraph instance carries an implicit taskgroup, so consecutive instances
cannot overlap. `unroll` (evaluate.py --unroll) folds that many iterations into
ONE recorded instance, recovering the cross-iteration overlap inside the graph
and paying the barrier once per `unroll` iterations. It is a run argument for
krylov / lulesh / mnmg. It only does anything under USE_TASKGRAPH, so evaluate.py
sweeps it for the taskgraph configurations only.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# CGIR optimization combos compared on top of the references (each -> one
# taskgraph:<opt> configuration). Names are CGIR pass names, comma or space
# separated: copy-fuse reduce-node transitive-reduction prog-fuse jit sequence
# batch. They are passed to XKOMP as OMP_TASKGRAPH_OPT and to NODES as
# NODES_TASKITER_CGIR_OPT (the same string drives both runtimes).
#
# The default is the INCREMENTAL pipeline of the paper: each combo adds one of
# the passes of the paper's pass table to the previous one, in the order the
# library applies them. That order is what makes the resulting bars readable as
# "what does this pass add on top of the ones before it".
#
#   +reduce     the two graph-reduction passes (control-node elimination and
#               transitive reduction)
#   +jit        recompiling each task body at -O3 in the runtime, WITHOUT fusing
#               -- isolated on purpose, so the fusion bar below cannot be
#               credited with a gain that is only recompilation
#   +prog-fuse  fusing serial chains of programs before compiling them
#   +packing    the sequence/batch packing passes (host super-tasks and vendor
#               command graphs)
#
# copy-fuse is deliberately absent: the pass exists in the library but is out of
# the paper's scope, and a configuration nobody reports is machine time wasted.
DEFAULT_OPTS = [
    "reduce-node,transitive-reduction",
    "reduce-node,transitive-reduction,jit",
    "reduce-node,transitive-reduction,jit,prog-fuse",
    # `sequence` is not here: measured on all three applications it batched
    # nothing (no same-device chain survives `batch`), so it only lengthened the
    # pipeline. `packing` is `batch` alone.
    "reduce-node,transitive-reduction,jit,prog-fuse,batch",
]

# Short legend label for each pipeline, keyed by the LAST pass added. Keeps the
# figures readable: the full pass list is still written to runs.csv, and the
# `opt` column remains the ground truth.
# Every pass name CGIR accepts, mirroring CGIR_FORALL_COMMAND_GRAPH_PASS in
# cgir/include/cgir/command-graph-pass.hpp. The harness validates --opts against
# this list and refuses to launch on an unknown name.
#
# It has to: both runtimes warn-and-ignore an unrecognised pass, so a typo does
# not fail, it silently runs a DIFFERENT pipeline than the one the results are
# labelled with. That is not hypothetical -- a recipe written with shell line
# continuations inside single quotes produced the token "\<newline>reduce-node",
# which neither runtime split on, so `reduce-node` never ran in three of the four
# pipelines of a whole overnight sweep and the CSV said otherwise.
CGIR_PASSES = frozenset([
    "copy-fuse", "reduce-node", "transitive-reduction",
    "prog-fuse", "jit", "sequence", "batch",
])

OPT_LABELS = {
    "reduce-node,transitive-reduction":                             "+reduce",
    "reduce-node,transitive-reduction,jit":                          "+jit",
    "reduce-node,transitive-reduction,jit,prog-fuse":                "+prog-fuse",
    "reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch": "+packing",
}


# --------------------------------------------------------------------------- #
# Backends. A backend fixes the tasking runtime and the device the work runs on;
# it is orthogonal to the configuration (which fixes the schedule and the CGIR
# pass set) and is chosen once per sweep with --target.
#
# `opt_env` is the environment variable through which that runtime takes its CGIR
# pass list. XKOMP and NODES accept the SAME pass names (both call
# cgir::command_graph_pass_set_from_str), which is what lets one --opts string
# drive an OpenMP and an OmpSs-2 sweep and makes the two comparable.
#
# `env` is applied to every run of the backend. NODES only reaches the CGIR path
# when taskiter.opt.use_cgir is on, and that option is a NODES config variable,
# not an environment variable, hence the NODES_CONFIG_OVERRIDE. (Its list
# separator is ',', so no comma-valued option may be set through it -- which is
# exactly why the pass list has an env var of its own.)
# --------------------------------------------------------------------------- #
# Backends. A backend fixes the tasking runtime and the device the work runs on;
# it is orthogonal to the configuration (which fixes the schedule and the CGIR
# pass set) and is chosen once per sweep with --target.
#
# `opt_env` is the environment variable through which that runtime takes its CGIR
# pass list. XKOMP and NODES accept the SAME pass names (both call
# cgir::command_graph_pass_set_from_str), which is what lets one --opts string
# drive an OpenMP and an OmpSs-2 sweep and makes the two comparable.
#
# `env` is applied to every run of the backend. NODES only reaches the CGIR path
# when taskiter.opt.use_cgir is on, and that option is a NODES config variable,
# not an environment variable, hence the NODES_CONFIG_OVERRIDE. (Its list
# separator is ',', so no comma-valued option may be set through it -- which is
# exactly why the pass list has an env var of its own.)
# --------------------------------------------------------------------------- #
@dataclass
class Backend:
    label: str                  # value of the `backend` column in runs.csv
    build: Dict[str, str]       # make variables
    opt_env: str                # env var carrying the CGIR pass list
    env: Dict[str, str] = field(default_factory=dict)


BACKENDS: Dict[str, Backend] = {
    "cpu":   Backend("cpu",   {"USE_TARGET": "0", "USE_OMPSS": "0"}, "OMP_TASKGRAPH_OPT"),
    "gpu":   Backend("gpu",   {"USE_TARGET": "1", "USE_OMPSS": "0"}, "OMP_TASKGRAPH_OPT"),
    "ompss": Backend("ompss", {"USE_TARGET": "0", "USE_OMPSS": "1"}, "NODES_TASKITER_CGIR_OPT",
                     {"NODES_CONFIG_OVERRIDE": "taskiter.opt.use_cgir=true"}),
}


@dataclass
class Config:
    label: str                 # legend label, e.g. "taskgraph:reduce-node,transitive-reduction"
    build: Dict[str, str]      # make variables, e.g. {"USE_SYNC": "1", ...}
    opt: Optional[str]         # CGIR pass list (None if no taskgraph)
    grain1: bool = False       # run with one task/kernel per loop (sync baseline)

    @property
    def taskgraph(self) -> bool:
        """Whether this configuration records a taskgraph -- i.e. whether it has
        the per-instance barrier that --unroll exists to amortize. Elsewhere the
        unroll is inert by construction (the apps group iterations identically in
        every configuration), so sweeping it there would only burn machine time."""
        return self.build.get("USE_TASKGRAPH") == "1"


# The configurations every sweep compares its pipelines against. None of them
# runs a CGIR pass, so a follow-up sweep that only varies how the passes are
# configured measures them again for nothing -- and lands a second copy of each
# in the results, which the analysis then has to choose between. `--omit` leaves
# them out; see evaluate.py.
REFERENCE_CONFIGS = ["synchronous", "no-taskgraph", "taskgraph:none"]


def default_configs(opts: List[str], omit: Optional[List[str]] = None) -> List[Config]:
    omit = set(omit or [])
    cfgs = [
        Config("synchronous",    {"USE_SYNC": "1", "USE_TASKGRAPH": "0"}, None, grain1=True),
        Config("no-taskgraph",   {"USE_SYNC": "0", "USE_TASKGRAPH": "0"}, None),
        Config("taskgraph:none", {"USE_SYNC": "0", "USE_TASKGRAPH": "1"}, ""),
    ]
    cfgs = [c for c in cfgs if c.label not in omit]
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


# All three C/C++ apps print the same four timing lines, in the same words, so
# one parser serves them. They are per taskgraph INSTANCE -- a group of `unroll`
# iterations -- and always in ms per iteration, and they map onto the runtime's
# own phases: instance 0 records the graph (XKOMP rc == 1), instance 1 builds and
# optimizes the command graph and runs the first replay (rc == 2), instances 2..
# are steady replays.
#
# `instance` vs `instances` disambiguates the singular lines from the plural ones
# without a lookahead: "instances 2..31" can never match "instance 1", because
# the 's' sits where the space would be. [^:\n] rather than [^:] so a line that
# is absent cannot reach forward to a colon several lines below.
def _parse_instances(text):
    return {
        "iter0_ms":  _grab(text, r"instance 0[^:\n]*:\s*" + _F + r"\s*ms"),
        "iter1_ms":  _grab(text, r"instance 1[^:\n]*:\s*" + _F + r"\s*ms"),
        "avg_ms":    _grab(text, r"instances \d+\.\.\d+[^:\n]*\(avg\)[^:\n]*:\s*" + _F + r"\s*ms"),
        "stddev_ms": _grab(text, r"instances \d+\.\.\d+[^:\n]*\(stddev\)[^:\n]*:\s*" + _F + r"\s*ms"),
    }


# The "answer" of a run: a number the app computes from its final state, which
# every configuration of the same problem must reproduce. It is what makes the
# optimization passes falsifiable -- a speedup from a run that computed something
# else is not a speedup. `verdict` is the app's own pass/fail line, when it has
# one. plot.py's report_answers() enforces both.
def _parse_krylov(text):
    m = _parse_instances(text)
    m.update({
        "elapsed_s": _grab(text, r"total solve time\s*:\s*" + _F),
        "flops":     _grab(text, r"theoretical flops\s*:\s*" + _F),
        "gflops":    _grab(text, r"performance\s*:\s*" + _F),
        "residual":  _grab(text, r"relative residual\s*:\s*" + _F),
        "error":     _grab(text, r"relative error\s*:\s*" + _F),
    })
    # Compared ACROSS configurations, not against a fixed threshold: a solver may
    # legitimately fail to converge on a given matrix (and then every
    # configuration says so, which is a property of the problem), but no
    # optimization pass may change the number it converged to.
    m["answer"] = m["residual"]
    return m


def _parse_lulesh(text):
    m = _parse_instances(text)
    m.update({
        # The app prints ":Elapsed time (s)  :  12.34" and ":FOM (z/s)  : 1.2e6",
        # i.e. colon-separated -- not "FOM = x". Matching on '=' silently left
        # both columns empty for every LULESH run ever recorded.
        "elapsed_s": _grab(text, r"Elapsed time[^:\n]*:\s*" + _F),
        "fom":       _grab(text, r"FOM[^:\n]*:\s*" + _F),
        # LULESH's own verdict. NOTE it is only a symmetry check over plane 0 of
        # the energy array (TotalRelDiff < 1e-9), so it catches an asymmetric
        # corruption but not a uniform one; `answer` below covers the rest.
        "verdict":   _grab(text, r"Verification[^:\n]*:\s*(\w+)", cast=str),
        "answer":    _grab(text, r"TotalAbsDiff\s*:\s*" + _F),
    })
    return m


def _parse_mnmg(text):
    # Same four instance lines as krylov / lulesh; here they are ms per fixpoint
    # ROUND. The extra work is the fallback: the round count is data-dependent
    # (the fixpoint runs to convergence), so a small graph -- or a large -u --
    # can leave fewer than three instances and no steady-state window at all.
    # "total time (end-to-end)" is the MNMGDatalog paper's metric (file IO + H2D +
    # setup + compute + D2H) and is printed in ms, unlike the other apps' seconds.
    m = _parse_instances(text)
    if m["avg_ms"] is None:
        m["avg_ms"] = m["iter1_ms"] if m["iter1_ms"] is not None else m["iter0_ms"]
        m["stddev_ms"] = 0.0 if m["avg_ms"] is not None else None
    total_ms = _grab(text, r"total time \(end-to-end\)\s*:\s*" + _F + r"\s*ms")
    m["elapsed_s"] = (total_ms / 1000.0) if total_ms is not None else None
    # The size of the transitive closure: an exact integer every configuration
    # must agree on.
    m["answer"] = _grab(text, r"TC\s*=\s*" + _F + r"\s*tuples")
    return m


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
    run_args: Callable                   # (variant, size, iters, cfg, grain, unroll) -> [args]
    parse: Callable[[str], dict]         # stdout+stderr -> metrics dict
    work: Callable[[int], tuple]         # size -> (value, label) for the top axis
    sizes: List[int]
    iters: int
    # Whether the problem size is a compile-time macro, so each size needs its
    # own binary. No app currently sets it; kept because it is a property of an
    # app, not of the harness, and the next one may need it.
    rebuild_per_size: bool = False
    build_defs: Optional[Callable] = None  # (size, iters, grain, unroll) -> extra make vars
    # Number of granularity knobs this app takes, i.e. how many ':'-separated
    # components one --grain entry may hold (0 = the app has no knob). Used by
    # evaluate.py to reject malformed / stale grain specs.
    grain_arity: int = 0
    # Backends the app has a port for, so an --target sweep skips an app that
    # has none rather than building a binary whose task constructs vanish.
    backends: List[str] = field(default_factory=lambda: ["cpu", "gpu"])
    # Unroll cap per backend, for an app that cannot fold iterations there.
    max_unroll: Dict[str, int] = field(default_factory=dict)
    # Same, per variant, for a solver whose iterations cannot share a graph
    # instance (a restarted method, whose restart ends with a host solve the next
    # one consumes). Pinning it here keeps the harness from writing a row
    # labelled u=8 that in fact ran at u=1.
    variant_max_unroll: Dict[str, int] = field(default_factory=dict)
    # Inner steps that one `-i` unit buys, per variant. `-i` does not mean the
    # same thing for every solver: for a restarted method it counts RESTART
    # CYCLES of several inner steps each, so the value giving CG 200 iterations
    # would give it many times more. The harness divides by this, so one --iters
    # stays comparable work across the variants.
    variant_iters_div: Dict[str, int] = field(default_factory=dict)
    # Presentation metadata for the paper table (plot.py --latex-table): the name
    # and application class to print. Kept next to the app rather than in the
    # plotting script, so one file describes each app completely.
    pretty: str = ""
    klass: str = ""
    # What the panel of the paper figure puts on its x axis: "size" (the problem
    # size sweep) or "variant" (the app's variants at a single size). Krylov earns
    # "variant": five solvers at one size say more about generality than one
    # solver at three sizes, and the paper has room for exactly one panel each.
    panel_x: str = "size"
    # Tolerance on the `answer` across configurations of one problem, as an
    # absolute floor and a relative part -- two runs agree when
    #     |a - b| <= rtol * max(|a|,|b|)      (they are relatively close)
    #  or (|a| <= atol and |b| <= atol)       (both are effectively zero)
    #
    # Both parts are needed. `rtol` alone catches the corruption this exists for:
    # a device barrier dropped from a fused kernel moved CG's residual from 4e-15
    # to 2.6e-03, eleven orders of magnitude clear of any tolerance. But it
    # rejects everything when the answer is a quantity whose CORRECT value is
    # zero: a converged solver's residual lands wherever round-off puts it, and
    # 4.02e-15 against 4.06e-15 is a 1% relative difference between two results
    # that are both exactly right. `atol` is the level below which the quantity
    # has stopped meaning anything, so two values under it agree by construction.
    answer_rtol: float = 1e-6
    answer_atol: float = 0.0


# ---- krylov: grid n, matrix N=n^3, work ~ n^3; -t/-s = task counts (0=threads) --
# grain is THIS size's entry, written "s:t" (alphabetical, so the first component
# is -s = SpMV sub-tasks per block, the second is -t = tasks per vector op). A
# single-component entry sets -s and leaves -t at 1. None -> the app default
# (-t 0 -s 0 = auto, i.e. omp threads). Sync is always 1/loop (-t 1 -s 1).
#
# -u is the iterations folded into one taskgraph instance. GMRES ignores it (it
# is a restarted solver: each restart ends with a host least-squares solve whose
# result the next restart consumes, so two restarts cannot share an instance).
# Matrix options, per solver. The five solvers do NOT accept the same ones: the
# driver gates each on the solver's `opt_mask` and rejects the rest outright
# (krylov/common/driver.cpp:160-172), because a Krylov method constrains its
# operator. The symmetric methods take a stencil; the methods that tolerate a
# non-symmetric operator take a convection strength instead.
#
#   solver     opt_mask (in <solver>/<solver>.cpp)   accepts
#   cg         OPT_STENCIL                           -S
#   cr         OPT_STENCIL                           -S
#   minres     OPT_STENCIL | OPT_SHIFT               -S -g
#   bicgstab   OPT_CONV                              -c
#   gmres      OPT_CONV | OPT_MEM                    -c -m
#
# The values below are each solver's own default, passed explicitly so the matrix
# the run solved is recorded in the `cmd` column rather than implied by whatever
# the source defaulted to on the day.
_KRYLOV_MATRIX = {
    "cg":       ["-S", "27"],   # 27-point stencil, SPD
    "cr":       ["-S", "27"],
    "minres":   ["-S", "27"],
    "bicgstab": ["-c", "1.0"],  # convection 1.0 => non-symmetric
    "gmres":    ["-c", "1.0"],
}


def _krylov_run(variant, size, iters, cfg, grain, unroll):
    if cfg.grain1:
        t, s = "1", "1"
    elif grain:
        s = str(grain[0])
        t = str(grain[1]) if len(grain) > 1 else "1"
    else:
        t, s = "0", "0"
    return (["-n", str(size), "-i", str(iters), "-t", t, "-s", s]
            + _KRYLOV_MATRIX.get(variant, [])
            + ["-u", str(unroll)])

KRYLOV = AppSpec(
    name="krylov",
    directory="krylov",
    variants=["cg", "cr", "bicgstab", "minres"],
    make_target=lambda v: v,
    binary=lambda v: f"./{v}.x",
    run_args=_krylov_run,
    parse=_parse_krylov,
    work=lambda n: (float(n) ** 3, "n\u00b3 (\u221d FLOPs)"),
    sizes=[32, 48, 64],
    iters=50,
    grain_arity=2,          # "s:t"
    backends=["cpu", "gpu", "ompss"],
    # The answer is the relative residual. Below 1e-9 the solve has converged and
    # the remaining digits are round-off, so two runs there agree whatever they
    # print; above it (a solver that did not converge on this matrix) the relative
    # test applies and every configuration must still land on the same value.
    answer_atol=1e-9,
    pretty="Krylov",
    klass="Iterative solvers",
    panel_x="variant",
)

# ---- lulesh: mesh side s, zones = s^3; -nb = tasks per loop ---------------------
# grain is THIS size's entry, a single component "nb" -> -nb (tasks per loop).
# None -> the app default (-nb 32). The synchronous config is always 1 (-nb 1).
def _lulesh_run(variant, size, iters, cfg, grain, unroll):
    nb = "1" if cfg.grain1 else (str(grain[0]) if grain else "32")
    return ["-i", str(iters), "-s", str(size), "-r", "11", "-b", "1", "-c", "1",
            "-nb", nb, "-u", str(unroll)]

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
    grain_arity=1,          # "nb"
    pretty="LULESH",
    klass="PDE time stepping",
    backends=["cpu", "gpu", "ompss"],
    # TotalAbsDiff is a symmetry residual of the final energy field: zero for a
    # correct run, and a few 1e-8 in practice. LULESH's own `Verification` line
    # (TotalRelDiff < 1e-9) is the primary check here -- see _parse_lulesh -- and
    # this floor keeps the cross-configuration test from firing on round-off.
    answer_atol=1e-6,
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

# capacity_mult per dataset. The open-addressing result set holds
# next_pow2(edges * mult) slots and MUST hold >= ~2x TC; below that it saturates,
# every insert probes result_cap times before giving up, and the run aborts with
# an overflow message. TC is a property of the graph, not of the edge count, so
# this cannot be derived from `size` -- the values come from the measured closures
# (see MNMGDatalog/README.md, "Capacity"):
#
#   dataset              edges           TC   mult   result set
#   OL.cedge             7,035      146,120     64        4 MiB
#   TG.cedge            23,874      481,121     64       16 MiB
#   p2p-Gnutella31     147,892  884,179,859   8192       16 GiB
#   usroad             165,435  871,365,688   8192       16 GiB
#   fe_ocean           409,593 1,669,750,513  8192       32 GiB
#   vsp_finan          552,020  910,070,918   2048       16 GiB
#   com-dblp         1,049,866 1,911,754,892  2048       32 GiB
#
# NOTE the old default of 4096 was NOT safe for p2p-Gnutella31 (82% load factor on
# a linear-probing table); an unknown graph gets 8192 and should be added here
# once its TC is known.
_MNMG_MULT = {
    7035:      64,
    23874:     64,
    147892:  8192,
    165435:  8192,
    409593:  8192,
    552020:  2048,
    1049866: 2048,
}
_MNMG_MULT_DEFAULT = 8192

# -u also gates the convergence test, which is then only evaluated every `unroll`
# rounds: the fixpoint may run up to unroll-1 extra (empty, harmless) rounds.
def _mnmg_run(variant, size, iters, cfg, grain, unroll):
    mult = _MNMG_MULT.get(size, _MNMG_MULT_DEFAULT)
    return [f"{_MNMG_DATA}/data_{size}.bin", str(mult), "-u", str(unroll)]

MNMG = AppSpec(
    name="mnmg",
    directory="MNMGDatalog",
    variants=[""],
    make_target=lambda v: "tc",
    binary=lambda v: "./tc.x",
    run_args=_mnmg_run,
    parse=_parse_mnmg,
    work=lambda n: (float(n), "edges"),
    # 147892 needs a 16 GiB result set (see _MNMG_MULT); the two small graphs are
    # sub-100 MiB. The remaining datasets (165435, 409593, 552020, 1049866) are
    # sized in _MNMG_MULT and can be added once the 16-32 GiB runs are budgeted.
    sizes=[7035, 23874, 147892],
    iters=0,                           # unused: round count comes from the data
    pretty="MNMG",
    klass="Graph analytics",
    backends=["cpu", "gpu", "ompss"],
    # The transitive closure has an exact size; there is nothing to round off.
    answer_rtol=0.0,
    answer_atol=0.0,
)

APPS: Dict[str, AppSpec] = {a.name: a for a in (KRYLOV, LULESH, MNMG)}
