#!/usr/bin/env bash
# ============================================================================
# Interactive installer: llvm · hwloc · cgir · xkrt · xkblas · xkomp
#
# Dependency order:
#   llvm ─▶ cgir ─┐
#                    ├─▶  xkrt  ─┬─▶  xkblas
#   hwloc ───────────┘           └─▶  xkomp
#
# The custom LLVM (anlsys/llvm-project) is a dependency of cgir and may
# optionally be used to build cgir/xkrt/xkblas/xkomp.  It can be bootstrapped
# with any compiler, but building those libraries requires an LLVM >= 20 (the
# custom one, or a system clang >= 20).
#
# The custom libomptarget (the LLVM "offload" runtime) forwards OpenMP target
# calls to XKRT and XKOMP, creating a build loop (libomptarget → xkrt + xkomp →
# cgir → clang).  It is therefore built in two stages: LLVM is first built
# WITHOUT offload, then (after xkrt + xkomp are installed) the offload runtime is
# added in place.  xkomp's tests link libomptarget too, so only the xkomp library
# is installed before offload; its tests are compiled afterwards:
#   llvm (no offload) ─▶ cgir ─▶ xkrt ─▶ xkomp (lib) ─▶ llvm offload/libomptarget ─▶ xkomp tests
#
# Requires: cmake >= 3.17, a C/C++ compiler, git, autoconf/automake (for hwloc)
# ============================================================================

# Make sure we are actually running under bash.  This script uses bash-only
# features (BASH_SOURCE, arrays, and globbing of unquoted variable expansions);
# if it gets started by another shell (zsh in particular behaves differently),
# re-exec it with bash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ─── UI helpers ───────────────────────────────────────────────────────────────
# All user-visible output goes to /dev/tty so command-substitution $() captures
# only the actual return values of functions.
_tty() { printf "$@" >/dev/tty; }

hr()      { _tty '%s\n' "────────────────────────────────────────────────────────────────────────"; }
step()    { _tty "\n${BOLD}${CYAN}▶${NC} ${BOLD}%s${NC}\n" "$*"; }
info()    { _tty "  ${BLUE}·${NC} %s\n" "$*"; }
success() { _tty "  ${GREEN}✓${NC} %s\n" "$*"; }
warn()    { _tty "  ${YELLOW}!${NC} %s\n" "$*"; }
fatal()   { _tty "  ${RED}✗ FATAL:${NC} %s\n" "$*"; exit 1; }

# ─── Arguments ────────────────────────────────────────────────────────────────
# --force / --rebuild : rebuild & reinstall every component even if it is already
#                       installed (ignore the per-component completion markers).
FORCE_REBUILD=false
VARIANT=""          # --variant NAME : tag paths so a build can coexist as a variant
CACHE_ARG=""        # optional positional path to an .xkrt_install cache file
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force|--rebuild) FORCE_REBUILD=true ;;
        --variant)   shift; [[ $# -gt 0 ]] || fatal "--variant requires a value (e.g. --variant fast)"; VARIANT="$1" ;;
        --variant=*) VARIANT="${1#*=}" ;;
        -h|--help)
            _tty "Usage: %s [--force] [--variant NAME] [path/to/.xkrt_install.cache]\n" "$(basename "$0")"
            _tty "  --force         rebuild & reinstall every component, ignoring the\n"
            _tty "                  per-component completion markers (.xkrt-installed).\n"
            _tty "  --variant NAME  tag install/build/module paths and env.sh with NAME so\n"
            _tty "                  the same branch + build type can be installed side-by-side\n"
            _tty "                  as an independent variant (also --variant=NAME).\n"
            _tty "  [cache]         path to an .xkrt_install cache to load defaults from and\n"
            _tty "                  save back to (default: ./.xkrt_install[.NAME].cache).\n"
            exit 0 ;;
        -*) fatal "unknown argument: $1  (try --help)" ;;
        *)  CACHE_ARG="$1" ;;
    esac
    shift
done

# _sanitize_ref REF
# Filesystem/module-safe form of a git ref (branch/tag): '/' → '-', so it stays a
# single path component (e.g. "release/latest" → "release-latest").  Install,
# build and module paths are keyed on the branch (not the raw commit), so updating
# a branch reuses the same locations instead of piling up one dir per commit.
_sanitize_ref() { printf '%s' "${1//\//-}"; }

# _already_installed MARKER COMMIT
# True when --force was not given AND MARKER exists AND records COMMIT — i.e. this
# exact commit is already installed under the (branch-keyed) prefix, so its build
# can be skipped.  If the branch has since moved (COMMIT differs), it returns false
# so the branch dir is rebuilt in place.
_already_installed() {
    [[ "$FORCE_REBUILD" == "true" ]] && return 1
    [[ -f "$1" && "$(cat "$1" 2>/dev/null)" == "$2" ]]
}

# _mark_installed MARKER COMMIT — record COMMIT in MARKER (creating its dir),
# stamping a successful install of that commit.
_mark_installed() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; }

# prompt_yn QUESTION [DEFAULT=yes]
# Returns 0 for yes, 1 for no.
prompt_yn() {
    local q="$1" dflt="${2:-yes}" tag
    [[ "$dflt" == "yes" ]] && tag="${BOLD}[Y/n]${NC}" || tag="${BOLD}[y/N]${NC}"
    while true; do
        _tty "  ${BOLD}${BLUE}?${NC} %b %b: " "$q" "$tag"
        local ans; read -r ans </dev/tty
        ans="${ans:-$dflt}"
        case "${ans,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *) _tty "    Please answer yes or no.\n" ;;
        esac
    done
}

# prompt_value QUESTION DEFAULT
# Prints the user's answer (or the default) on stdout.
prompt_value() {
    _tty "  ${BOLD}${BLUE}?${NC} %s [${DIM}%s${NC}]: " "$1" "$2"
    local ans; read -r ans </dev/tty
    printf '%s' "${ans:-$2}"
}

# run_spinner MESSAGE [--] COMMAND [ARGS...]
# Runs COMMAND in the background while animating a spinner + elapsed seconds on
# /dev/tty, so long, quiet operations (e.g. a huge 'git checkout' updating the
# working tree) don't look frozen.  COMMAND's output is captured and shown only
# if it fails (then the script aborts via fatal).  On success the line is
# replaced by a green check.  Use ONLY for non-interactive commands — output is
# captured, so anything that might prompt (clone/fetch/pull on a private repo)
# must NOT be wrapped or it would hang invisibly.
run_spinner() {
    local msg="$1"; shift
    [[ "${1:-}" == "--" ]] && shift

    local logf; logf="$(mktemp "${TMPDIR:-/tmp}/xkrt_spin.XXXXXX")"
    # Reset the ERR trap in the subshell so a failure is reported by us (via the
    # captured log below) rather than printing a FATAL from the background.
    ( trap - ERR; "$@" ) >"$logf" 2>&1 &
    local pid=$! rc=0
    local frames='|/-\' i=0 start=$SECONDS

    while kill -0 "$pid" 2>/dev/null; do
        _tty "\r  ${BLUE}%s${NC} %s ${DIM}(%ds)${NC}\033[K" \
             "${frames:i++%4:1}" "$msg" "$((SECONDS - start))"
        sleep 0.2
    done
    wait "$pid" || rc=$?

    if (( rc == 0 )); then
        _tty "\r  ${GREEN}✓${NC} %s ${DIM}(%ds)${NC}\033[K\n" "$msg" "$((SECONDS - start))"
        rm -f "$logf"
    else
        _tty "\r  ${RED}✗${NC} %s ${DIM}(failed after %ds)${NC}\033[K\n" "$msg" "$((SECONDS - start))"
        _tty "  ${DIM}----- command output -----${NC}\n"
        cat "$logf" >/dev/tty 2>/dev/null || true
        rm -f "$logf"
        fatal "command failed: $*"
    fi
}

# ─── CMake option parsing ─────────────────────────────────────────────────────

# parse_cmake_opts FILE
# Prints one "VARNAME|description|DEFAULT" line per xkoption/cgiroption/ocgoption
# entry.  cgir uses 'cgiroption(NAME "desc" DEFAULT [DEPVAR ...])', so trailing
# dependency-variable names after the default are tolerated (and ignored here).
parse_cmake_opts() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    grep -E '^\s*(xkoption|cgiroption|ocgoption)\s*\(' "$f" \
      | sed -E 's/^\s*(xkoption|cgiroption|ocgoption)\s*\(\s*([A-Za-z0-9_]+)\s+"([^"]+)"\s+(ON|OFF)[^)]*\)/\2|\3|\4/' \
      | grep -E '^[A-Za-z0-9_]+\|' \
      || true
}

# ask_cmake_opts LIBNAME CMAKE_FILE [CACHED_OPTS]
# Asks whether to use cmake defaults or customise, then prompts accordingly.
# If CACHED_OPTS is non-empty, first offers to reuse it verbatim.
# Prints the accumulated "-DVAR=VAL ..." flag string on stdout.
ask_cmake_opts() {
    local lib="$1" f="$2" cached="${3:-}" flags=""

    # Fast path: reuse the cmake options recorded in a previous run.
    if [[ -n "$cached" ]]; then
        _tty "\n  ${BOLD}Cached cmake options for %s${NC}:\n    ${DIM}%s${NC}\n" "$lib" "$cached"
        if prompt_yn "  Reuse these cached cmake options for $lib?" "yes"; then
            printf '%s' "$cached"; return
        fi
    fi

    if [[ ! -f "$f" ]]; then
        _tty "  ${YELLOW}!${NC} CMakeLists.txt not found at %s – skipping option prompts.\n" "$f"
        printf ''; return
    fi

    local opts
    opts=$(parse_cmake_opts "$f")

    # No xkoption/cgiroption/ocgoption entries: just offer the free-form flags field.
    if [[ -z "$opts" ]]; then
        _tty "  ${DIM}(no xkoption/cgiroption/ocgoption entries found in CMakeLists.txt)${NC}\n"
        _tty "  ${BOLD}${BLUE}?${NC} Extra cmake flags for %s (or Enter to skip): " "$lib"
        local extra; read -r extra </dev/tty
        printf '%s' "${extra:-}"; return
    fi

    # There are options: ask whether to use all defaults or customise.
    _tty "\n  ${BOLD}CMake options for %s${NC} (parsed from CMakeLists.txt):\n" "$lib"
    _tty "  ${DIM}Note: these reflect the local clone; may differ on other branches.${NC}\n\n"

    if prompt_yn "  Use all default cmake options?" "yes"; then
        # Fast path: return empty flags; cmake will apply its own defaults.
        printf ''; return
    fi

    # Customise: ask each option individually, then offer extra flags.
    while IFS='|' read -r var desc default; do
        [[ -z "$var" ]] && continue
        local def_yn
        [[ "$default" == "ON" ]] && def_yn="yes" || def_yn="no"
        _tty "    ${DIM}%-45s${NC} %s\n" "$var" "$desc"
        if prompt_yn "      Enable ${BOLD}${var}${NC}?" "$def_yn"; then
            flags="${flags} -D${var}=ON"
        else
            flags="${flags} -D${var}=OFF"
        fi
    done <<< "$opts"

    _tty "\n  ${BOLD}${BLUE}?${NC} Extra cmake flags for %s (e.g. -DFOO=bar), or Enter to skip: " "$lib"
    local extra; read -r extra </dev/tty
    flags="${flags}${extra:+ $extra}"

    printf '%s' "${flags# }"   # strip leading space
}

# ─── Configuration cache ─────────────────────────────────────────────────────

# Every Phase-1 output variable, listed once.  Used to (a) snapshot cached values
# into _C_* as prompt defaults and (b) serialise the config in _write_cache — so
# adding a config variable only needs editing this one list.
CONFIG_VARS=(
    BASE_DIR REPO_DIR INSTALL_DIR MODULES_DIR VARIANT
    CC CXX
    INSTALL_LLVM LLVM_BRANCH LLVM_BUILD_TYPE LLVM_PROJECTS LLVM_RUNTIMES
    LLVM_CMAKE_TARGETS LLVM_CMAKE_RUNTIME_TARGETS LLVM_EXTRA_CMAKE_OPTS
    LLVM_GPU_SUMMARY USE_LLVM_FOR_BUILD LLVM_BOOTSTRAP_CC LLVM_BOOTSTRAP_CXX
    INSTALL_HWLOC HWLOC_BRANCH HWLOC_CONFIGURE_OPTS
    INSTALL_CGIR   CGIR_BRANCH   CGIR_BUILD_TYPE   CGIR_CMAKE_OPTS
    INSTALL_XKRT   XKRT_BRANCH   XKRT_BUILD_TYPE   XKRT_CMAKE_OPTS
    INSTALL_XKBLAS XKBLAS_BRANCH XKBLAS_BUILD_TYPE XKBLAS_CMAKE_OPTS
    INSTALL_XKOMP  XKOMP_BRANCH  XKOMP_BUILD_TYPE  XKOMP_CMAKE_OPTS
)

# _write_cache FILE
# Serialises every Phase-1 output variable into a sourceable bash file so the
# user can skip re-answering all questions on a re-run.
_write_cache() {
    local f="$1"
    {
        printf '# xkrt install configuration — %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        declare -p "${CONFIG_VARS[@]}"
    } > "$f"
}

# ─── Module-file generation ──────────────────────────────────────────────────

# generate_modulefile LIBNAME PREFIX ENVVAR OUTFILE [DEP ...]
# ENVVAR : environment variable set to the install prefix (e.g. XKRT_HOME)
# DEP    : zero or more dependency module paths ("name/hash/type") that will be
#          auto-loaded via 'module load' when this module is loaded.
generate_modulefile() {
    local lib="$1" prefix="$2" envvar="$3" out="$4"
    shift 4
    local -a deps=("$@")

    mkdir -p "$(dirname "$out")"

    {
        cat <<MODEOF
#%Module1.0

set whatis    "$lib"
set software  "$lib"
set description "$lib"

conflict "\$software"

MODEOF

        # Emit dependency loads before path manipulation so their libs are
        # visible as soon as this module is loaded.
        if (( ${#deps[@]} > 0 )); then
            echo "# ── Dependencies ────────────────────────────────────────────────────────────"
            echo "module use \"$MODULES_DIR\""
            for dep in "${deps[@]}"; do
                echo "module load \"$dep\""
            done
            echo ""
        fi

        cat <<MODEOF
set prefix "$prefix"

prepend-path PATH               "\$prefix/bin"
prepend-path MANPATH            "\$prefix/share/man"
prepend-path INFOPATH           "\$prefix/share/info"
prepend-path LIBRARY_PATH       "\$prefix/lib"
prepend-path LIBRARY_PATH       "\$prefix/lib64"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib64"
prepend-path CMAKE_PREFIX_PATH  "\$prefix"
prepend-path CMAKE_LIBRARY_PATH "\$prefix/lib"
prepend-path CMAKE_INCLUDE_PATH "\$prefix/include"
prepend-path C_INCLUDE_PATH     "\$prefix/include"
prepend-path CPATH              "\$prefix/include"
prepend-path PKG_CONFIG_PATH    "\$prefix/lib/pkgconfig"

setenv $envvar "\$prefix"
MODEOF
    } > "$out"
}

# ─── Git helpers ─────────────────────────────────────────────────────────────

# clone_or_update URL DEST REF
# REF can be a branch name or a tag.
clone_or_update() {
    local url="$1" dest="$2" ref="$3"

    if [[ ! -d "$dest/.git" ]]; then
        info "Cloning $(basename "$dest") from $url"
        # --progress forces the transfer meter even when stderr is not a TTY,
        # so large clones (e.g. LLVM) don't look frozen.
        git clone --progress "$url" "$dest"
    else
        info "Fetching latest for $(basename "$dest")"
        git -C "$dest" fetch --all --tags --progress
    fi

    # 'git checkout' can spend many seconds updating the working tree/index of a
    # huge repo (LLVM) with no output of its own, so show a spinner.  It is a
    # purely local operation, so capturing its output (in run_spinner) is safe.
    run_spinner "Checking out $ref" \
        git -C "$dest" checkout "$ref"

    # Pull only if this ref is a remote branch (tags are immutable)
    if git -C "$dest" ls-remote --exit-code --heads origin "$ref" >/dev/null 2>&1; then
        git -C "$dest" pull --progress
    fi
}

# ─── Build environment helpers ────────────────────────────────────────────────

# _init_module_system
# Tries to make the 'module' shell function available by sourcing common Lmod /
# Environment-Modules init files.  Sets _MODULE_AVAILABLE=true on success.
# Non-fatal: if no module system is found we fall back to manual env exports.
_MODULE_AVAILABLE=false
_init_module_system() {
    if declare -f module > /dev/null 2>&1; then
        _MODULE_AVAILABLE=true; return 0
    fi
    local f
    for f in \
        "${LMOD_PKG:-NOFILE}/init/bash" \
        "/usr/share/lmod/lmod/init/bash" \
        "/usr/local/lmod/lmod/init/bash" \
        "/opt/apps/lmod/lmod/init/bash" \
        "/usr/share/modules/init/bash" \
        "/etc/profile.d/modules.sh"
    do
        [[ -f "$f" ]] || continue
        # shellcheck disable=SC1090
        source "$f" 2>/dev/null && { _MODULE_AVAILABLE=true; return 0; }
    done
    warn "Module system not found; environment will be configured manually."
}

# _activate_prefix PREFIX [MOD_NAME MOD_SUBPATH]
# Exports every variable that a 'module load' would set for PREFIX:
#   PATH, CPATH, C_INCLUDE_PATH, CPLUS_INCLUDE_PATH,
#   LIBRARY_PATH, LD_LIBRARY_PATH,
#   CMAKE_PREFIX_PATH, CMAKE_LIBRARY_PATH, CMAKE_INCLUDE_PATH,
#   PKG_CONFIG_PATH
# If the module system is available it also runs 'module load MOD_NAME/MOD_SUBPATH'
# (belt-and-suspenders: both the env vars and the module are applied).
_activate_prefix() {
    local prefix="$1" mod_name="${2:-}" mod_sub="${3:-}"

    # Export all paths that compilers, linkers, cmake and pkg-config consult.
    export PATH="${prefix}/bin${PATH:+:$PATH}"
    export C_INCLUDE_PATH="${prefix}/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    export CPLUS_INCLUDE_PATH="${prefix}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export CPATH="${prefix}/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${prefix}/lib:${prefix}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib:${prefix}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CMAKE_PREFIX_PATH="${prefix}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export CMAKE_LIBRARY_PATH="${prefix}/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
    export CMAKE_INCLUDE_PATH="${prefix}/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
    export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

    # Also load the module if the system is available (idempotent, non-fatal).
    if [[ "$_MODULE_AVAILABLE" == "true" && -n "$mod_name" && -n "$mod_sub" ]]; then
        module use "$MODULES_DIR" 2>/dev/null || true
        module load "${mod_name}/${mod_sub}" 2>/dev/null || true
    fi
}

# _llvm_host_target
# Maps uname -m to the LLVM backend name needed in LLVM_TARGETS_TO_BUILD.
_llvm_host_target() {
    case "$(uname -m)" in
        x86_64)  printf 'X86'      ;;
        aarch64) printf 'AArch64'  ;;
        arm*)    printf 'ARM'      ;;
        ppc64le) printf 'PowerPC'  ;;
        ppc64)   printf 'PowerPC'  ;;
        riscv64) printf 'RISCV'    ;;
        s390x)   printf 'SystemZ'  ;;
        *)       printf 'X86'      ;;   # safe fallback
    esac
}

# _list_contains LIST TOKEN
# Returns 0 if the ';'-separated LIST contains TOKEN, 1 otherwise.
_list_contains() {
    local IFS=';' x
    for x in $1; do [[ "$x" == "$2" ]] && return 0; done
    return 1
}

# _strip_token LIST TOKEN
# Prints the ';'-separated LIST with every occurrence of TOKEN (and empty
# fields) removed.
_strip_token() {
    local IFS=';' x result=""
    for x in $1; do
        [[ -z "$x" || "$x" == "$2" ]] && continue
        result="${result:+$result;}$x"
    done
    printf '%s' "$result"
}

# build_llvm RUNTIMES LABEL
# (Re)configure, build and install LLVM in $LLVM_BUILD_DIR with the given
# LLVM_ENABLE_RUNTIMES value (RUNTIMES may be empty to enable no runtimes).
# cmake is always re-run, so this can be invoked twice for the staged build:
# first without the offload runtime, then again to add the custom libomptarget
# once XKRT exists.  The top-level build always uses the bootstrap compiler so a
# second invocation is not seen as a compiler change (which forces a full rebuild).
# Globals: LLVM_BUILD_DIR LLVM_INSTALL_DIR LLVM_REPO_DIR LLVM_BUILD_TYPE
#   LLVM_PROJECTS LLVM_CMAKE_RUNTIME_TARGETS LLVM_CMAKE_TARGETS
#   LLVM_EXTRA_CMAKE_OPTS LLVM_BOOTSTRAP_CC LLVM_BOOTSTRAP_CXX
build_llvm() {
    local runtimes="$1" label="$2"
    # Assemble the LLVM_ENABLE_RUNTIMES flags.  The host list NEVER includes
    # 'openmp': the host OpenMP runtime (libomp/archer/libompd/docs) is unused —
    # xkomp provides it.  But 'offload' (the host libomptarget) needs the device
    # runtime libompdevice.a, which is built by 'openmp' on GPU targets — so when
    # offload is requested, enable openmp per-GPU-target only (it does not matter
    # whether the caller also listed 'openmp'; it is always stripped from the host
    # and added to the GPU targets).
    local -a rt_flags=()
    if [[ -n "$runtimes" ]]; then
        local _host_rt _gpu_rt _t; local -a _rtts=()
        _host_rt="$(_strip_token "$runtimes" openmp)"       # host: never openmp
        [[ -n "$_host_rt" ]] && rt_flags+=("-DLLVM_ENABLE_RUNTIMES=${_host_rt}")
        if _list_contains "$runtimes" offload; then
            _gpu_rt="${_host_rt:+${_host_rt};}openmp"        # e.g. "offload;openmp"
            IFS=';' read -r -a _rtts <<< "$LLVM_CMAKE_RUNTIME_TARGETS"
            for _t in ${_rtts[@]+"${_rtts[@]}"}; do
                [[ "$_t" =~ ^(nvptx|amdgcn|spirv) ]] && \
                    rt_flags+=("-DRUNTIMES_${_t}_LLVM_ENABLE_RUNTIMES=${_gpu_rt}")
            done
        fi
    fi

    # Pin the host / default target triple.  LLVM otherwise auto-detects it via
    # config.guess, which on some nodes (seen when bootstrapping with clang) yields
    # an empty/"unknown" triple — producing a clang whose default target is
    # 'unknown' that cannot compile anything (breaks hwloc's configure, etc.).  Ask
    # the bootstrap compiler for the host triple (both gcc and clang support
    # -dumpmachine), falling back to LLVM's config.guess, then a uname-derived
    # triple.  Passed for BOTH host and default so a stale build-dir cache (these
    # are non-FORCE cache vars in LLVM) can never keep an old empty value.
    local _host_triple
    _host_triple="$("$LLVM_BOOTSTRAP_CC" -dumpmachine 2>/dev/null || true)"
    [[ -z "$_host_triple" ]] && _host_triple="$(sh "$LLVM_REPO_DIR/llvm/cmake/config.guess" 2>/dev/null || true)"
    [[ -z "$_host_triple" ]] && _host_triple="$(uname -m)-unknown-linux-gnu"
    info "Host / default target triple: $_host_triple"

    info "Configuring LLVM (${label}) …"
    cd "$LLVM_BUILD_DIR"

    # shellcheck disable=SC2086
    # The top-level LLVM build always uses the bootstrap compiler.  LLVM_DIR /
    # MLIR_DIR are scrubbed from the environment here (they are exported AFTER
    # stage 1 to pin the XK* libraries to this LLVM) so building LLVM itself — and
    # its runtimes sub-build, e.g. the deferred offload/libomptarget — never tries
    # to resolve an external LLVM install.  XKRT_DIR / XKOMP_DIR / CMAKE_PREFIX_PATH
    # are kept, so find_package(XKRT)/find_package(XKOMP) in libomptarget still work.
    env -u LLVM_DIR -u MLIR_DIR \
        CC="$LLVM_BOOTSTRAP_CC" CXX="$LLVM_BOOTSTRAP_CXX" \
        cmake \
        -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
        -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
        ${rt_flags[@]+"${rt_flags[@]}"} \
        -DLLVM_RUNTIME_TARGETS="$LLVM_CMAKE_RUNTIME_TARGETS" \
        -DLLVM_TARGETS_TO_BUILD="$LLVM_CMAKE_TARGETS" \
        -DLLVM_HOST_TRIPLE="$_host_triple" \
        -DLLVM_DEFAULT_TARGET_TRIPLE="$_host_triple" \
        -DCMAKE_CXX_FLAGS="-Wno-c2y-extensions" \
        $LLVM_EXTRA_CMAKE_OPTS \
        "$LLVM_REPO_DIR/llvm"

    info "Building & installing LLVM (${label}) — this may take a while …"
    env -u LLVM_DIR -u MLIR_DIR make install -j "$(nproc)"
}

# ─── Component build helpers (cgir · xkrt · xkblas · xkomp) ────────────────────
# These factor out the boilerplate shared by the cmake-based components.  The
# component-specific bits (configure flags, build command, deps, activation) stay
# explicit in each component's block, between _component_prep and _component_finish.

# _component_prep PREFIX NAME URL
# Clone/update NAME, derive its (branch+variant-keyed) paths and export them as
# globals for downstream stages, then set _NEED_BUILD.  Call this as a PLAIN
# statement (NOT inside an if-condition) so 'set -e' stays active in its body and
# a git/rm/mkdir/cd failure still aborts — matching the original inline blocks.
#   reads:   <PREFIX>_BRANCH <PREFIX>_BUILD_TYPE   (+ global VARIANT_SFX)
#   exports: <PREFIX>_HASH <PREFIX>_REF <PREFIX>_INSTALL_DIR <PREFIX>_BUILD_DIR <PREFIX>_MOD
#   sets:    _NEED_BUILD=yes → marker removed, build dir wiped+created, cwd = build dir
#            _NEED_BUILD=no  → already installed at this commit (--force overrides)
_component_prep() {
    local P="$1" name="$2" url="$3"
    local _bn="${P}_BRANCH" _tn="${P}_BUILD_TYPE"
    local branch="${!_bn}" btype="${!_tn}" hash ref idir bdir mod
    step "Building & installing $name"
    clone_or_update "$url" "$REPO_DIR/$name" "$branch"
    hash="$(git -C "$REPO_DIR/$name" rev-parse HEAD | cut -c1-12)" \
        || fatal "$name: cannot read git HEAD in $REPO_DIR/$name"
    ref="$(_sanitize_ref "$branch")$VARIANT_SFX"          # branch(+variant) → path/module key
    idir="$INSTALL_DIR/$name/$ref/$btype"
    bdir="$REPO_DIR/$name/build/$ref/$btype"
    mod="$MODULES_DIR/$name/$ref/$btype"
    declare -g "${P}_HASH=$hash" "${P}_REF=$ref" "${P}_INSTALL_DIR=$idir" \
               "${P}_BUILD_DIR=$bdir" "${P}_MOD=$mod"
    if _already_installed "$idir/.xkrt-installed" "$hash"; then
        info "$name already installed ($ref/$btype @ $hash) — skipping build (--force to rebuild)."
        _NEED_BUILD=no
        return 0
    fi
    rm -f "$idir/.xkrt-installed"
    rm -rf "$bdir" && mkdir -p "$bdir" || fatal "$name: cannot prepare build dir $bdir"
    cd "$bdir" || fatal "$name: cannot enter build dir $bdir"
    _NEED_BUILD=yes
}

# _component_finish PREFIX NAME ENVVAR ACTIVATE [DEP ...]
# Generate the module file, optionally activate the prefix (so later components
# find it), record the module load line, refresh env.sh and report.
#   reads: <PREFIX>_INSTALL_DIR <PREFIX>_REF <PREFIX>_BUILD_TYPE <PREFIX>_MOD
#   ACTIVATE: "yes" to put this prefix on CMAKE_PREFIX_PATH / LLVM_DIR etc.
_component_finish() {
    local P="$1" name="$2" envvar="$3" activate="$4"; shift 4
    local -a deps=("$@")
    local _in="${P}_INSTALL_DIR" _rn="${P}_REF" _tn="${P}_BUILD_TYPE" _mn="${P}_MOD"
    local idir="${!_in}" ref="${!_rn}" btype="${!_tn}" mod="${!_mn}"
    generate_modulefile "$name" "$idir" "$envvar" "$mod" ${deps[@]+"${deps[@]}"}
    [[ "$activate" == "yes" ]] && _activate_prefix "$idir" "$name" "$ref/$btype"
    MOD_LOAD+=("module load $name/$ref/$btype")
    write_env_sh
    success "$name installed → $idir"
    success "module file      → $mod"
}

# ─── Error trap ───────────────────────────────────────────────────────────────
trap 'fatal "error on line $LINENO – aborting."' ERR

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 – GATHER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# _dflt_yn CACHED_BOOL FALLBACK
# Maps a cached "true"/"false" to a "yes"/"no" prompt default; uses FALLBACK when
# there is no cached value.
_dflt_yn() {
    case "${1:-}" in
        true)  printf 'yes' ;;
        false) printf 'no'  ;;
        *)     printf '%s' "$2" ;;
    esac
}

# _dflt_member HAVE_CACHE LIST TOKEN FALLBACK
# "yes"/"no" prompt default for an option encoded as TOKEN's presence in a cached
# ';'-separated LIST; uses FALLBACK when there is no cache.
_dflt_member() {
    [[ "$1" == "true" ]] || { printf '%s' "$4"; return; }
    if _list_contains "$2" "$3"; then printf 'yes'; else printf 'no'; fi
}

# prompt_cmake_component PREFIX NAME URL DEFAULT_BRANCH DEFAULT_BUILD_TYPE
# Ask whether to install a cmake component and, if so, its branch/build type and
# cmake options (clone first so the option list is parsed from its CMakeLists).
# Cached answers (via _C_* snapshot) seed the defaults.  Sets the globals the
# build phase reads: INSTALL_<PREFIX>, <PREFIX>_BRANCH/_BUILD_TYPE/_CMAKE_OPTS.
prompt_cmake_component() {
    local P="$1" name="$2" url="$3" dbranch="$4" dbtype="$5"
    local _cin="_C_INSTALL_${P}" _cbr="_C_${P}_BRANCH" _cbt="_C_${P}_BUILD_TYPE" _cop="_C_${P}_CMAKE_OPTS"
    declare -g "INSTALL_${P}=false" "${P}_BRANCH=" "${P}_BUILD_TYPE=" "${P}_CMAKE_OPTS="
    if prompt_yn "Install $name?" "$(_dflt_yn "${!_cin:-}" "yes")"; then
        local branch btype opts
        branch="$(prompt_value "Branch" "${!_cbr:-$dbranch}")"
        btype="$(prompt_value "Build type (Release/Debug)" "${!_cbt:-$dbtype}")"
        clone_or_update "$url" "$REPO_DIR/$name" "$branch"
        opts="$(ask_cmake_opts "$name" "$REPO_DIR/$name/CMakeLists.txt" "${!_cop:-}")"
        declare -g "INSTALL_${P}=true" "${P}_BRANCH=$branch" "${P}_BUILD_TYPE=$btype" "${P}_CMAKE_OPTS=$opts"
    fi
}

# Cache file: an explicit path (positional arg) wins; otherwise default to a
# per-variant name in the CWD so different variants keep independent configs.
_CLI_VARIANT="$(_sanitize_ref "$VARIANT")"   # --variant wins over any cached VARIANT
_C_VARIANT=""                                 # variant recorded in the loaded cache
if [[ -n "$CACHE_ARG" ]]; then
    CACHE_FILE="$CACHE_ARG"
else
    CACHE_FILE="$(pwd)/.xkrt_install${_CLI_VARIANT:+.$_CLI_VARIANT}.cache"
fi
REUSE_CACHE=false
HAVE_CACHE=false

if [[ -f "$CACHE_FILE" ]]; then
    # Load the cache up front so its values can seed the default answer of every
    # question below, whether or not the user skips straight to the build.
    # shellcheck disable=SC1090
    source "$CACHE_FILE"
    HAVE_CACHE=true

    # Snapshot the cached values into _C_* so the prompts below can offer them as
    # defaults (each component block resets its working vars before prompting).
    for _v in "${CONFIG_VARS[@]}"; do declare -g "_C_$_v=${!_v:-}"; done

    _tty "\n"
    hr
    _tty "  ${BOLD}${YELLOW}Cached configuration found${NC}\n"
    _tty "  %s\n" "$CACHE_FILE"
    _tty "  Created: %s\n" "$(date -r "$CACHE_FILE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                               || stat -c '%y' "$CACHE_FILE" 2>/dev/null | cut -d. -f1)"
    hr
    _tty "\n"
    _tty "  ${DIM}Tip: even if you answer 'no' below, these cached values are used as the\n"
    _tty "  default answer to every question — just press Enter to accept each one.${NC}\n\n"
    if prompt_yn "Reuse this configuration and skip straight to the build?" "yes"; then
        REUSE_CACHE=true
        success "Configuration loaded — jumping to build phase."
    else
        info "Re-running configuration (cached values pre-filled as the defaults)."
    fi
fi

# Resolve the effective variant: a CLI --variant always wins; otherwise adopt the
# value recorded in the loaded cache (so re-running with just the cache path keeps
# the variant).  VARIANT_SFX is folded into every component's ref key below.
VARIANT="$(_sanitize_ref "${_CLI_VARIANT:-${VARIANT:-}}")"
# Keep the variant safe for paths, module names and the unquoted 'module load'
# lines written into env.sh ('/' was already turned into '-' above).
if [[ -n "$VARIANT" && ! "$VARIANT" =~ ^[A-Za-z0-9._-]+$ ]]; then
    fatal "invalid variant '$VARIANT' — use only letters, digits, '.', '_', '-' (and '/')."
fi
VARIANT_SFX="${VARIANT:+-$VARIANT}"
# Warn if a CLI --variant overrides a different variant baked into the loaded
# cache: the current install uses the CLI value, but on the reuse path the cache
# file is not re-tagged, so update.sh (which reads the variant from the cache)
# would target the cache's variant.  Give each variant its own cache to be safe.
if [[ -n "$_CLI_VARIANT" && "$HAVE_CACHE" == "true" && "$_CLI_VARIANT" != "$_C_VARIANT" ]]; then
    warn "cache records variant '${_C_VARIANT:-<none>}' but --variant '$VARIANT' was given — using '$VARIANT'."
    warn "tip: keep one cache per variant (e.g. ./.xkrt_install.$VARIANT.cache) so update.sh matches."
fi
[[ -n "$VARIANT" ]] && info "Variant '${VARIANT}' — install/build/module paths and env.sh are tagged with it."

if [[ "$REUSE_CACHE" == "false" ]]; then

_tty "\n"
hr
_tty "  ${BOLD}xkrt ecosystem – interactive installer${NC}\n"
_tty "  llvm  ·  hwloc  ·  cgir  ·  xkrt  ·  xkblas  ·  xkomp\n"
hr
_tty "\n"

# ── Installation base ─────────────────────────────────────────────────────────
step "Installation directory"
BASE_DIR=$(prompt_value "Base directory (repos, installs and modules go here)" "${_C_BASE_DIR:-$(pwd)}")
BASE_DIR="$(realpath -m "$BASE_DIR")"
REPO_DIR="$BASE_DIR/repo"
INSTALL_DIR="$BASE_DIR/install"
MODULES_DIR="$BASE_DIR/modules"
info "Repos    → $REPO_DIR"
info "Install  → $INSTALL_DIR"
info "Modules  → $MODULES_DIR"

# Create the repo directory now so Phase 1 can clone into it for CMakeLists.txt
# parsing before the full build phase begins.
mkdir -p "$REPO_DIR"

# ── Compilers ─────────────────────────────────────────────────────────────────
step "Compilers"

# Minimum required clang major version.
readonly MIN_CLANG_VER=20

# _detect_clang
# Scans PATH for clang-<N>/clang++-<N> pairs and the unversioned clang/clang++.
# Prints one line: "STATUS CC CXX VERSION"
#   STATUS "ok"   – found a pair with version >= MIN_CLANG_VER (CC/CXX/VERSION set)
#   STATUS "old"  – found clang but all versions < MIN_CLANG_VER (VERSION=max found)
#   STATUS "none" – no clang found at all
_detect_clang() {
    local best_ver=0 best_cc="" best_cxx="" max_any=0
    local d base ver cc cxx

    # 1. Search versioned clang-<N> / clang++-<N> in every PATH directory.
    local -a path_dirs
    IFS=: read -ra path_dirs <<< "${PATH:-}"
    for d in "${path_dirs[@]}"; do
        [[ -d "$d" ]] || continue
        for cc in "$d"/clang-[0-9]*; do
            [[ -x "$cc" ]] || continue
            base="${cc##*/}"
            # Accept only exact "clang-<digits>" (exclude clang-cpp-19, clangd-19, …)
            [[ "$base" =~ ^clang-([0-9]+)$ ]] || continue
            ver="${BASH_REMATCH[1]}"
            cxx="${d}/clang++-${ver}"
            [[ -x "$cxx" ]] || continue          # need the matching C++ compiler
            (( ver > max_any )) && max_any=$ver
            if (( ver >= MIN_CLANG_VER && ver > best_ver )); then
                best_ver=$ver; best_cc="$cc"; best_cxx="$cxx"
            fi
        done
    done

    # 2. Also check unversioned 'clang' / 'clang++' (may be a symlink to latest).
    if command -v clang &>/dev/null && command -v clang++ &>/dev/null; then
        local vs; vs=$(clang --version 2>/dev/null | head -1)
        if [[ "$vs" =~ ([0-9]+)\.[0-9]+ ]]; then
            ver="${BASH_REMATCH[1]}"
            (( ver > max_any )) && max_any=$ver
            if (( ver >= MIN_CLANG_VER && ver > best_ver )); then
                best_ver=$ver
                best_cc=$(command -v clang)
                best_cxx=$(command -v clang++)
            fi
        fi
    fi

    if [[ -n "$best_cc" ]]; then
        printf 'ok %s %s %d' "$best_cc" "$best_cxx" "$best_ver"
    elif (( max_any > 0 )); then
        printf 'old - - %d' "$max_any"
    else
        printf 'none - - 0'
    fi
}

info "Scanning PATH for clang >= ${MIN_CLANG_VER} …"
read -r _clang_status _def_cc _def_cxx _clang_ver <<< "$(_detect_clang)"

# An LLVM/clang >= MIN_CLANG_VER is needed to build the libraries, but it does
# not have to be on PATH: we only record the detection result here and let the
# user pick/point at a compiler at the "Build compiler" step below.
case "$_clang_status" in
    ok)   success "Found ${_def_cc} / ${_def_cxx}  (version ${_clang_ver})" ;;
    old)  warn "Found clang ${_clang_ver}, but >= ${MIN_CLANG_VER} is needed to build the libraries." ;;
    none) warn "No clang >= ${MIN_CLANG_VER} found in PATH." ;;
esac
[[ "$_clang_status" == "ok" ]] || \
    info "You can still proceed: install the custom LLVM and build with it, or give the full path to a compiler below."

# Pick a sensible default *bootstrap* compiler (any compiler is fine for that):
# prefer a detected clang >= MIN_CLANG_VER, else any clang, else gcc, else cc/c++.
if [[ "$_clang_status" == "ok" ]]; then
    _boot_cc_default="$_def_cc"; _boot_cxx_default="$_def_cxx"
else
    _boot_cc_default="$(command -v clang  || command -v gcc || command -v cc  || echo cc)"
    _boot_cxx_default="$(command -v clang++ || command -v g++ || command -v c++ || echo c++)"
fi

# ── LLVM (custom patched) ─────────────────────────────────────────────────────
step "LLVM  [cmake; custom patched toolchain — anlsys/llvm-project; dependency of cgir]"
_tty "\n"
_tty "  The anlsys fork adds new OpenMP pragma support (access clauses, etc.) and\n"
_tty "  provides the LLVM that cgir depends on.\n"
_tty "  It can be bootstrapped with any compiler.  You may optionally use it to\n"
_tty "  build cgir/xkrt/xkblas/xkomp — those require an LLVM >= ${MIN_CLANG_VER}\n"
_tty "  (the custom one, or a system clang >= ${MIN_CLANG_VER} detected above).\n\n"

INSTALL_LLVM=false
LLVM_BRANCH="" LLVM_BUILD_TYPE=""
LLVM_PROJECTS="" LLVM_RUNTIMES=""
LLVM_CMAKE_TARGETS="" LLVM_CMAKE_RUNTIME_TARGETS=""
LLVM_EXTRA_CMAKE_OPTS=""
LLVM_GPU_SUMMARY=""   # human-readable, for summary display
USE_LLVM_FOR_BUILD=false                      # use custom LLVM to build the libs?
LLVM_BOOTSTRAP_CC="" LLVM_BOOTSTRAP_CXX=""    # compiler used to build LLVM itself

if prompt_yn "Install custom patched LLVM?" "$(_dflt_yn "${_C_INSTALL_LLVM:-}" "no")"; then
    INSTALL_LLVM=true
    LLVM_BRANCH=$(prompt_value "Branch" "${_C_LLVM_BRANCH:-main}")
    LLVM_BUILD_TYPE=$(prompt_value "Build type (Release/Debug/RelWithDebInfo)" "${_C_LLVM_BUILD_TYPE:-Release}")

    # ── Host LLVM backend (auto-detected from uname -m) ──────────────────────
    _host_tgt=$(_llvm_host_target)
    info "Host LLVM backend: ${_host_tgt}  (uname -m = $(uname -m))"

    # ── GPU offload targets (multi-select) ────────────────────────────────────
    _tty "\n"
    _tty "  ${BOLD}GPU offload targets${NC} — select the GPU architectures to support:\n"
    _tty "\n"
    _tty "    ${BOLD}1)${NC} NVIDIA / CUDA\n"
    _tty "       LLVM backend : NVPTX\n"
    _tty "       Runtime target: nvptx64-nvidia-cuda\n"
    _tty "\n"
    _tty "    ${BOLD}2)${NC} AMD / ROCm\n"
    _tty "       LLVM backend : AMDGPU\n"
    _tty "       Runtime target: amdgcn-amd-amdhsa\n"
    _tty "\n"
    # Reconstruct the previously-selected GPU numbers from the cached backends.
    _gpu_default=""
    _list_contains "${_C_LLVM_CMAKE_TARGETS:-}" NVPTX  && _gpu_default="${_gpu_default}${_gpu_default:+ }1"
    _list_contains "${_C_LLVM_CMAKE_TARGETS:-}" AMDGPU && _gpu_default="${_gpu_default}${_gpu_default:+ }2"
    _tty "  Enter numbers separated by spaces (e.g. \"1 2\"), or Enter for [%s]: " "${_gpu_default:-host-only}"
    read -r _gpu_sel </dev/tty
    _gpu_sel="${_gpu_sel:-$_gpu_default}"

    _backends="$_host_tgt"
    _runtime_tgts="default"
    _gpu_names=()

    for _g in $_gpu_sel; do
        case "$_g" in
            1)
                _backends="${_backends};NVPTX"
                _runtime_tgts="${_runtime_tgts};nvptx64-nvidia-cuda"
                _gpu_names+=("NVIDIA/CUDA")
                ;;
            2)
                _backends="${_backends};AMDGPU"
                _runtime_tgts="${_runtime_tgts};amdgcn-amd-amdhsa"
                _gpu_names+=("AMD/ROCm")
                ;;
            *)
                warn "Unknown selection '$_g' – ignored."
                ;;
        esac
    done

    LLVM_CMAKE_TARGETS="$_backends"
    LLVM_CMAKE_RUNTIME_TARGETS="$_runtime_tgts"

    if (( ${#_gpu_names[@]} == 0 )); then
        LLVM_GPU_SUMMARY="host-only"
    else
        LLVM_GPU_SUMMARY=$(IFS='+'; printf '%s' "${_gpu_names[*]}")
    fi

    info "LLVM_TARGETS_TO_BUILD  = $LLVM_CMAKE_TARGETS"
    info "LLVM_RUNTIME_TARGETS   = $LLVM_CMAKE_RUNTIME_TARGETS"

    # ── Projects ─────────────────────────────────────────────────────────────
    _tty "\n  ${BOLD}LLVM projects${NC} (clang is always included):\n"
    _projects="clang"
    if prompt_yn "  Include MLIR  (recommended for CGIR)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" mlir "yes")"; then
        _projects="${_projects};mlir"
    fi
    if prompt_yn "  Include lld  (LLVM linker — recommended for GPU offload)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" lld "yes")"; then
        _projects="${_projects};lld"
    fi
    if prompt_yn "  Include bolt (binary optimizer)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" bolt "no")"; then
        _projects="${_projects};bolt"
    fi
    LLVM_PROJECTS="$_projects"

    # ── Runtimes ─────────────────────────────────────────────────────────────
    # A single yes/no selecting "offload" or "" (no runtimes).  build_llvm turns
    # "offload" into: host libomptarget, plus the 'openmp' device runtime
    # (libompdevice.a) on GPU targets only (offload needs it).  The host OpenMP
    # runtime (libomp/archer/libompd/docs) is never built — xkomp provides it.
    # ("openmp;offload" from an older cache is also accepted; openmp is always
    # stripped from the host and placed on the GPU targets.)
    _tty "\n  ${BOLD}LLVM runtimes${NC}:\n"
    # Default from cache: yes if runtimes were enabled before (openmp or offload).
    _rt_default="yes"
    if [[ "$HAVE_CACHE" == "true" ]]; then
        if _list_contains "${_C_LLVM_RUNTIMES:-}" openmp || _list_contains "${_C_LLVM_RUNTIMES:-}" offload
        then _rt_default="yes"; else _rt_default="no"; fi
    fi
    _tty "  ${DIM}Builds the host libomptarget + the GPU device runtime (libompdevice.a);\n"
    _tty "  the host OpenMP runtime is NOT built (xkomp provides it).  offload also\n"
    _tty "  depends on XKRT and XKOMP, so it is built last — after XKRT and XKOMP.${NC}\n"
    if prompt_yn "  Build OpenMP offload (libomptarget + device runtime)?" "$_rt_default"; then
        LLVM_RUNTIMES="offload"
    else
        LLVM_RUNTIMES=""
    fi

    # ── Extra flags ───────────────────────────────────────────────────────────
    LLVM_EXTRA_CMAKE_OPTS=$(prompt_value "Extra cmake flags for LLVM (or Enter to keep/skip)" "${_C_LLVM_EXTRA_CMAKE_OPTS:-}")

    # ── Bootstrap compiler (any compiler may build LLVM) ──────────────────────
    _tty "\n  ${BOLD}Bootstrap compiler${NC} — used only to compile LLVM itself.\n"
    _tty "  ${DIM}Any C/C++ compiler works here; no minimum version.${NC}\n"
    LLVM_BOOTSTRAP_CC=$(prompt_value  "Bootstrap C compiler"   "${_C_LLVM_BOOTSTRAP_CC:-$_boot_cc_default}")
    LLVM_BOOTSTRAP_CXX=$(prompt_value "Bootstrap C++ compiler" "${_C_LLVM_BOOTSTRAP_CXX:-$_boot_cxx_default}")

    # ── Use the custom LLVM to build the libraries? ───────────────────────────
    _tty "\n"
    if prompt_yn "Use this custom LLVM to build cgir/xkrt/xkblas/xkomp?" "$(_dflt_yn "${_C_USE_LLVM_FOR_BUILD:-}" "yes")"; then
        USE_LLVM_FOR_BUILD=true
    else
        USE_LLVM_FOR_BUILD=false
    fi

fi

# ── Build compiler ────────────────────────────────────────────────────────────
# cgir/xkrt/xkblas/xkomp must be built with an LLVM >= MIN_CLANG_VER: either
# the custom LLVM (built above) or a system clang >= MIN_CLANG_VER.
step "Build compiler  [for cgir · xkrt · xkblas · xkomp]"
if [[ "$INSTALL_LLVM" == "true" && "$USE_LLVM_FOR_BUILD" == "true" ]]; then
    # The custom LLVM provides clang >= 20.  Its exact path depends on the git
    # hash, so CC/CXX are resolved in Phase 2 right after LLVM is installed.
    CC="" CXX=""
    success "Libraries will be built with the custom LLVM (clang/clang++)."
else
    # Not using the custom LLVM → the libraries need an LLVM/clang >= MIN_CLANG_VER.
    # If none was auto-detected we do NOT fail; instead we ask for the full paths
    # (e.g. an LLVM installed in a non-standard location not on PATH).
    _cc_default="${_C_CC:-$_def_cc}"
    _cxx_default="${_C_CXX:-$_def_cxx}"
    if [[ "$_clang_status" != "ok" ]]; then
        if [[ "$_clang_status" == "old" ]]; then
            warn "Found clang ${_clang_ver} in PATH, older than the expected >= ${MIN_CLANG_VER}."
        else
            warn "No clang was found in PATH."
        fi
        info "Enter the full path to the C/C++ compiler to build the libraries with"
        info "(an LLVM/clang >= ${MIN_CLANG_VER} is expected; this is not verified)."
        # The '-' detection placeholder is not a usable default; fall back to a
        # cached compiler, else any compiler we could find.
        _cc_default="${_C_CC:-$_boot_cc_default}"
        _cxx_default="${_C_CXX:-$_boot_cxx_default}"
    fi
    _tty "\n"
    CC=$(prompt_value  "C compiler (full path; LLVM >= ${MIN_CLANG_VER})"   "$_cc_default")
    CXX=$(prompt_value "C++ compiler (full path; LLVM >= ${MIN_CLANG_VER})" "$_cxx_default")
    command -v "$CC"  >/dev/null 2>&1 || warn "'$CC' was not found — double-check the path."
    command -v "$CXX" >/dev/null 2>&1 || warn "'$CXX' was not found — double-check the path."
fi
export CC CXX

# ── hwloc ─────────────────────────────────────────────────────────────────────
step "hwloc  [autotools; no cmake dependencies]"
_tty "\n"
_tty "  hwloc is required by xkrt for hardware topology discovery.\n"
_tty "  If you skip this step, xkrt will attempt to use a system-installed hwloc\n"
_tty "  (searched in standard paths by Findhwloc.cmake).\n\n"
_tty "  hwloc's optional GPU backends (cuda/nvml/rsmi/opencl/levelzero) make\n"
_tty "  libhwloc hard-link GPU libraries that are often not on the linker/loader\n"
_tty "  path, which breaks downstream links.  They are disabled by default; pass\n"
_tty "  e.g. '--enable-rsmi --with-rocm=/opt/rocm' below to re-enable any you need.\n\n"

INSTALL_HWLOC=false; HWLOC_BRANCH=""; HWLOC_CONFIGURE_OPTS=""
if prompt_yn "Install hwloc?" "$(_dflt_yn "${_C_INSTALL_HWLOC:-}" "yes")"; then
    INSTALL_HWLOC=true
    HWLOC_BRANCH=$(prompt_value "Branch / tag" "${_C_HWLOC_BRANCH:-v2.14}")
    HWLOC_CONFIGURE_OPTS=$(prompt_value "Extra hwloc configure flags (or Enter to keep/skip)" "${_C_HWLOC_CONFIGURE_OPTS:-}")
fi

# ── cgir ────────────────────────────────────────────────────────────────────
step "cgir  [cmake; no runtime dependencies; parallel to hwloc]"
prompt_cmake_component CGIR cgir "https://github.com/JLESC-Tasking-Group/opencg" release/latest Release

# ── xkrt ──────────────────────────────────────────────────────────────────────
step "xkrt  [cmake; depends on hwloc + cgir]"
prompt_cmake_component XKRT xkrt "https://gitlab.inria.fr/xkaapi/dev-v2" release/latest Release

# ── xkblas ────────────────────────────────────────────────────────────────────
step "xkblas  [cmake; depends on xkrt; parallel to xkomp]"
_tty "\n"
_tty "  Tip: if you enable USE_CBLAS, also set USE_OPENBLAS, USE_MKL, or\n"
_tty "  USE_CRAYBLAS via the 'extra cmake flags' prompt below.\n\n"
prompt_cmake_component XKBLAS xkblas "https://gitlab.inria.fr/xkblas/dev" release/v2.0-latest Release

# ── xkomp ─────────────────────────────────────────────────────────────────────
step "xkomp  [cmake; depends on xkrt; parallel to xkblas]"
prompt_cmake_component XKOMP xkomp "https://github.com/anlsys/xkomp" release/latest Release

# ── Summary & confirmation ────────────────────────────────────────────────────
_tty "\n"; hr
_tty "  ${BOLD}Summary${NC}\n"; hr
_tty "\n"
[[ -n "$VARIANT" ]] && _tty "  ${BOLD}variant${NC}: %s   ${DIM}(paths & env.sh tagged '-%s')${NC}\n\n" "$VARIANT" "$VARIANT"

_lib_row() {
    local name="$1" install="$2" branch="${3:-}" btype="${4:-}"
    if [[ "$install" == "true" ]]; then
        _tty "  ${GREEN}✓${NC}  %-10s  branch: %-22s  build: %s\n" \
             "$name" "$branch" "$btype"
    else
        _tty "  ${DIM}✗  %-10s  (skipped)${NC}\n" "$name"
    fi
}

if [[ "$INSTALL_LLVM" == "true" ]]; then
    _tty "  ${GREEN}✓${NC}  %-10s  branch: %-22s  build: %s\n" \
         "llvm" "$LLVM_BRANCH" "$LLVM_BUILD_TYPE"
    _tty "          projects: %-20s  runtimes: %s\n" \
         "$LLVM_PROJECTS" "${LLVM_RUNTIMES:-none}"
    _tty "          GPU targets: %s\n" "$LLVM_GPU_SUMMARY"
    _tty "          bootstrap: %-20s  build libs with llvm: %s\n" \
         "$LLVM_BOOTSTRAP_CC" "$USE_LLVM_FOR_BUILD"
else
    _tty "  ${DIM}✗  %-10s  (skipped)${NC}\n" "llvm"
fi
if [[ "$INSTALL_LLVM" == "true" && "$USE_LLVM_FOR_BUILD" == "true" ]]; then
    _tty "  ${DIM}library compiler: custom LLVM clang (resolved after LLVM build)${NC}\n"
else
    _tty "  ${DIM}library compiler: %s${NC}\n" "$CC"
fi
_lib_row "hwloc"  "$INSTALL_HWLOC"  "$HWLOC_BRANCH"  "autotools"
_lib_row "cgir" "$INSTALL_CGIR" "$CGIR_BRANCH" "$CGIR_BUILD_TYPE"
_lib_row "xkrt"   "$INSTALL_XKRT"   "$XKRT_BRANCH"   "$XKRT_BUILD_TYPE"
_lib_row "xkblas" "$INSTALL_XKBLAS" "$XKBLAS_BRANCH" "$XKBLAS_BUILD_TYPE"
_lib_row "xkomp"  "$INSTALL_XKOMP"  "$XKOMP_BRANCH"  "$XKOMP_BUILD_TYPE"
_tty "\n"

if ! prompt_yn "Proceed with installation?" "yes"; then
    info "Aborted by user."; exit 0
fi

_write_cache "$CACHE_FILE"
success "Configuration saved → $CACHE_FILE"

fi # REUSE_CACHE — end of Phase 1

# Edge-safe defaults — also covers reusing a cache written by an older version
# of this script that predates the LLVM-as-build-compiler options.
: "${INSTALL_LLVM:=false}"
: "${USE_LLVM_FOR_BUILD:=false}"
: "${LLVM_BOOTSTRAP_CC:=${CC:-cc}}"
: "${LLVM_BOOTSTRAP_CXX:=${CXX:-c++}}"
: "${HWLOC_CONFIGURE_OPTS:=}"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 – BUILD & INSTALL
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$REPO_DIR" "$INSTALL_DIR" "$MODULES_DIR"

declare -a MOD_LOAD=()   # module load lines for final usage message
# Variant installs get their own env file (env.<variant>.sh) so they never
# clobber the base env.sh (or another variant's).
ENV_FILE="$BASE_DIR/env${VARIANT:+.$VARIANT}.sh"

# write_env_sh
# (Re)write env.sh so it loads every module accumulated in MOD_LOAD so far.  It is
# called after each component installs (or is skipped), so env.sh is always up to
# date — a partial install still leaves a usable env.sh for what succeeded.
# Uses globals: ENV_FILE, MODULES_DIR, MOD_LOAD.
write_env_sh() {
    {
cat <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh on $(date '+%Y-%m-%d %H:%M:%S')
#
# Source this file to load the installed xkrt-ecosystem modules:
#   source "$ENV_FILE"

# Make 'module' available if the current shell did not export it.  (zsh, notably,
# does not export shell functions to child processes, so 'bash env.sh' or a script
# sourcing this file won't have inherited Lmod's 'module' function.)  The Lmod /
# Modules environment (e.g. \$LMOD_PKG) IS inherited, so re-source its init.
if ! command -v module >/dev/null 2>&1; then
    for _f in "\${LMOD_PKG:-/nonexistent}/init/bash" /usr/share/lmod/lmod/init/bash /usr/local/lmod/lmod/init/bash /opt/apps/lmod/lmod/init/bash /usr/share/modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "\$_f" ] || continue
        . "\$_f" || true
        command -v module >/dev/null 2>&1 && break
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "env.sh: no 'module' command and no module-system init script was found;" >&2
    echo "        load Lmod / Environment Modules, then re-source this file." >&2
    return 1 2>/dev/null || exit 1
fi

module use "$MODULES_DIR"
EOF
    for line in ${MOD_LOAD[@]+"${MOD_LOAD[@]}"}; do
        printf '%s\n' "$line"
    done
    } > "$ENV_FILE"
}

_init_module_system      # try to make 'module' available for _activate_prefix

# ── LLVM ──────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_LLVM" == "true" ]]; then
    step "Building & installing LLVM"

    # Use the local clone if it lives inside the scripts dir; otherwise clone.
    LLVM_REPO_DIR="$REPO_DIR/llvm-project"
    if [[ -d "$SCRIPT_DIR/llvm-project/.git" ]]; then
        info "Found local llvm-project clone at $SCRIPT_DIR/llvm-project"
        if [[ "$SCRIPT_DIR/llvm-project" != "$LLVM_REPO_DIR" ]]; then
            # Symlink so we don't duplicate the (large) checkout
            mkdir -p "$(dirname "$LLVM_REPO_DIR")"
            [[ -e "$LLVM_REPO_DIR" ]] || ln -s "$SCRIPT_DIR/llvm-project" "$LLVM_REPO_DIR"
        fi
    else
        clone_or_update \
            "https://github.com/anlsys/llvm-project" \
            "$LLVM_REPO_DIR" \
            "$LLVM_BRANCH"
    fi

    # Always fetch + checkout the requested branch in the repo we'll use.
    # LLVM is a very large repository, so keep the progress meters on (no -q)
    # to make it obvious the fetch/checkout are working and not hung.
    info "Updating LLVM sources for branch '$LLVM_BRANCH' (large repo — this can take a while) …"
    git -C "$LLVM_REPO_DIR" fetch --all --tags --progress
    # Checking out a branch in the LLVM tree rewrites a huge working tree and can
    # sit silently for many seconds — show a spinner so it does not look hung.
    run_spinner "Checking out '$LLVM_BRANCH' (updating the working tree)" \
        git -C "$LLVM_REPO_DIR" checkout "$LLVM_BRANCH"
    if git -C "$LLVM_REPO_DIR" ls-remote --exit-code --heads origin "$LLVM_BRANCH" >/dev/null 2>&1; then
        git -C "$LLVM_REPO_DIR" pull --progress
    fi

    LLVM_HASH=$(git -C "$LLVM_REPO_DIR" rev-parse HEAD | cut -c1-12)
    LLVM_REF="$(_sanitize_ref "$LLVM_BRANCH")$VARIANT_SFX"   # branch(+variant) → path/module key
    LLVM_INSTALL_DIR="$INSTALL_DIR/llvm/$LLVM_REF/$LLVM_BUILD_TYPE"
    LLVM_BUILD_DIR="$LLVM_REPO_DIR/build/$LLVM_REF/$LLVM_BUILD_TYPE"

    mkdir -p "$LLVM_BUILD_DIR"

    # The custom libomptarget (the "offload" runtime) forwards OpenMP target
    # calls to XKRT, so it depends on XKRT → CGIR → clang.  To break that loop
    # we build LLVM now WITHOUT offload, and (re)build the offload runtime later,
    # after CGIR and XKRT are installed (see "custom libomptarget" below).
    if _list_contains "$LLVM_RUNTIMES" offload; then
        LLVM_BUILD_OFFLOAD=true
    else
        LLVM_BUILD_OFFLOAD=false
    fi
    # Stage 1 builds clang/llvm only.  Defer BOTH offload (needs xkrt/xkomp) and
    # openmp (built GPU-only in stage 2 via build_llvm's per-target split, so the
    # host libomp/archer/libompd/docs are never built).
    LLVM_STAGE1_RUNTIMES="$(_strip_token "$(_strip_token "$LLVM_RUNTIMES" offload)" openmp)"

    if _already_installed "$LLVM_INSTALL_DIR/.xkrt-installed" "$LLVM_HASH"; then
        info "LLVM already installed ($LLVM_REF/$LLVM_BUILD_TYPE @ $LLVM_HASH) — skipping build (--force to rebuild)."
    else
        # Rebuilding stage 1 invalidates any previously-built offload runtime.
        rm -f "$LLVM_INSTALL_DIR/.xkrt-installed" "$LLVM_INSTALL_DIR/.xkrt-offload-installed"
        if [[ "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
            info "offload/libomptarget requested → deferred until after XKRT (dependency loop)."
            build_llvm "$LLVM_STAGE1_RUNTIMES" "1/2 — without libomptarget"
        else
            build_llvm "$LLVM_STAGE1_RUNTIMES" "single stage"
        fi
        _mark_installed "$LLVM_INSTALL_DIR/.xkrt-installed" "$LLVM_HASH"
    fi

    # Pin this EXACT LLVM/MLIR for every downstream build (cgir directly, and
    # the rest transitively), independently of whether it is also the compiler.
    # _activate_prefix puts the prefix on CMAKE_PREFIX_PATH; LLVM_DIR / MLIR_DIR
    # point find_package() straight at this build's cmake packages.  Both env
    # vars are read by find_package and inherited by every downstream cmake
    # (including transitive find_dependency(LLVM) pulled in via cgir).
    _activate_prefix "$LLVM_INSTALL_DIR" "llvm" "$LLVM_REF/$LLVM_BUILD_TYPE"
    if   [[ -d "$LLVM_INSTALL_DIR/lib/cmake/llvm"   ]]; then export LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
    elif [[ -d "$LLVM_INSTALL_DIR/lib64/cmake/llvm" ]]; then export LLVM_DIR="$LLVM_INSTALL_DIR/lib64/cmake/llvm"
    fi
    if   [[ -d "$LLVM_INSTALL_DIR/lib/cmake/mlir"   ]]; then export MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
    elif [[ -d "$LLVM_INSTALL_DIR/lib64/cmake/mlir" ]]; then export MLIR_DIR="$LLVM_INSTALL_DIR/lib64/cmake/mlir"
    fi
    info "Pinned LLVM_DIR=${LLVM_DIR:-<not found>}"
    if [[ -n "${MLIR_DIR:-}" ]]; then
        info "Pinned MLIR_DIR=$MLIR_DIR"
    else
        warn "No MLIR cmake package under $LLVM_INSTALL_DIR — enable the 'mlir' LLVM project if cgir needs MLIR."
    fi

    # If the user opted to build the libraries with the custom LLVM, also switch
    # the build compiler to its clang/clang++.
    if [[ "$USE_LLVM_FOR_BUILD" == "true" ]]; then
        CC="$LLVM_INSTALL_DIR/bin/clang"
        CXX="$LLVM_INSTALL_DIR/bin/clang++"
        export CC CXX
        success "Libraries will be built with: $CC"
    fi

    # ── Module file ───────────────────────────────────────────────────────────
    LLVM_MOD_DIR="$MODULES_DIR/llvm/$LLVM_REF"
    LLVM_MOD="$LLVM_MOD_DIR/$LLVM_BUILD_TYPE"
    mkdir -p "$LLVM_MOD_DIR"
    cat > "$LLVM_MOD" <<MODEOF
#%Module1.0

set whatis    "LLVM (anlsys patched)"
set software  "llvm"
set description "LLVM toolchain — anlsys/llvm-project @ $LLVM_HASH"

conflict "\$software"

set prefix "$LLVM_INSTALL_DIR"

prepend-path PATH               "\$prefix/bin"
prepend-path MANPATH            "\$prefix/share/man"
prepend-path LIBRARY_PATH       "\$prefix/lib"
prepend-path LIBRARY_PATH       "\$prefix/lib64"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib64"
prepend-path CMAKE_PREFIX_PATH  "\$prefix"
prepend-path CMAKE_LIBRARY_PATH "\$prefix/lib"
prepend-path CMAKE_INCLUDE_PATH "\$prefix/include"
prepend-path C_INCLUDE_PATH     "\$prefix/include"
prepend-path CPATH              "\$prefix/include"
prepend-path PKG_CONFIG_PATH    "\$prefix/lib/pkgconfig"

# Set CC/CXX so downstream builds automatically use this clang.
setenv CC  "\$prefix/bin/clang"
setenv CXX "\$prefix/bin/clang++"
setenv LLVM_HOME "\$prefix"
MODEOF

    MOD_LOAD+=("module load llvm/$LLVM_REF/$LLVM_BUILD_TYPE")
    write_env_sh
    if [[ "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
        success "LLVM (without libomptarget) installed → $LLVM_INSTALL_DIR"
        info    "libomptarget (offload) will be built after XKRT."
    else
        success "LLVM  installed → $LLVM_INSTALL_DIR"
    fi
    success "module file     → $LLVM_MOD"
fi

# ── hwloc ─────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_HWLOC" == "true" ]]; then
    step "Building & installing hwloc"
    clone_or_update \
        "https://github.com/open-mpi/hwloc" \
        "$REPO_DIR/hwloc" \
        "$HWLOC_BRANCH"

    HWLOC_HASH=$(git -C "$REPO_DIR/hwloc" rev-parse HEAD | cut -c1-12)
    HWLOC_REF="$(_sanitize_ref "$HWLOC_BRANCH")$VARIANT_SFX"   # branch/tag(+variant) → path/module key
    HWLOC_INSTALL_DIR="$INSTALL_DIR/hwloc/$HWLOC_REF"

    if _already_installed "$HWLOC_INSTALL_DIR/.xkrt-installed" "$HWLOC_HASH"; then
        info "hwloc already installed ($HWLOC_REF @ $HWLOC_HASH) — skipping build (--force to rebuild)."
    else
        rm -f "$HWLOC_INSTALL_DIR/.xkrt-installed"
        cd "$REPO_DIR/hwloc"
        if [[ ! -f configure ]]; then
            info "Running autogen.sh …"
            ./autogen.sh
        fi
        info "Configuring …"
        # Disable hwloc's optional GPU backends by default — they make libhwloc
        # hard-depend on GPU libraries (librocm_smi64, libcudart, …) that are often
        # not on the linker/loader search path and break downstream links (xkrt).
        # Any user HWLOC_CONFIGURE_OPTS come last so they can re-enable a specific
        # backend, e.g. "--enable-rsmi --with-rocm=/opt/rocm".
        # shellcheck disable=SC2086
        ./configure --prefix="$HWLOC_INSTALL_DIR" $HWLOC_CONFIGURE_OPTS
        info "Building …"
        make -j "$(nproc)"
        info "Installing …"
        make install
        _mark_installed "$HWLOC_INSTALL_DIR/.xkrt-installed" "$HWLOC_HASH"
    fi

    # Activate the installed hwloc so downstream builds (xkrt) see its headers,
    # libraries and cmake config.  This sets PATH, CPATH, LD_LIBRARY_PATH,
    # CMAKE_PREFIX_PATH, etc. — equivalent to 'module load hwloc/…'.
    HWLOC_MOD="$MODULES_DIR/hwloc/$HWLOC_REF/default"
    generate_modulefile "hwloc" "$HWLOC_INSTALL_DIR" "HWLOC_HOME" "$HWLOC_MOD"
    _activate_prefix "$HWLOC_INSTALL_DIR" "hwloc" "$HWLOC_REF/default"
    MOD_LOAD+=("module load hwloc/$HWLOC_REF/default")
    write_env_sh
    success "hwloc  installed → $HWLOC_INSTALL_DIR"
    success "module file      → $HWLOC_MOD"
fi

# ── cgir ────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_CGIR" == "true" ]]; then
    _component_prep CGIR cgir "https://github.com/JLESC-Tasking-Group/opencg"
    if [[ "$_NEED_BUILD" == "yes" ]]; then
        # If a custom LLVM was built, force cgir's find_package(LLVM)/find_package(MLIR)
        # onto that exact build so it can never silently fall back to a system LLVM/MLIR.
        _cgir_llvm_flags=""
        if [[ "$INSTALL_LLVM" == "true" ]]; then
            [[ -n "${LLVM_DIR:-}" ]] && _cgir_llvm_flags="-DLLVM_DIR=$LLVM_DIR"
            [[ -n "${MLIR_DIR:-}" ]] && _cgir_llvm_flags="${_cgir_llvm_flags:+$_cgir_llvm_flags }-DMLIR_DIR=$MLIR_DIR"
        fi

        # shellcheck disable=SC2086
        cmake $CGIR_CMAKE_OPTS $_cgir_llvm_flags \
            -DCMAKE_BUILD_TYPE="$CGIR_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$CGIR_INSTALL_DIR" \
            "$REPO_DIR/cgir"
        make install -j "$(nproc)"
        _mark_installed "$CGIR_INSTALL_DIR/.xkrt-installed" "$CGIR_HASH"
    fi

    # cgir builds against (and links) the custom LLVM/MLIR whenever one was built,
    # so its module must load that llvm module at runtime (a system LLVM, like a
    # system hwloc, ships no module).  Activate it so xkrt finds its cmake config.
    declare -a _cgir_deps=()
    [[ "$INSTALL_LLVM" == "true" ]] && _cgir_deps+=("llvm/$LLVM_REF/$LLVM_BUILD_TYPE")
    _component_finish CGIR cgir CGIR_HOME yes ${_cgir_deps[@]+"${_cgir_deps[@]}"}
fi

# ── xkrt ──────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKRT" == "true" ]]; then
    _component_prep XKRT xkrt "https://gitlab.inria.fr/xkaapi/dev-v2"
    if [[ "$_NEED_BUILD" == "yes" ]]; then
        # shellcheck disable=SC2086
        cmake $XKRT_CMAKE_OPTS \
            -DCMAKE_BUILD_TYPE="$XKRT_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$XKRT_INSTALL_DIR" \
            "$REPO_DIR/xkrt"
        make install -j "$(nproc)"
        _mark_installed "$XKRT_INSTALL_DIR/.xkrt-installed" "$XKRT_HASH"
    fi

    # Activate xkrt so xkblas and xkomp find its headers, libs and cmake config.
    # Deps: cgir is always required; hwloc only when installed from source here
    # (otherwise it is system-provided and ships no module).
    declare -a _xkrt_deps=()
    [[ "$INSTALL_CGIR"  == "true" ]] && _xkrt_deps+=("cgir/$CGIR_REF/$CGIR_BUILD_TYPE")
    [[ "$INSTALL_HWLOC" == "true" ]] && _xkrt_deps+=("hwloc/$HWLOC_REF/default")
    _component_finish XKRT xkrt XKRT_HOME yes ${_xkrt_deps[@]+"${_xkrt_deps[@]}"}
fi

# ── xkblas ────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKBLAS" == "true" ]]; then
    _component_prep XKBLAS xkblas "https://gitlab.inria.fr/xkblas/dev"
    if [[ "$_NEED_BUILD" == "yes" ]]; then
        # shellcheck disable=SC2086
        cmake $XKBLAS_CMAKE_OPTS \
            -DCMAKE_BUILD_TYPE="$XKBLAS_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$XKBLAS_INSTALL_DIR" \
            "$REPO_DIR/xkblas"
        make install -j "$(nproc)"
        _mark_installed "$XKBLAS_INSTALL_DIR/.xkrt-installed" "$XKBLAS_HASH"
    fi

    # xkblas is a leaf (nothing else builds against it): no activation.
    declare -a _xkblas_deps=()
    [[ "$INSTALL_XKRT" == "true" ]] && _xkblas_deps+=("xkrt/$XKRT_REF/$XKRT_BUILD_TYPE")
    _component_finish XKBLAS xkblas XKBLAS_HOME no ${_xkblas_deps[@]+"${_xkblas_deps[@]}"}
fi

# ── xkomp ─────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKOMP" == "true" ]]; then
    _component_prep XKOMP xkomp "https://github.com/anlsys/xkomp"
    if [[ "$_NEED_BUILD" == "yes" ]]; then
        # shellcheck disable=SC2086
        cmake $XKOMP_CMAKE_OPTS \
            -DCMAKE_BUILD_TYPE="$XKOMP_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$XKOMP_INSTALL_DIR" \
            "$REPO_DIR/xkomp"
        # Build & install ONLY the xkomp library — the 'xkomp' target plus its
        # 'compat_links' symlinks (libomp.so/libiomp5.so) — NOT tests/.  xkomp's
        # CMakeLists always does add_subdirectory(tests/), and those tests link the
        # custom libomptarget, which does not exist yet (libomptarget depends on the
        # xkomp library and is built in the offload stage below).  A plain
        # 'make install' would build 'all' first (tests included) and fail here.
        # 'cmake --install' installs the already-built artifacts without an 'all'
        # build; the tests are compiled later, once libomptarget exists (see the
        # "xkomp tests" step after the offload stage).
        make -j "$(nproc)" xkomp compat_links
        cmake --install .
        _mark_installed "$XKOMP_INSTALL_DIR/.xkrt-installed" "$XKOMP_HASH"
    fi

    # xkomp is a leaf (nothing else builds against it): no activation.
    declare -a _xkomp_deps=()
    [[ "$INSTALL_XKRT" == "true" ]] && _xkomp_deps+=("xkrt/$XKRT_REF/$XKRT_BUILD_TYPE")
    _component_finish XKOMP xkomp XKOMP_HOME no ${_xkomp_deps[@]+"${_xkomp_deps[@]}"}
fi

# ── LLVM offload runtime (custom libomptarget) ─────────────────────────────────
# Deferred from the LLVM step above: the custom libomptarget depends on XKRT and
# XKOMP (it forwards OpenMP target calls to them), so it can only be built now
# that both exist.  We reconfigure the existing LLVM build to add the offload
# runtime and install it into the same prefix (only the offload runtime is
# compiled — the rest of LLVM is already built).
if [[ "$INSTALL_LLVM" == "true" && "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
    step "Building & installing custom libomptarget (LLVM offload runtime)"

    if _already_installed "$LLVM_INSTALL_DIR/.xkrt-offload-installed" "$LLVM_HASH"; then
        success "libomptarget (offload) already installed → $LLVM_INSTALL_DIR (skipping; --force to rebuild)"
    else
        rm -f "$LLVM_INSTALL_DIR/.xkrt-offload-installed"

        # libomptarget does find_package(XKRT) and find_package(XKOMP), so both must
        # be discoverable BEFORE we reconfigure LLVM.  Explicitly (re)load their
        # modules and export the variables find_package consults: it looks at
        # <pkg>_DIR and CMAKE_PREFIX_PATH (NOT <pkg>_HOME).  _activate_prefix puts
        # each prefix on CMAKE_PREFIX_PATH, and the LLVM runtimes sub-build inherits
        # these environment variables, which is what resolves the lookups.
        if [[ "$INSTALL_XKRT" == "true" ]]; then
            info "Loading the XKRT module so libomptarget (offload) can find it …"
            _activate_prefix "$XKRT_INSTALL_DIR" "xkrt" "$XKRT_REF/$XKRT_BUILD_TYPE"
            export XKRT_HOME="$XKRT_INSTALL_DIR"
            export XKRT_DIR="$XKRT_INSTALL_DIR"          # find_package(XKRT) hint
            success "XKRT activated (CMAKE_PREFIX_PATH now includes $XKRT_INSTALL_DIR)"
        else
            warn "XKRT was not installed by this script, but the custom libomptarget"
            warn "depends on it — load your XKRT module (or put its prefix on"
            warn "CMAKE_PREFIX_PATH, or set XKRT_DIR) before this step."
        fi

        if [[ "$INSTALL_XKOMP" == "true" ]]; then
            info "Loading the XKOMP module so libomptarget (offload) can find it …"
            _activate_prefix "$XKOMP_INSTALL_DIR" "xkomp" "$XKOMP_REF/$XKOMP_BUILD_TYPE"
            export XKOMP_HOME="$XKOMP_INSTALL_DIR"
            export XKOMP_DIR="$XKOMP_INSTALL_DIR"        # find_package(XKOMP) hint
            success "XKOMP activated (CMAKE_PREFIX_PATH now includes $XKOMP_INSTALL_DIR)"
        else
            warn "XKOMP was not installed by this script, but the custom libomptarget"
            warn "depends on it — load your XKOMP module (or put its prefix on"
            warn "CMAKE_PREFIX_PATH, or set XKOMP_DIR) before this step."
        fi

        # Force the LLVM runtimes to fully reconfigure & rebuild.  Stage 1 configured
        # the runtimes WITHOUT offload; just re-running cmake with offload added does
        # NOT reconfigure the already-stamped runtimes sub-build (see the cached
        # runtimes-stamps/ under $LLVM_BUILD_DIR/runtimes), so the host libomptarget.so
        # gets silently skipped while the device *.bc still build.  Removing the
        # runtimes sub-build forces a clean openmp+offload build for every
        # LLVM_RUNTIME_TARGETS entry, while keeping the already-built clang/llvm/mlir
        # (far cheaper than wiping the whole LLVM build).
        rm -rf "$LLVM_BUILD_DIR/runtimes"
        build_llvm "$LLVM_RUNTIMES" "2/2 — with libomptarget"

        # Sanity-check that the host offload runtime (libomptarget.so) was produced;
        # without it, OpenMP target links fail with "unable to find library -lomptarget".
        _llvm_omptarget=""
        for _d in "$LLVM_INSTALL_DIR/lib" "$LLVM_INSTALL_DIR/lib64"; do
            for _f in "$_d"/libomptarget.so*; do
                [[ -e "$_f" ]] && { _llvm_omptarget="$_f"; break 2; }
            done
        done
        if [[ -n "$_llvm_omptarget" ]]; then
            success "libomptarget (offload) installed → $_llvm_omptarget"
        else
            warn "offload build finished but no host libomptarget.so was found under"
            warn "$LLVM_INSTALL_DIR/lib{,64} — OpenMP target links (-lomptarget) will fail."
            warn "Check the LLVM offload build output above."
        fi
        _mark_installed "$LLVM_INSTALL_DIR/.xkrt-offload-installed" "$LLVM_HASH"
    fi
fi

# ── xkomp tests (require the patched LLVM + custom libomptarget) ────────────────
# xkomp's tests/ link the OpenMP target runtime (libomptarget), which only exists
# after the offload stage above.  They were intentionally NOT built when the xkomp
# library was installed (libomptarget did not exist yet), so build them now.  This
# is best-effort: the libraries are already installed, so a test build failure
# warns instead of aborting the whole install.
if [[ "$INSTALL_XKOMP" == "true" && "${LLVM_BUILD_OFFLOAD:-false}" == "true" ]]; then
    step "Building xkomp tests (require libomptarget)"
    if [[ -f "${XKOMP_BUILD_DIR:-/nonexistent}/CMakeCache.txt" ]]; then
        cd "$XKOMP_BUILD_DIR"
        # 'make' now builds the remaining targets — the test executables — with the
        # patched clang, which can finally resolve -lomptarget from the LLVM install.
        if make -j "$(nproc)"; then
            success "xkomp tests built → $XKOMP_BUILD_DIR/tests"
            _tty "  ${DIM}Run them with:  ( source %s && cd %s && ctest --output-on-failure )${NC}\n" \
                 "$ENV_FILE" "$XKOMP_BUILD_DIR"
        else
            warn "xkomp tests failed to build — the xkomp library itself is installed and"
            warn "usable.  Inspect the output above (tests dir: $XKOMP_BUILD_DIR/tests)."
        fi
    else
        warn "xkomp tests skipped: no configured build dir at ${XKOMP_BUILD_DIR:-<unset>}."
        warn "xkomp was likely already installed from a previous run — re-run with --force"
        warn "to reconfigure and build the tests."
    fi
elif [[ "$INSTALL_XKOMP" == "true" ]]; then
    info "xkomp library installed; tests not built — they need the custom libomptarget"
    info "(enable the LLVM 'offload' runtime to build them)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done – print usage  (env.sh was written earlier, before the offload stage)
# ─────────────────────────────────────────────────────────────────────────────

_tty "\n"; hr
_tty "  ${BOLD}${GREEN}Installation complete!${NC}\n"; hr
_tty "\n  A source-able environment file was written — load everything with:\n\n"
_tty "    ${BOLD}source %s${NC}\n" "$ENV_FILE"
_tty "\n  ${DIM}(it runs the following; add them to your shell rc or a job script if you prefer)${NC}\n\n"
_tty "    module use %s\n" "$MODULES_DIR"
for line in ${MOD_LOAD[@]+"${MOD_LOAD[@]}"}; do
    _tty "    %s\n" "$line"
done
_tty "\n  Then compile against the installed libraries, for example:\n"
_tty "    clang++ main.cc -lxkblas\n"
_tty "\n  You may also test the installation by running 'xkrt_info'\n"
_tty "\n"
