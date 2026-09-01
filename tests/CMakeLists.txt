cmake_minimum_required(VERSION 3.20)
include(CTest)

project(tests CXX)

#
# XKOMP test-suite
# ----------------
# Two top-level families, each then organised by construct:
#
#   directive/   standard C/C++ OpenMP, written with `#pragma omp ...` only
#                (parallel, worksharing/{for,sections,single}, master, masked,
#                 barrier, atomic, critical, tasking, teams, composite, target,
#                 access)
#   xkomp/       tests of the xkomp++.h C++ interfaces (e.g.
#                pragma_omp_taskgraph(gid, flags, [&]{ ... }))
#
# The source-tree hierarchy is mirrored in the build tree: a test at
#   tests/<family>/<construct>/<name>.cc
# is built to
#   <build>/tests/<family>/<construct>/<name>
# (so two tests may share a short name without colliding).
#
# Each test documents its status at the top of the file:
#   // xkomp: supported          -> expected to PASS
#   // xkomp: XFAIL <reason>      -> construct not supported yet; registered with
#                                    xkomp_add_xfail_test() so it shows up as
#                                    FAILED in `make test` (gap tracking) until
#                                    the construct is implemented.
#
# All tests are compiled with -fopenmp (otherwise clang silently drops every
# `#pragma omp`) and linked against the in-tree xkomp library so the
# __kmpc_*/omp_* symbols resolve to it (libomp.so/libiomp5.so/libgomp.so are
# symlinks onto libxkomp.so created by the top-level build).
#
# Correctness is checked with assert() (see common.h), so the suite must be
# configured in Debug mode.
#

# --- enforce Debug so that asserts are live ----------------------------------
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR
        "The XKOMP test-suite checks correctness with assert(), so it must be "
        "configured in Debug mode.\nRe-run cmake with -DCMAKE_BUILD_TYPE=Debug or disable tests.")
endif()

find_package(HWLOC REQUIRED)
find_package(XKRT  REQUIRED)

# Per-test wall-clock timeout: a deadlocked/hung test is killed and reported as
# FAILED instead of hanging `make test` forever. Override with
#   -DXKOMP_TEST_TIMEOUT=<seconds>
set(XKOMP_TEST_TIMEOUT 5 CACHE STRING
    "Per-test timeout in seconds; a hung/deadlocked test is killed and reported as failed")

option(XKOMP_TESTS_TARGET
       "Build tests that use 'omp target' (require the offload runtime)" ON)

# The 'access' clause is an XKOMP extension that only the patched LLVM/clang can
# parse (https://github.com/anlsys/llvm-project, iwomp26-access-clause).  Turn
# this OFF when building the suite with a stock compiler.
option(XKOMP_TESTS_ACCESS
       "Build tests for the XKOMP 'access' clause (require the patched LLVM/clang)" ON)

# ---------------------------------------------------------------------------
# Offload target detection (for `omp target` tests)
# ---------------------------------------------------------------------------
# `target` tests must be compiled for the GPU actually present on the build
# machine (otherwise `omp target` regions never reach the device). We detect the
# offload triple + arch with the arch-probe tools that ship next to the compiler
# (nvptx-arch / amdgpu-arch -- the same probes clang's `--offload-arch=native`
# uses), falling back to `nvidia-smi`. Results are overridable cache variables:
#   -DXKOMP_TESTS_OFFLOAD_TRIPLE=nvptx64-nvidia-cuda -DXKOMP_TESTS_OFFLOAD_ARCH=sm_80
# When neither is found (and none given), offload is disabled and the `omp
# target` tests are skipped.
set(XKOMP_TESTS_OFFLOAD_TRIPLE "" CACHE STRING
    "Offload target triple for 'omp target' tests (auto-detected if empty)")
set(XKOMP_TESTS_OFFLOAD_ARCH   "" CACHE STRING
    "Offload GPU arch for 'omp target' tests, e.g. sm_80 / gfx90a (auto-detected if empty)")
option(XKOMP_TESTS_OFFLOAD_MANDATORY
       "Compile target tests with -fopenmp-offload-mandatory (enforce GPU offload)" ON)

# effective values (seeded from the cache overrides, else auto-detected)
set(_XK_OFF_TRIPLE "${XKOMP_TESTS_OFFLOAD_TRIPLE}")
set(_XK_OFF_ARCH   "${XKOMP_TESTS_OFFLOAD_ARCH}")

# only probe for a GPU when target tests are actually wanted
if(XKOMP_TESTS_TARGET AND (NOT _XK_OFF_TRIPLE OR NOT _XK_OFF_ARCH))
    get_filename_component(_xk_cc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_xk_det_triple "")
    set(_xk_det_arch   "")

    # 1) NVIDIA via nvptx-arch (ships with clang/offload; prints e.g. "sm_80")
    find_program(XKOMP_NVPTX_ARCH nvptx-arch HINTS "${_xk_cc_bin}")
    if(XKOMP_NVPTX_ARCH)
        execute_process(COMMAND "${XKOMP_NVPTX_ARCH}"
            OUTPUT_VARIABLE _xk_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _xk_rc)
        if(_xk_rc EQUAL 0 AND _xk_out)
            string(REGEX MATCHALL "[^\r\n]+" _xk_lines "${_xk_out}")
            list(GET _xk_lines 0 _xk_det_arch)          # first GPU's arch
            set(_xk_det_triple "nvptx64-nvidia-cuda")
        endif()
    endif()

    # 2) AMD via amdgpu-arch (prints e.g. "gfx90a")
    if(NOT _xk_det_arch)
        find_program(XKOMP_AMDGPU_ARCH amdgpu-arch HINTS "${_xk_cc_bin}")
        if(XKOMP_AMDGPU_ARCH)
            execute_process(COMMAND "${XKOMP_AMDGPU_ARCH}"
                OUTPUT_VARIABLE _xk_out OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _xk_rc)
            if(_xk_rc EQUAL 0 AND _xk_out)
                string(REGEX MATCHALL "[^\r\n]+" _xk_lines "${_xk_out}")
                list(GET _xk_lines 0 _xk_det_arch)
                set(_xk_det_triple "amdgcn-amd-amdhsa")
            endif()
        endif()
    endif()

    # 3) NVIDIA fallback via nvidia-smi (compute capability "8.0" -> "sm_80")
    if(NOT _xk_det_arch)
        find_program(XKOMP_NVIDIA_SMI nvidia-smi)
        if(XKOMP_NVIDIA_SMI)
            execute_process(COMMAND "${XKOMP_NVIDIA_SMI}"
                    --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE _xk_out OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _xk_rc)
            if(_xk_rc EQUAL 0 AND _xk_out)
                string(REGEX MATCHALL "[^\r\n]+" _xk_lines "${_xk_out}")
                list(GET _xk_lines 0 _xk_cap)           # e.g. "8.0"
                string(REPLACE "." "" _xk_cap "${_xk_cap}")
                set(_xk_det_arch "sm_${_xk_cap}")
                set(_xk_det_triple "nvptx64-nvidia-cuda")
            endif()
        endif()
    endif()

    if(NOT _XK_OFF_TRIPLE)
        set(_XK_OFF_TRIPLE "${_xk_det_triple}")
    endif()
    if(NOT _XK_OFF_ARCH)
        set(_XK_OFF_ARCH "${_xk_det_arch}")
    endif()
endif()

if(XKOMP_TESTS_TARGET AND _XK_OFF_TRIPLE AND _XK_OFF_ARCH)
    set(XKOMP_TESTS_OFFLOAD ON)
    message(STATUS "XKOMP tests: 'omp target' offload -> ${_XK_OFF_TRIPLE} (${_XK_OFF_ARCH})")
elseif(XKOMP_TESTS_TARGET)
    set(XKOMP_TESTS_OFFLOAD OFF)
    message(STATUS "XKOMP tests: no offload GPU detected -- 'omp target' tests are SKIPPED "
                   "(override with -DXKOMP_TESTS_OFFLOAD_TRIPLE=<triple> -DXKOMP_TESTS_OFFLOAD_ARCH=<arch>)")
else()
    set(XKOMP_TESTS_OFFLOAD OFF)  # target tests disabled -> offload irrelevant
endif()

set(XKOMP_TEST_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}        # tests/common.h
    ${CMAKE_SOURCE_DIR}/include        # <xkomp/*.h>
    ${CMAKE_BINARY_DIR}/include)       # generated <xkomp/support.h>

# Common settings shared by every test target.
function(_xkomp_test_setup tgt)
    target_compile_features(${tgt} PRIVATE cxx_std_20)
    # -fopenmp: lower the pragmas; -UNDEBUG: keep assert() live regardless.
    target_compile_options(${tgt} PRIVATE -fopenmp -g -UNDEBUG)
    target_link_options(${tgt}    PRIVATE -fopenmp)
    target_include_directories(${tgt} PRIVATE ${XKOMP_TEST_INCLUDES})
    # Put the top-level build dir first on the link path so the `-lomp` that
    # -fopenmp appends resolves to the libomp.so -> libxkomp.so symlink (whose
    # SONAME is libxkomp.so) rather than to clang's own libomp.
    target_link_directories(${tgt} BEFORE PRIVATE ${CMAKE_BINARY_DIR})
    if(TARGET compat_links)
        add_dependencies(${tgt} compat_links)
    endif()
    target_link_libraries(${tgt} PRIVATE ${XKOMP} ${HWLOC_LIBRARIES} m)
    # The xkomp/ (xkomp++.h) tests include <xkrt/*.h>; link the imported target so its
    # include directories propagate (xkomp links XKRT privately, so it does not
    # carry them transitively).  Fall back to the plain variables otherwise.
    if(TARGET XKRT::xkrt)
        target_link_libraries(${tgt} PRIVATE XKRT::xkrt)
    else()
        target_link_libraries(${tgt} PRIVATE ${XKRT_LIBRARIES})
        target_include_directories(${tgt} PRIVATE ${XKRT_INCLUDE_DIRS})
    endif()
endfunction()

# Derive, from a source path "<dir>/<base>.cc", a unique CMake target name, the
# mirrored output directory and the (hierarchical) CTest name.
macro(_xkomp_test_paths src)
    get_filename_component(_xk_dir  "${src}" DIRECTORY)   # e.g. directive/parallel
    get_filename_component(_xk_base "${src}" NAME_WE)     # e.g. omp_parallel
    set(_xk_test "${_xk_dir}/${_xk_base}")                # CTest name
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _xk_tgt "${_xk_test}")  # CMake target
endmacro()

# place the built executable under <build>/tests/<dir>/ with name <base>
function(_xkomp_test_output tgt dir base)
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
        OUTPUT_NAME "${base}")
endfunction()

# Expected-to-PASS test.  usage: xkomp_add_test(<source> <label>)
function(xkomp_add_test src label)
    _xkomp_test_paths("${src}")
    add_executable(${_xk_tgt} ${src})
    _xkomp_test_setup(${_xk_tgt})
    _xkomp_test_output(${_xk_tgt} "${_xk_dir}" "${_xk_base}")
    add_test(NAME "${_xk_test}" COMMAND ${_xk_tgt})
    set_tests_properties("${_xk_test}" PROPERTIES LABELS "${label}" TIMEOUT ${XKOMP_TEST_TIMEOUT})
endfunction()

# Test for a construct XKOMP does not support yet.  Built on demand (so a link
# failure does not break `make`) and driven through run_xfail.cmake, which runs
# the test as a normal FAILURE when the construct cannot be compiled/linked or
# aborts at run time.  It therefore shows up as "Failed" in `make test` -- so the
# gap is tracked -- and turns "Passed" once the construct is implemented.
function(xkomp_add_xfail_test src label)
    _xkomp_test_paths("${src}")
    add_executable(${_xk_tgt} EXCLUDE_FROM_ALL ${src})
    _xkomp_test_setup(${_xk_tgt})
    _xkomp_test_output(${_xk_tgt} "${_xk_dir}" "${_xk_base}")
    add_test(NAME "${_xk_test}"
        COMMAND ${CMAKE_COMMAND}
            -DTARGET=${_xk_tgt}
            -DBIN_DIR=${CMAKE_BINARY_DIR}
            -DEXE=$<TARGET_FILE:${_xk_tgt}>
            -DRUN_TIMEOUT=${XKOMP_TEST_TIMEOUT}
            -P ${CMAKE_SOURCE_DIR}/cmake/run_xfail.cmake)
    set_tests_properties("${_xk_test}" PROPERTIES LABELS "unsupported;${label}")
endfunction()

# Add the GPU offload flags to a target for the detected/overridden device.
function(_xkomp_test_offload_setup tgt)
    set(_off
        -fopenmp-targets=${_XK_OFF_TRIPLE}
        --offload-arch=${_XK_OFF_ARCH})
    if(XKOMP_TESTS_OFFLOAD_MANDATORY)
        list(APPEND _off -fopenmp-offload-mandatory)  # no host fallback: must offload
    endif()
    # offload flags are needed at BOTH compile and link (the driver embeds the
    # device image and links the offload runtime at link time)
    target_compile_options(${tgt} PRIVATE ${_off})
    target_link_options(${tgt}    PRIVATE ${_off})
    target_compile_options(${tgt} PRIVATE -fopenmp-version=60)
endfunction()

# Expected-to-PASS test compiled with the PACKED task-body JIT ABI
# (-fopenmp-task-jit-abi=packed): each task body is forwarded to CGIR as a
# self-contained void(void*,size_t) kernel over a packed capture buffer. Same
# source as xkomp_add_test but a distinct target/CTest/output name (suffix
# "_packed") so one source can be registered under both ABIs. Requires the
# patched clang that knows the flag (as the whole taskgraph/JIT group does).
function(xkomp_add_packed_test src label)
    _xkomp_test_paths("${src}")
    set(_xk_tgt  "${_xk_tgt}_packed")
    set(_xk_test "${_xk_test}_packed")
    set(_xk_base "${_xk_base}_packed")
    add_executable(${_xk_tgt} ${src})
    _xkomp_test_setup(${_xk_tgt})
    target_compile_options(${_xk_tgt} PRIVATE -fopenmp-task-jit-abi=packed)
    _xkomp_test_output(${_xk_tgt} "${_xk_dir}" "${_xk_base}")
    add_test(NAME "${_xk_test}" COMMAND ${_xk_tgt})
    set_tests_properties("${_xk_test}" PROPERTIES LABELS "${label};packed" TIMEOUT ${XKOMP_TEST_TIMEOUT})
endfunction()

# Expected-to-PASS test compiled with the POINTERS task-body JIT ABI
# (-fopenmp-task-jit-abi=pointers): each task body is forwarded to CGIR as a
# void(void**) kernel with one deduplicable per-value parameter per capture.
# The compiler default is now 'none' (no IR forwarded), so this variant is what
# keeps the pointers JIT/fusion path covered in-tree. Suffix "_pointers" so one
# source can be registered under both ABIs (mirrors xkomp_add_packed_test).
function(xkomp_add_pointers_test src label)
    _xkomp_test_paths("${src}")
    set(_xk_tgt  "${_xk_tgt}_pointers")
    set(_xk_test "${_xk_test}_pointers")
    set(_xk_base "${_xk_base}_pointers")
    add_executable(${_xk_tgt} ${src})
    _xkomp_test_setup(${_xk_tgt})
    target_compile_options(${_xk_tgt} PRIVATE -fopenmp-task-jit-abi=pointers)
    _xkomp_test_output(${_xk_tgt} "${_xk_dir}" "${_xk_base}")
    add_test(NAME "${_xk_test}" COMMAND ${_xk_tgt})
    set_tests_properties("${_xk_test}" PROPERTIES LABELS "${label};pointers" TIMEOUT ${XKOMP_TEST_TIMEOUT})
endfunction()

# Expected-to-PASS `omp target` test: like xkomp_add_test but compiled for the
# machine's GPU. Skipped entirely when no offload device was detected (see the
# offload-detection block above), since it cannot validate offload without one.
function(xkomp_add_target_test src label)
    if(NOT XKOMP_TESTS_OFFLOAD)
        return()
    endif()
    _xkomp_test_paths("${src}")
    add_executable(${_xk_tgt} ${src})
    _xkomp_test_setup(${_xk_tgt})
    _xkomp_test_offload_setup(${_xk_tgt})
    _xkomp_test_output(${_xk_tgt} "${_xk_dir}" "${_xk_base}")
    add_test(NAME "${_xk_test}" COMMAND ${_xk_tgt})
    set_tests_properties("${_xk_test}" PROPERTIES LABELS "${label};gpu" TIMEOUT ${XKOMP_TEST_TIMEOUT})
endfunction()

##############################################################################
# directive/ : standard C/C++ OpenMP (#pragma omp ...)                         #
##############################################################################

# --- parallel ---------------------------------------------------------------
xkomp_add_test(directive/parallel/omp_parallel.cc               parallel)
xkomp_add_test(directive/parallel/omp_parallel_num_threads.cc   parallel)
xkomp_add_test(directive/parallel/omp_parallel_firstprivate.cc  parallel)
xkomp_add_test(directive/parallel/omp_parallel_nested.cc        parallel)
xkomp_add_xfail_test(directive/parallel/omp_parallel_reduction.cc parallel)
xkomp_add_xfail_test(directive/parallel/omp_parallel_if.cc        parallel)

# --- worksharing : for / sections / single ----------------------------------
xkomp_add_test(directive/worksharing/for/omp_for.cc                 worksharing)
xkomp_add_test(directive/worksharing/for/omp_for_lastprivate.cc     worksharing)
xkomp_add_test(directive/worksharing/for/omp_for_nowait.cc          worksharing)
xkomp_add_test(directive/worksharing/for/omp_for_unsigned.cc        worksharing)
xkomp_add_test(directive/worksharing/for/omp_for_increments.cc      worksharing)
xkomp_add_test(directive/worksharing/for/omp_parallel_for.cc        worksharing)
xkomp_add_xfail_test(directive/worksharing/for/omp_for_schedule_dynamic.cc worksharing)
xkomp_add_xfail_test(directive/worksharing/for/omp_for_static_chunk.cc      worksharing)
xkomp_add_xfail_test(directive/worksharing/for/omp_for_collapse.cc          worksharing)
xkomp_add_xfail_test(directive/worksharing/for/omp_for_int64.cc             worksharing)
xkomp_add_xfail_test(directive/worksharing/for/omp_for_reduction.cc         worksharing)

xkomp_add_xfail_test(directive/worksharing/sections/omp_sections.cc         worksharing)

xkomp_add_test(directive/worksharing/single/omp_single.cc           worksharing)
xkomp_add_test(directive/worksharing/single/omp_single_nowait.cc    worksharing)
xkomp_add_xfail_test(directive/worksharing/single/omp_single_copyprivate.cc worksharing)

# --- master / masked --------------------------------------------------------
xkomp_add_test(directive/master/omp_master.cc                   master)
xkomp_add_test(directive/masked/omp_masked.cc                  masked)

# --- barrier / atomic / critical --------------------------------------------
xkomp_add_test(directive/barrier/omp_barrier.cc                 barrier)
xkomp_add_test(directive/atomic/omp_atomic.cc                   atomic)
xkomp_add_xfail_test(directive/critical/omp_critical.cc         critical)

# --- tasking ----------------------------------------------------------------
xkomp_add_test(directive/tasking/omp_task.cc                    tasking)
xkomp_add_test(directive/tasking/omp_taskwait.cc               tasking)
xkomp_add_test(directive/tasking/omp_task_depend.cc            tasking)
xkomp_add_test(directive/tasking/omp_task_depend_inoutset.cc   tasking)
xkomp_add_test(directive/tasking/omp_task_depend_mutexinoutset.cc tasking)
xkomp_add_test(directive/tasking/omp_task_dag.cc               tasking)
xkomp_add_test(directive/tasking/omp_task_axpy.cc              tasking)
xkomp_add_test(directive/tasking/omp_task_nested.cc            tasking)
xkomp_add_test(directive/tasking/omp_parallel_tasks.cc         tasking)
xkomp_add_test(directive/tasking/omp_task_untied.cc            tasking)
xkomp_add_xfail_test(directive/tasking/omp_taskyield.cc        tasking)
xkomp_add_test(directive/tasking/omp_taskgroup.cc             tasking)
xkomp_add_test(directive/tasking/omp_task_if.cc               tasking)

# --- teams (host) -- not implemented ----------------------------------------
xkomp_add_xfail_test(directive/teams/omp_teams.cc              teams)

# --- taskgraph (directive form) -- `#pragma omp taskgraph` not in the compiler
#     yet (the xkomp/ tests cover the same feature via pragma_omp_taskgraph)
xkomp_add_xfail_test(directive/taskgraph/omp_taskgraph.cc      taskgraph)

# --- composite : several constructs combined --------------------------------
xkomp_add_test(directive/composite/omp_nested_worksharing.cc   composite)
xkomp_add_test(directive/composite/omp_manual_reduction.cc     composite)
xkomp_add_test(directive/composite/omp_producer_consumer.cc    composite)
xkomp_add_test(directive/composite/omp_parallel_matvec.cc      composite)

# --- target -- needs the offload runtime + a GPU, hence guarded -------------
if(XKOMP_TESTS_TARGET)
    xkomp_add_target_test(directive/target/omp_target_map.cc          target)
    xkomp_add_target_test(directive/target/omp_target_update.cc       target)
    xkomp_add_target_test(directive/target/omp_target_data.cc         target)
    xkomp_add_target_test(directive/target/omp_target_axpy.cc         target)
    xkomp_add_target_test(directive/target/omp_target_axpy_deps.cc    target)
    xkomp_add_target_test(directive/target/omp_target_dot.cc          target)
endif()

# --- access clause (XKOMP extension) -- needs the patched LLVM/clang ---------
if(XKOMP_TESTS_ACCESS)
    xkomp_add_test(directive/access/omp_access.cc              access)
    xkomp_add_test(directive/access/omp_access_regions.cc      access)
    xkomp_add_test(directive/access/omp_access_concurrent.cc   access)
    xkomp_add_test(directive/access/omp_access_war_waw.cc      access)
    xkomp_add_test(directive/access/omp_access_dag.cc          access)
    # target + access: also needs the offload runtime + a GPU
    if(XKOMP_TESTS_TARGET)
        xkomp_add_target_test(directive/access/omp_access_target.cc          access)
        xkomp_add_target_test(directive/access/omp_access_target_overlap.cc  access)
        xkomp_add_target_test(directive/access/omp_access_stencil_target.cc  access)
    endif()
endif()

##############################################################################
# ompt/ : OMPT bridge (XKRT-T -> OMPT)                                        #
##############################################################################
# Builds a small standard OMPT tool (loadable module) and a plain-OpenMP driver.
# The driver runs with OMP_TOOL_LIBRARIES pointing at the tool; the test passes
# when the tool prints "XKOMP-OMPT:OK" from its finalize callback. Requires
# XKOMP built with -DSUPPORT_OMPT=ON (and XKRT with -DUSE_TOOLS=ON).
if(SUPPORT_OMPT)
    # standard OMPT tool (.so) -- only needs the vendored omp-tools.h, no xkomp link
    add_library(xkomp_ompt_tool MODULE ompt/tool.cc)
    target_compile_features(xkomp_ompt_tool PRIVATE cxx_std_20)
    target_include_directories(xkomp_ompt_tool PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # plain-OpenMP driver, linked to the in-tree xkomp like every other test
    add_executable(ompt_driver ompt/driver.cc)
    _xkomp_test_setup(ompt_driver)
    _xkomp_test_output(ompt_driver "ompt" "driver")

    add_test(NAME "ompt/driver" COMMAND ompt_driver)
    set_tests_properties("ompt/driver" PROPERTIES
        ENVIRONMENT "OMP_TOOL_LIBRARIES=$<TARGET_FILE:xkomp_ompt_tool>"
        PASS_REGULAR_EXPRESSION "XKOMP-OMPT:OK"
        FAIL_REGULAR_EXPRESSION "XKOMP-OMPT:MISSING"
        LABELS "ompt")
endif()

##############################################################################
# xkomp/ : tests of the xkomp++.h C++ interfaces                            #
##############################################################################

# --- taskgraph (pragma_omp_taskgraph) ---------------------------------------
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host.cc        taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_global.cc  taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_local.cc   taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_host.cc              taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_host_deps.cc         taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_ptr.cc     taskgraph)
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_local.cc  taskgraph)

# firstprivate STRUCT (aggregate) capture. Under the default (pointers) ABI this
# falls back to the packed args[0]==tt kernel (aggregate not fusable); under the
# packed ABI it uses the self-contained aggregate-memcpy kernel. Registered under
# both to check the fallback stays correct and the packed aggregate path works.
xkomp_add_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_struct.cc taskgraph)

# pointers task-body JIT ABI (-fopenmp-task-jit-abi=pointers) coverage. The
# compiler default is 'none' (no IR forwarded), so these explicit variants keep
# the pointers JIT/fusion path exercised:
#   axpby_host        : scalar firstprivate only + externalized globals (no captures)
#   axpby_host_local  : scalar firstprivate + shared-array (SHARED_ADDR) chain
#   axpby_host_struct : aggregate (struct) firstprivate -> packed args[0]==tt fallback
#   axpy_host_global  : externalized global (no captures)
#   axpy_host_local   : shared-array capture (SHARED_ADDR)
#   axpy_host_ptr     : shared-pointer capture (SHARED_PTR)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host.cc        taskgraph)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_local.cc  taskgraph)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_struct.cc taskgraph)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_global.cc  taskgraph)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_local.cc   taskgraph)
xkomp_add_pointers_test(xkomp/taskgraph/xkomp_taskgraph_axpy_host_ptr.cc     taskgraph)

# packed task-body JIT ABI (-fopenmp-task-jit-abi=packed) coverage:
#   host        : scalar firstprivate only + externalized globals (no captures)
#   host_local  : scalar firstprivate + shared-array (SHARED_ADDR) chain
#   host_struct : aggregate (struct) firstprivate + shared-array chain
xkomp_add_packed_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host.cc        taskgraph)
xkomp_add_packed_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_local.cc  taskgraph)
xkomp_add_packed_test(xkomp/taskgraph/xkomp_taskgraph_axpby_host_struct.cc taskgraph)

# target/offload taskgraphs
if(XKOMP_TESTS_TARGET)
    xkomp_add_target_test(xkomp/taskgraph/xkomp_taskgraph_axpby_target.cc  taskgraph)
    xkomp_add_target_test(xkomp/taskgraph/xkomp_taskgraph_dot_target.cc    taskgraph)
endif()
