#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "ompss-2_settings.h"
#include "ompss-2_wrappers.h"

#define xPRAGMA(x) PRAGMA(x)
#define PRAGMA(x) _Pragma(#x)

#define GRAINSIZE(wrapper_label) CONC_(wrapper_label, grainsize)
#define COLLAPSE_DEPTH(wrapper_label) CONC_(wrapper_label, collapse_depth)
#define COLLAPSED_LENGTHS(wrapper_label) CONC_(wrapper_label, collapse_length)
#define PARAMS(wrapper_label) CONC_(wrapper_label, params_name)
#define LAYER_DEPENDENCIES(wrapper_label) CONC_(wrapper_label, layer_deps)
#define TRACE_LABEL(wrapper_label) CONC_(wrapper_label, trace_label)
#define ATTENTION_RELEASE_CONDITION(wrapper_label) CONC_(wrapper_label, release_condition)
#define ATTENTION_DEPENDENCIES(wrapper_label) CONC_(wrapper_label, deps)

#ifndef ENABLE_TASKITER
    #define RELEASE_DEPENDENCY(dependency, clause) xPRAGMA(oss release clause(dependency))
#else // Release clause is not compatible with taskiter
    #define RELEASE_DEPENDENCY(dependency, clause) ;
#endif

enum dependency_state_t { IN = 0, CONCURRENT };
#define DEP_STATE_DEFAULT IN

// Private
struct DepHandler {
    int *tok_deps; // Addresses for our dependencies
    int B;         // Batch size
    int T;         // Token sequence size
    int slice_B;   // Current slicing size on batch dimension
    int slice_T;   // Current slicing size on token sequence dimension

    int dep_B_subsize; // Min slice size allowed
    int dep_T_subsize; // Min slice size allowed
    int dep_B_size;
    int dep_T_size;
    enum dependency_state_t dep_state; // We alternate between in and concurrent to allow inner tasks unrolling
};

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------ PRIVATE FUNCTIONS AND MACROS ----------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

void _dep_check(struct DepHandler *dep_handler) {
    if (dep_handler == NULL) {
        printf("ERROR : Given pointer to dependency handler is NULL\n");
        exit(EXIT_FAILURE);
    }
    if (dep_handler->tok_deps == NULL) {
        printf("ERROR : Unexpected configuration\n");
        exit(EXIT_FAILURE);
    }
    if (dep_handler->slice_B == 0) {
        printf("ERROR : No slice shape has been defined yet\n");
        exit(EXIT_FAILURE);
    }
}

int _dep_id_to_dep(struct DepHandler *dep_handler, int tok_id, int off_b, int off_t) {
    int seq_dep = off_b + ((tok_id / (dep_handler->slice_T / dep_handler->dep_T_subsize))) * dep_handler->dep_T_size;
    int tok_dep = off_t + (tok_id % (dep_handler->slice_T / dep_handler->dep_T_subsize));
    return seq_dep + tok_dep;
}

void _dep_set_slh(struct DepHandler *dep_handler, struct SliceHandler *slh, int t, int b) {
    slh->b_start = b;
    slh->b_end = b + dep_handler->slice_B;
    slh->t_start = t;
    slh->t_end = t + dep_handler->slice_T;
}

void _dep_switch_dep_state(struct DepHandler *dep_handler) { dep_handler->dep_state = dep_handler->dep_state == IN ? CONCURRENT : IN; }

// Generate the multi-deps for OmpSs-2
#define GET_DEP(dep_handler, off_b, off_t, slice_flat_size)                                                                                               \
    {                                                                                                                                                     \
        dep_handler->tok_deps[_dep_id_to_dep(dep_handler, id, off_b, off_t)], id = 0;                                                                     \
        slice_flat_size                                                                                                             \
    }

int count = 0;
void reshape_dimensions_from_flatten_id(int *reshape_out, int collapse, size_t *dims, size_t id) {
    size_t buffer = id;
    for (int dim = collapse - 1; dim >= 0; dim--) {
        reshape_out[dim] = buffer % dims[dim];
        buffer /= dims[dim];
    }
}

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- INTERFACE FUNCTIONS ---------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

int *dep_init(int B, int T, int dep_size_B, int dep_size_T) {
    assert((B % dep_size_B) == 0);
    assert((T % dep_size_T) == 0);
    assert(B > 0);
    assert(T > 0);
    assert(dep_size_B > 0);
    assert(dep_size_T > 0);
    assert(dep_size_B <= B);
    assert(dep_size_T <= T);

    #ifdef __cplusplus
    struct DepHandler *dep_handler = new DepHandler;
    #else
    struct DepHandler *dep_handler = malloc(sizeof(struct DepHandler));
    #endif

    dep_handler->B = B;
    dep_handler->T = T;
    dep_handler->dep_B_subsize = dep_size_B;
    dep_handler->dep_T_subsize = dep_size_T;
    dep_handler->dep_B_size = B / dep_size_B;
    dep_handler->dep_T_size = T / dep_size_T;

    dep_handler->slice_B = 0;
    dep_handler->slice_T = 0;
    dep_handler->dep_state = DEP_STATE_DEFAULT;

    #ifdef __cplusplus
    dep_handler->tok_deps = new int[dep_handler->dep_T_size * dep_handler->dep_B_size];
    #else
    dep_handler->tok_deps = malloc(dep_handler->dep_T_size * dep_handler->dep_B_size * sizeof(int));
    #endif

    return (int *)dep_handler;
}

void dep_finish(int *dep_handler) {
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;
    if (_dep_handler != NULL)
        if (_dep_handler->tok_deps != NULL)
            free(_dep_handler->tok_deps);
    free(_dep_handler);
}

void dep_set_slice_shape(int *dep_handler, int slice_size_B, int slice_size_T) {
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;
    assert(slice_size_B > 0);
    assert(slice_size_T > 0);
    assert(slice_size_B % _dep_handler->dep_B_subsize == 0);
    assert(slice_size_T % _dep_handler->dep_T_subsize == 0);

    _dep_handler->slice_B = slice_size_B;
    _dep_handler->slice_T = slice_size_T;
}

void dep_reset(int *dep_handler) {
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;
    _dep_switch_dep_state(_dep_handler);
    if (_dep_handler->dep_state != DEP_STATE_DEFAULT) {
        if (_dep_handler->dep_state == IN) {
            xPRAGMA(oss task label("dep_reset") in({
                _dep_handler->tok_deps[i], i = 0;
                _dep_handler->dep_T_size * _dep_handler->dep_B_size
            })) // Equivalent to a barrier for taskiter
            {}
        } else {
            xPRAGMA(oss task label("dep_reset") concurrent({
                _dep_handler->tok_deps[i], i = 0;
                _dep_handler->dep_T_size * _dep_handler->dep_B_size
            })) // Equivalent to a barrier for taskiter
            {}
        }
    }
    _dep_handler->dep_state = DEP_STATE_DEFAULT; // Reset the state coin
}

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------  WRAPPERS FUNCTIONS IMPLEMENTATION --------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

// If OmpSs-2 is disable, just by-pass everything and call directly the called kernel
#define BY_PASS(wrapper_label, slh)                                                                                                                       \
    const int collapse = COLLAPSE_DEPTH(wrapper_label);                                                                                                   \
    assert(collapse <= DIMS_MAX);                                                                                                                         \
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;                                                                                   \
    int sliced_B = _dep_handler->B; /* These two variables can be used to define grainsize */                                                             \
    int sliced_T = _dep_handler->T;                                                                                                                       \
    size_t dims[DIMS_MAX] = {COLLAPSED_LENGTHS(wrapper_label)};                                                                                           \
    for (int dim = 0; dim < DIMS_MAX; dim++) { /* Set the value of the loops not affected by the collapse, for usage convenience */                       \
        slh.dims_starts[dim] = 0;                                                                                                                         \
        slh.dims_stops[dim] = dims[dim];                                                                                                                  \
    }                                                                                                                                                     \
    slh.b_end = _dep_handler->B;                                                                                                                          \
    slh.t_end = _dep_handler->T;                                                                                                                          \
    func(PARAMS(wrapper_label), slh);


#if 0

// The goal of this function bellow is allow to paralellism in the tokens computation by slicing it two times
// The first slicing is over B and T, and will be setup with dependencies, allowing thus to have groups of tokens
// computed in parallel across the layer, to overlap computation
// The other 'inner' loop will leverage parallel computation inside the group of tokens. It is equivalent to a manually written taskloop.
// The main idea here is to removed what would have been written through nested tasks, in order to leverage the taskiter pragma used in the main loop
// #define COMMOM_SLICING(func, args, flabel, layers_dependencies, grainsize, collapse, lengths, slh)
#define COMMOM_SLICING(wrapper_label, slh)                                                                                                                \
    /* func is the kernel to call;                                                                                                                        \
    args is its arguments names;                                                                                                                          \
    flabel is the label for the oss pragma;                                                                                                               \
    layer_dependencies is the concurrent dependency clause for the parameters;                                                                            \
    grainsize is the grainsize of the manually-written taskloop;                                                                                          \
    collapse is the collapse depth for the taskloop;                                                                                                      \
    lengths are the lengths of the loop inside the taskloop. We are also allowing to define the lengths of loops inside the taskloop but outside the      \
    collapse depth;                                                                                                                                       \
    slh is a SliceHandler instance  */                                                                                                                    \
    const int collapse = COLLAPSE_DEPTH(wrapper_label);                                                                                                   \
    const size_t grainsize = GRAINSIZE(wrapper_label);                                                                                                       \
    assert(collapse <= DIMS_MAX);                                                                                                                         \
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;                                                                                   \
    _dep_check(_dep_handler);                                                                                                                             \
    _dep_switch_dep_state(_dep_handler);                                                                                                                  \
                                                                                                                                                          \
    /* Define the flatten size of a dependency block */                                                                                                   \
    int slice_dep_flat_size = (_dep_handler->slice_B / _dep_handler->dep_B_subsize) * (_dep_handler->slice_T / _dep_handler->dep_T_subsize);              \
    int sliced_B = _dep_handler->slice_B; /* These two variables can be used to define grainsize */                                                       \
    int sliced_T = _dep_handler->slice_T;                                                                                                                 \
    size_t dims[DIMS_MAX] = {COLLAPSED_LENGTHS(wrapper_label)}; /* Dimensions after slicing of the loops that can be collapse */                          \
    for (int dim = collapse; dim < DIMS_MAX; dim++) {           /* Set the value of the loops not affected by the collapse, for usage convenience */      \
        slh.dims_starts[dim] = 0;                                                                                                                         \
        slh.dims_stops[dim] = dims[dim];                                                                                                                  \
    }                                                                                                                                                     \
    assert(grainsize <= dims[collapse - 1]); /* Otherwise the collapse depth should be reduced*/                                                          \
                                                                                                                                                          \
    /* First slicing over B and T */                                                                                                                      \
    for (int b = 0; b < _dep_handler->B; b += _dep_handler->slice_B) {                                                                                    \
        int dep_b = (b / _dep_handler->dep_B_subsize) * _dep_handler->dep_T_size; /* First dependency idx of the batch(es) in the dependency array */     \
        for (int t = 0; t < _dep_handler->T; t += _dep_handler->slice_T) {                                                                                \
            int dep_t = (t / _dep_handler->slice_T) * (_dep_handler->slice_T / _dep_handler->dep_T_subsize); /* First dependency idx of the token(s) */   \
            _dep_set_slh(_dep_handler, &slh, t, b);                                                                                                       \
                                                                                                                                                          \
            /* We are now inside a slice, and need to reproduce manually a taskloop */                                                                    \
            /* We are flattening because the number of loops is not pre-determined */                                                                     \
            size_t flatten_length = 1; /* Flatten size of our manually defined taskloop */                                                                \
            for (long dim = collapse - 1; dim >= 0; dim--)                                                                                                \
                flatten_length *= dims[dim];                                                                                                              \
                                                                                                                                                          \
            for (size_t flat_id = 0; flat_id < flatten_length; flat_id += grainsize) {                                                                    \
                /* Lengths of the task from the taskloop are written in slh.dims_starts/stops */                                                          \
                reshape_dimensions_from_flatten_id(slh.dims_starts, collapse, dims, flat_id);                                                             \
                for (int dim = 0; dim < collapse - 1; dim++)                                                                                              \
                    slh.dims_stops[dim] = slh.dims_starts[dim] + 1; /* The grainsize only affects the innermost loop. The outer ones have size 1 */       \
                if (flat_id + grainsize >= flatten_length) /*For the last iteration of the innermost loop, loop length can be smaller than grainsize*/    \
                    slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + flatten_length - flat_id;                                              \
                else /* Otherwise we use the defined grainsize */                                                                                         \
                    slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + grainsize;                                                             \
                                                                                                                                                          \
                /* Create the corresponding task */                                                                                                       \
                /* We are using here a coin to use once in two the in or concurrent dependency clause. This is to allow tasks from the taskloop to        \
                overlap, while avoiding tasks from two different layers to overlap if they have the same dependencies */                                  \
                if (_dep_handler->dep_state == IN) {                                                                                                      \
                    xPRAGMA(oss task label(TRACE_LABEL(wrapper_label)) in(GET_DEP(_dep_handler, dep_b, dep_t, slice_dep_flat_size))LAYER_DEPENDENCIES(    \
                        wrapper_label) firstprivate(PARAMS(wrapper_label)) firstprivate(func)) {                                                          \
                        func(PARAMS(wrapper_label), slh);                                                                                                 \
                    }                                                                                                                                     \
                } else {                                                                                                                                  \
                    xPRAGMA(oss task label(TRACE_LABEL(wrapper_label)) concurrent(GET_DEP(_dep_handler, dep_b, dep_t, slice_dep_flat_size))               \
                                LAYER_DEPENDENCIES(wrapper_label) firstprivate(PARAMS(wrapper_label)) firstprivate(func)) {                               \
                        func(PARAMS(wrapper_label), slh);                                                                                                 \
                    }                                                                                                                                     \
                }                                                                                                                                         \
            }                                                                                                                                             \
        }                                                                                                                                                 \
    }

#define ATTENTION_FORWARD_release_condition (dep_first_idx > i || i >= (dep_first_idx + slice_dep_flat_size))
#define ATTENTION_BACKWARD_Q_release_condition (i >= (dep_first_idx + slice_dep_flat_size))

#define ATTENTION_FORWARD_deps                                                                                                                            \
    tok_deps[dep_b + k], k = 0;                                                                                                                           \
    seq_dep_flat_size
#define ATTENTION_BACKWARD_Q_deps                                                                                                                         \
    tok_deps[dep_b + k], k = 0;                                                                                                                           \
    seq_dep_flat_size

// This code is the same than the one defined in 'COMMOM_SLICING'. The only difference is that we put a barrier defined with dependencies at the beginning
// of this section. We are also using the release clause to avoid having an implicit barrier at the end of this section
#define ATTENTION_SLICING(wrapper_label, slh)                                                                                                             \
    assert(COLLAPSE_DEPTH(wrapper_label) <= DIMS_MAX);                                                                                                    \
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;                                                                                   \
    _dep_check(_dep_handler);                                                                                                                             \
    _dep_switch_dep_state(_dep_handler);                                                                                                                  \
    int *tok_deps = _dep_handler->tok_deps;                                                                                                               \
    int seq_dep_flat_size = (_dep_handler->slice_B / _dep_handler->dep_B_subsize) * (_dep_handler->dep_T_size);                                           \
    int slice_dep_flat_size = (_dep_handler->slice_B / _dep_handler->dep_B_subsize) * (_dep_handler->slice_T / _dep_handler->dep_T_subsize);              \
    int sliced_B = _dep_handler->slice_B;                                                                                                                 \
    int sliced_T = _dep_handler->slice_T;                                                                                                                 \
    const size_t grainsize = GRAINSIZE(wrapper_label);                                                                                                             \
    int collapse = COLLAPSE_DEPTH(wrapper_label);                                                                                                         \
    size_t dims[COLLAPSE_DEPTH(wrapper_label)] = {COLLAPSED_LENGTHS(wrapper_label)};                                                                      \
    assert(grainsize <= dims[collapse - 1]);                                                                                                              \
    for (int b = 0; b < _dep_handler->B; b += _dep_handler->slice_B) {                                                                                    \
        int dep_b = (b / _dep_handler->dep_B_subsize) * _dep_handler->dep_T_size;                                                                         \
        for (int t = 0; t < _dep_handler->T; t += _dep_handler->slice_T) {                                                                                \
            int dep_t = (t / (_dep_handler->slice_T)) * ((_dep_handler->slice_T) / _dep_handler->dep_T_subsize);                                          \
            _dep_set_slh(_dep_handler, &slh, t + 0, b); /* Offset if a token sequence is on multiple MPI ranks */                                         \
            int dep_first_idx = _dep_id_to_dep(_dep_handler, 0, dep_b, dep_t);                                                                            \
            size_t flatten_length = 1;                                                                                                                    \
            for (long dim = collapse - 1; dim >= 0; dim--)                                                                                                \
                flatten_length *= dims[dim];                                                                                                              \
                                                                                                                                                          \
            for (size_t flat_id = 0; flat_id < flatten_length; flat_id += grainsize) {                                                                    \
                reshape_dimensions_from_flatten_id(slh.dims_starts, collapse, dims, flat_id);                                                             \
                for (int dim = 0; dim < collapse - 1; dim++)                                                                                              \
                    slh.dims_stops[dim] = slh.dims_starts[dim] + 1;                                                                                       \
                if (flat_id + grainsize >= flatten_length)                                                                                                \
                    slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + flatten_length - flat_id;                                              \
                else                                                                                                                                      \
                    slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + grainsize;                                                             \
                if (_dep_handler->dep_state == IN) {                                                                                                      \
                    xPRAGMA(oss task label(TRACE_LABEL(wrapper_label)) in({ATTENTION_DEPENDENCIES(wrapper_label)})LAYER_DEPENDENCIES(wrapper_label)       \
                                firstprivate(PARAMS(wrapper_label), slh, func)) {                                                                         \
                        for (int i = dep_b; i < dep_b + seq_dep_flat_size; i++) {                                                                         \
                            if (ATTENTION_RELEASE_CONDITION(wrapper_label)) /* Keep only the dependencies relative to the slice*/                         \
                                RELEASE_DEPENDENCY(tok_deps[i], in)                                                                                       \
                        }                                                                                                                                 \
                                                                                                                                                          \
                        func(PARAMS(wrapper_label), slh);                                                                                                 \
                    }                                                                                                                                     \
                } else {                                                                                                                                  \
                    xPRAGMA(oss task label(TRACE_LABEL(wrapper_label)) concurrent({ATTENTION_DEPENDENCIES(wrapper_label)})                                \
                                LAYER_DEPENDENCIES(wrapper_label) firstprivate(PARAMS(wrapper_label), slh, func)) {                                       \
                        for (int i = dep_b; i < dep_b + seq_dep_flat_size; i++) {                                                                         \
                            if (ATTENTION_RELEASE_CONDITION(wrapper_label)) /* Keep only the dependencies relative to the slice*/                         \
                                RELEASE_DEPENDENCY(tok_deps[i], concurrent)                                                                               \
                        }                                                                                                                                 \
                                                                                                                                                          \
                        func(PARAMS(wrapper_label), slh);                                                                                                 \
                    }                                                                                                                                     \
                }                                                                                                                                         \
            }                                                                                                                                             \
        }                                                                                                                                                 \
    }

#ifdef OSS
// This kernel does not use the first slicing, thus it is only keeping the manually written taskloop
void MATMUL_PARAMS_BACKWARD_wrapper_name(void (*func)(MATMUL_PARAMS_BACKWARD_header, struct SliceHandler slh), MATMUL_PARAMS_BACKWARD_header,
                                         int *dep_handler, float *start_dep) {
    size_t grainsize = MATMUL_PARAMS_BACKWARD_grainsize;
    int collapse = MATMUL_PARAMS_BACKWARD_collapse_depth;
    assert(collapse <= DIMS_MAX);
    struct DepHandler *_dep_handler = (struct DepHandler *)dep_handler;
    struct SliceHandler slh = {0};
    _dep_check(_dep_handler);
    size_t dims[MATMUL_PARAMS_BACKWARD_collapse_depth] = {MATMUL_PARAMS_BACKWARD_collapse_length};
    assert(grainsize <= dims[collapse - 1]); /* Otherwise the collapse depth should be reduced*/ /* First slicing */
    size_t flatten_length = 1;
    for (long dim = collapse - 1; dim >= 0; dim--)
        flatten_length *= dims[dim];
    /* We are flattening because the number of loops is not constant */
    for (size_t flat_id = 0; flat_id < flatten_length; flat_id += grainsize) {
        reshape_dimensions_from_flatten_id(slh.dims_starts, collapse, dims, flat_id);
        for (int dim = 0; dim < collapse - 1; dim++)
            slh.dims_stops[dim] = slh.dims_starts[dim] + 1;
        if (flat_id + grainsize >= flatten_length)
            slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + flatten_length - flat_id;
        else
            slh.dims_stops[collapse - 1] = slh.dims_starts[collapse - 1] + grainsize;
        xPRAGMA(oss task label(MATMUL_PARAMS_BACKWARD_trace_label)
                    MATMUL_PARAMS_BACKWARD_layer_deps firstprivate(MATMUL_PARAMS_BACKWARD_params_name, slh, func, start_dep)) {
            func(MATMUL_PARAMS_BACKWARD_params_name, slh);
        }
    }
}
#else
void MATMUL_PARAMS_BACKWARD_wrapper_name(void (*func)(MATMUL_PARAMS_BACKWARD_header, struct SliceHandler slh), MATMUL_PARAMS_BACKWARD_header,
                                         int *dep_handler, float *start_dep) {
    struct SliceHandler slh = {0};
    BY_PASS(MATMUL_PARAMS_BACKWARD, slh)
}
#endif

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------ GENERATE WRAPPERS FUNCTIONS ------------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */
#ifdef OSS
    #define GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(wrapper_label)                                                                                             \
        GEN_DEPENDENCY_WRAPPER_HEADER(wrapper_label) {                                                                                                    \
            struct SliceHandler slh = {0};                                                                                                                \
            COMMOM_SLICING(wrapper_label, slh)                                                                                                            \
        }

    #define GEN_ATTENTION_DEPENDENCY_WRAPPER_FUNC(wrapper_label)                                                                                          \
        GEN_DEPENDENCY_WRAPPER_HEADER(wrapper_label) {                                                                                                    \
            struct SliceHandler slh = {0};                                                                                                                \
            ATTENTION_SLICING(wrapper_label, slh)                                                                                                         \
        }
#else
    #define GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(wrapper_label)                                                                                             \
        GEN_DEPENDENCY_WRAPPER_HEADER(wrapper_label) {                                                                                                    \
            struct SliceHandler slh = {0};                                                                                                                \
            BY_PASS(wrapper_label, slh)                                                                                                                   \
        }

    #define GEN_ATTENTION_DEPENDENCY_WRAPPER_FUNC(wrapper_label)                                                                                          \
        GEN_DEPENDENCY_WRAPPER_HEADER(wrapper_label) {                                                                                                    \
            struct SliceHandler slh = {0};                                                                                                                \
            BY_PASS(wrapper_label, slh)                                                                                                                   \
        }
#endif

// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(ENCODER_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(ENCODER_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(LAYERNORM_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(LAYERNORM_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(MATMUL_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(MATMUL_INPUT_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(GELU_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(GELU_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(RESIDUAL_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(RESIDUAL_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(SOFTMAX_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(CROSSENTROPY_FORWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(CROSSENTROPY_SOFTMAX_BACKWARD)
// GEN_COMMON_DEPENDENCY_WRAPPER_FUNC(ATTENTION_BACKWARD_KV)

// GEN_ATTENTION_DEPENDENCY_WRAPPER_FUNC(ATTENTION_FORWARD)
// GEN_ATTENTION_DEPENDENCY_WRAPPER_FUNC(ATTENTION_BACKWARD_Q)

#endif