#ifndef __OMPSS_2_WRAPPERS_H__
#define __OMPSS_2_WRAPPERS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ompss-2_settings.h"

#define DIMS_MAX 2
struct SliceHandler {
    int b_start;
    int b_end;
    int t_start;
    int t_end;
    int dims_starts[DIMS_MAX];
    int dims_stops[DIMS_MAX];
};

// Utils

#define CONC_(A, B) A##_##B // Concatenates 2 macros

// Generates the header of a wrapper function
#define GEN_DEPENDENCY_WRAPPER_HEADER(wrapper_label)                                                                                                      \
    void CONC_(wrapper_label, wrapper_name)(void (*func)(CONC_(wrapper_label, header), struct SliceHandler slh), CONC_(wrapper_label, header),            \
                                            int *dep_handler)

// Wrapper declaration

GEN_DEPENDENCY_WRAPPER_HEADER(ENCODER_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(ENCODER_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(LAYERNORM_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(LAYERNORM_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(MATMUL_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(MATMUL_INPUT_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(ATTENTION_BACKWARD_Q);
GEN_DEPENDENCY_WRAPPER_HEADER(ATTENTION_BACKWARD_KV);
GEN_DEPENDENCY_WRAPPER_HEADER(GELU_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(GELU_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(RESIDUAL_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(RESIDUAL_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(SOFTMAX_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(CROSSENTROPY_FORWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(CROSSENTROPY_SOFTMAX_BACKWARD);
GEN_DEPENDENCY_WRAPPER_HEADER(ATTENTION_FORWARD);

void MATMUL_PARAMS_BACKWARD_wrapper_name(void (*func)(MATMUL_PARAMS_BACKWARD_header, struct SliceHandler slh), MATMUL_PARAMS_BACKWARD_header,
                                         int *dep_handler, float *start_dep);

// Dependency handler

int *dep_init(int B, int T, int dep_size_B, int dep_size_T);
void dep_finish(int *dep_handler);
void dep_set_slice_shape(int *dep_handler, int slice_size_B, int slice_size_T);
void dep_reset(int *dep_handler);


#ifdef __cplusplus
}
#endif


#endif // __OMPSS_2_WRAPPERS_H__