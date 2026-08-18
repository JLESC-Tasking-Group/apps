#ifndef __OMPSS_2_SETTINGS__
#define __OMPSS_2_SETTINGS__

// #undef ENABLE_TASKITER

// Constants

#define MB_SUBSIZE 1
#define MT_SUBSIZE 8
#define MATMUL_LOOP_UNROLL MT_SUBSIZE

// Wrappers definitions 

// All of the values here have been optimized for an application using 1 socket

#define ENCODER_FORWARD_header float *out, int *inp, float *wte, float *wpe, int B, int T, int C // Header of the function, without SliceHandler
#define ENCODER_FORWARD_params_name out, inp, wte, wpe, B, T, C                                  // Parameters name of the function, without the types
#define ENCODER_FORWARD_trace_label "Encoder Forward"                                            // Label to be used in the paraver trace
#define ENCODER_FORWARD_wrapper_name dep_encoder_forward                                         // Wrapper function to call to use the dependencies
#define ENCODER_FORWARD_layer_deps concurrent(*out, *inp, *wte, *wpe)                            // Dependencies for the layer buffers
#define ENCODER_FORWARD_grainsize 2                                                              // Equivalent to grainsize clause from oss pragmas
#define ENCODER_FORWARD_collapse_depth 2                                                         // Equivalent to collapse clause from oss pragmas
#define ENCODER_FORWARD_collapse_length sliced_B, sliced_T                                       // Lengths before slicing of the collapsed loops

#define ENCODER_BACKWARD_header float *dwte, float *dwpe, float *dout, int *inp, int B, int T, int C
#define ENCODER_BACKWARD_params_name dwte, dwpe, dout, inp, B, T, C
#define ENCODER_BACKWARD_trace_label "Encoder Backward"
#define ENCODER_BACKWARD_wrapper_name dep_encoder_backward
#define ENCODER_BACKWARD_layer_deps concurrent(*dwte, *dwpe, *dout, *inp)
#define ENCODER_BACKWARD_grainsize 1
#define ENCODER_BACKWARD_collapse_depth 2
#define ENCODER_BACKWARD_collapse_length sliced_B, sliced_T

#define LAYERNORM_FORWARD_header float *out, float *mean, float *rstd, float *inp, float *weight, float *bias, int B, int T, int C
#define LAYERNORM_FORWARD_params_name out, mean, rstd, inp, weight, bias, B, T, C
#define LAYERNORM_FORWARD_trace_label "Layernorm Forward"
#define LAYERNORM_FORWARD_wrapper_name dep_layernorm_forward
#define LAYERNORM_FORWARD_layer_deps concurrent(*out, *mean, *rstd, *inp, *weight, *bias)
#define LAYERNORM_FORWARD_grainsize 4
#define LAYERNORM_FORWARD_collapse_depth 2
#define LAYERNORM_FORWARD_collapse_length sliced_B, sliced_T

#define LAYERNORM_BACKWARD_header                                                                                                                         \
    float *dinp, float *dweight, float *dbias, float *dout, float *inp, float *weight, float *mean, float *rstd, int B, int T, int C
#define LAYERNORM_BACKWARD_params_name dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C
#define LAYERNORM_BACKWARD_trace_label "Layernorm Backward"
#define LAYERNORM_BACKWARD_wrapper_name dep_layernorm_backward
#define LAYERNORM_BACKWARD_layer_deps concurrent(*dinp, *dweight, *dbias, *dout, *inp, *weight, *mean, *rstd)
#define LAYERNORM_BACKWARD_grainsize 8
#define LAYERNORM_BACKWARD_collapse_depth 2
#define LAYERNORM_BACKWARD_collapse_length sliced_B, sliced_T

#define MATMUL_FORWARD_header float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC
#define MATMUL_FORWARD_params_name out, inp, weight, bias, B, T, C, OC
#define MATMUL_FORWARD_trace_label "Matmul Forward"
#define MATMUL_FORWARD_wrapper_name dep_matmul_forward
#define MATMUL_FORWARD_layer_deps concurrent(*out, *inp, *weight, *bias)
#define MATMUL_FORWARD_grainsize OC / 6
#define MATMUL_FORWARD_collapse_depth 1
#define MATMUL_FORWARD_collapse_length OC

#define MATMUL_INPUT_BACKWARD_header float *dinp, const float *dout, const float *weight, int B, int T, int C, int OC
#define MATMUL_INPUT_BACKWARD_params_name dinp, dout, weight, B, T, C, OC
#define MATMUL_INPUT_BACKWARD_trace_label "Matmul Input Backward"
#define MATMUL_INPUT_BACKWARD_wrapper_name dep_matmul_input_backward
#define MATMUL_INPUT_BACKWARD_layer_deps concurrent(*dinp, *dout, *weight)
#define MATMUL_INPUT_BACKWARD_grainsize 2
#define MATMUL_INPUT_BACKWARD_collapse_depth 2
#define MATMUL_INPUT_BACKWARD_collapse_length sliced_B, sliced_T

#define MATMUL_PARAMS_BACKWARD_header float *dweight, float *dbias, const float *dout, const float *inp, int B, int T, int C, int OC
#define MATMUL_PARAMS_BACKWARD_params_name dweight, dbias, dout, inp, B, T, C, OC
#define MATMUL_PARAMS_BACKWARD_trace_label "Matmul Params Backward"
#define MATMUL_PARAMS_BACKWARD_wrapper_name dep_matmul_params_backward
#define MATMUL_PARAMS_BACKWARD_layer_deps concurrent(*dweight, *dbias, *dout, *inp) in(*start_dep)
#define MATMUL_PARAMS_BACKWARD_grainsize OC / 10
#define MATMUL_PARAMS_BACKWARD_collapse_depth 1
#define MATMUL_PARAMS_BACKWARD_collapse_length OC

#define ATTENTION_FORWARD_header float *out, float *preatt, float *att, float *inp, int B, int T, int C, int NH
#define ATTENTION_FORWARD_params_name out, preatt, att, inp, B, T, C, NH
#define ATTENTION_FORWARD_trace_label "Attention Forward"
#define ATTENTION_FORWARD_wrapper_name dep_attention_forward
#define ATTENTION_FORWARD_layer_deps concurrent(*out, *preatt, *att, *inp)
#define ATTENTION_FORWARD_grainsize 2
#define ATTENTION_FORWARD_collapse_depth 2                   // Max is 3
#define ATTENTION_FORWARD_collapse_length sliced_B, sliced_T //, NH

#define ATTENTION_BACKWARD_Q_header float *dinp, float *dpreatt, float *datt, float *dout, float *inp, float *att, int B, int T, int C, int NH
#define ATTENTION_BACKWARD_Q_params_name dinp, dpreatt, datt, dout, inp, att, B, T, C, NH
#define ATTENTION_BACKWARD_Q_trace_label "Attention Backward Query"
#define ATTENTION_BACKWARD_Q_wrapper_name dep_attention_backward_Q
#define ATTENTION_BACKWARD_Q_layer_deps concurrent(*dinp, *dpreatt, *datt, *dout, *inp, *att)
#define ATTENTION_BACKWARD_Q_grainsize 4
#define ATTENTION_BACKWARD_Q_collapse_depth 2                   // Max is 3
#define ATTENTION_BACKWARD_Q_collapse_length sliced_B, sliced_T //, NH

#define ATTENTION_BACKWARD_KV_header float *dinp, float *dpreatt, float *dout, float *inp, float *att, int B, int T, int C, int NH
#define ATTENTION_BACKWARD_KV_params_name dinp, dpreatt, dout, inp, att, B, T, C, NH
#define ATTENTION_BACKWARD_KV_trace_label "Attention Backward Key-Value"
#define ATTENTION_BACKWARD_KV_wrapper_name dep_attention_backward_KV
#define ATTENTION_BACKWARD_KV_layer_deps concurrent(*dinp, *dpreatt, *dout, *inp, *att)
#define ATTENTION_BACKWARD_KV_grainsize 8
#define ATTENTION_BACKWARD_KV_collapse_depth 2                   // Max is 4
#define ATTENTION_BACKWARD_KV_collapse_length sliced_B, sliced_T //, NH, C / NH

#define GELU_FORWARD_header float *out, float *inp, int B, int T, int C
#define GELU_FORWARD_params_name out, inp, B, T, C
#define GELU_FORWARD_trace_label "Gelu Forward"
#define GELU_FORWARD_wrapper_name dep_gelu_forward
#define GELU_FORWARD_layer_deps concurrent(*out, *inp)
#define GELU_FORWARD_grainsize 2
#define GELU_FORWARD_collapse_depth 1
#define GELU_FORWARD_collapse_length sliced_T

#define GELU_BACKWARD_header float *dinp, float *inp, float *dout, int B, int T, int C
#define GELU_BACKWARD_params_name dinp, inp, dout, B, T, C
#define GELU_BACKWARD_trace_label "Gelu Backward"
#define GELU_BACKWARD_wrapper_name dep_gelu_backward
#define GELU_BACKWARD_layer_deps concurrent(*dinp, *inp, *dout)
#define GELU_BACKWARD_grainsize 1
#define GELU_BACKWARD_collapse_depth 1
#define GELU_BACKWARD_collapse_length sliced_T

#define RESIDUAL_FORWARD_header float *out, float *inp1, float *inp2, int B, int T, int C
#define RESIDUAL_FORWARD_params_name out, inp1, inp2, B, T, C
#define RESIDUAL_FORWARD_trace_label "Residual Forward"
#define RESIDUAL_FORWARD_wrapper_name dep_residual_forward
#define RESIDUAL_FORWARD_layer_deps concurrent(*out, *inp1, *inp2)
#define RESIDUAL_FORWARD_grainsize 1
#define RESIDUAL_FORWARD_collapse_depth 1
#define RESIDUAL_FORWARD_collapse_length sliced_B

#define RESIDUAL_BACKWARD_header float *dinp1, float *dinp2, float *dout, int B, int T, int C
#define RESIDUAL_BACKWARD_params_name dinp1, dinp2, dout, B, T, C
#define RESIDUAL_BACKWARD_trace_label "Residual Backward"
#define RESIDUAL_BACKWARD_wrapper_name dep_residual_backward
#define RESIDUAL_BACKWARD_layer_deps concurrent(*dinp1, *dinp2, *dout)
#define RESIDUAL_BACKWARD_grainsize 1
#define RESIDUAL_BACKWARD_collapse_depth 1
#define RESIDUAL_BACKWARD_collapse_length sliced_B

#define SOFTMAX_FORWARD_header float *probs, float *logits, int B, int T, int V, int Vp
#define SOFTMAX_FORWARD_params_name probs, logits, B, T, V, Vp
#define SOFTMAX_FORWARD_trace_label "Softmax Forward"
#define SOFTMAX_FORWARD_wrapper_name dep_softmax_forward
#define SOFTMAX_FORWARD_layer_deps concurrent(*probs, *logits)
#define SOFTMAX_FORWARD_grainsize 2
#define SOFTMAX_FORWARD_collapse_depth 2
#define SOFTMAX_FORWARD_collapse_length sliced_B, sliced_T

#define CROSSENTROPY_FORWARD_header float *losses, float *probs, int *targets, int B, int T, int Vp
#define CROSSENTROPY_FORWARD_params_name losses, probs, targets, B, T, Vp
#define CROSSENTROPY_FORWARD_trace_label "Crossentropy Forward"
#define CROSSENTROPY_FORWARD_wrapper_name dep_crossentropy_forward
#define CROSSENTROPY_FORWARD_layer_deps concurrent(*losses, *probs, *targets)
#define CROSSENTROPY_FORWARD_grainsize 4
#define CROSSENTROPY_FORWARD_collapse_depth 2
#define CROSSENTROPY_FORWARD_collapse_length sliced_B, sliced_T

#define CROSSENTROPY_SOFTMAX_BACKWARD_header float *dlogits, float *dlosses, float *probs, int *targets, int B, int T, int V, int Vp
#define CROSSENTROPY_SOFTMAX_BACKWARD_params_name dlogits, dlosses, probs, targets, B, T, V, Vp
#define CROSSENTROPY_SOFTMAX_BACKWARD_trace_label "Crossentropy Softmax Backward"
#define CROSSENTROPY_SOFTMAX_BACKWARD_wrapper_name dep_crossentropy_softmax_backward
#define CROSSENTROPY_SOFTMAX_BACKWARD_layer_deps concurrent(*dlogits, *dlosses, *probs, *targets)
#define CROSSENTROPY_SOFTMAX_BACKWARD_grainsize 4
#define CROSSENTROPY_SOFTMAX_BACKWARD_collapse_depth 2
#define CROSSENTROPY_SOFTMAX_BACKWARD_collapse_length sliced_B, sliced_T

#endif // __OMPSS_2_SETTINGS__