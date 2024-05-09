#include "activation_functions.h"
#include <stdint.h>
extern float *fc_forward_prop(float *input, float *layer_weights, float *layer_biases, int input_size,
                              int output_size, ActivationFunc activation_func);
                              
extern float *fc_forward_prop_t(float *input, int input_size, float* output, int output_size, float *weights, float *biases, ActivationFunc activation_func);