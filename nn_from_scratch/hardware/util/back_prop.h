/* Header file for b_prop */

#ifndef ACTIVATION_FUNCTION_TYPEDEF
#define ACTIVATION_FUNCTION_TYPEDEF
#include <stdint.h>
typedef float (*ActivationFunc)(float, int);
#endif

void fc_back_prop(float *input_gradient, float *output_gradient, float *weights,
                  int input_size, int output_size, ActivationFunc activation_func_,
                  float *gradient_weights, float *gradient_biases);


float *light_fc_back_prop(float *input_gradient, float *weights,
                          int input_size, int output_layer_size, uint8_t *deriv_activation_val);


void specific_fc_back_prop(float *input_gradient, float *net_input,
                           int input_size, ActivationFunc activation_func,
                           float *gradient_weights, float *gradient_biases, int n_neurons, int offset);