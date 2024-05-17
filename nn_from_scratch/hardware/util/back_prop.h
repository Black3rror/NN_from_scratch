#ifndef BACK_PROP_H
#define BACK_PROP_H
#include <stdint.h>
#include "activation_functions.h"

void fc_back_prop(float *input_gradient, float *net_inputs, float *weights,
                  int input_size, int net_inputs_size, ActivationFunc activation_func, ActivationFunc activation_func_deriv,
                  float *gradient_weights, float *gradient_biases);

float *light_fc_back_prop(float *input_gradient, float *weights,
                          int input_size, int output_layer_size, uint8_t *deriv_activation_val);

void specific_fc_back_prop(float *input_gradient, float *net_input,
                           int input_size, ActivationFunc activation_func,
                           float *gradient_weights, float *gradient_biases, int n_neurons);
#endif