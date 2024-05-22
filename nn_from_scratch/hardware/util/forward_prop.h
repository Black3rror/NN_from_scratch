
#include "activation_functions.h"
#include <stdint.h>
extern float *fc_forward_prop(float *input, float *layer_weights, float *layer_biases, int input_size,
                              int output_size, ActivationFunc activation_func);

extern float *fc_forward_prop_t(float *input, int input_size, float *output, int output_size, float *weights, float *biases, ActivationFunc activation_func);
/*
#define FC_FORWARD_PROP_VARIANT(activation_function)               \
    fc_forward_prop_##activation_function(input, weights, biases,  \
                                          input_size, output_size) \
    {                                                              \
        output = malloc(output_size * sizeof(float));              \
        for (int i = 0; i < output_size; i++)                      \
        {                                                          \
            sum = 0;                                               \
            for (int j = 0; j < input_size; j++)                   \
            {                                                      \
                sum += input[j] * weights[i + j * output_size];    \
            }                                                      \
            sum += biases[i];                                      \
            output[i] = activation_func(sum);                      \
        }                                                          \
        return output;                                             \
    }


FC_FORWARD_PROP_VARIANT(relu)*/