/* Header file for b_prop */

#ifndef ACTIVATION_FUNCTION_TYPEDEF
#define ACTIVATION_FUNCTION_TYPEDEF
typedef float(*ActivationFunc)(float, int);
#endif

extern void fc_back_prop(float *input, float *output, float *weights, float *biases,
                         int input_size, int output_size, ActivationFunc activation_func,
                         float *gradient_weights, float *gradient_biases);