/* Header file for b_prop */

#ifndef ACTIVATION_FUNCTION_TYPEDEF
#define ACTIVATION_FUNCTION_TYPEDEF

typedef float (*ActivationFunc)(float, int);
#endif
typedef struct
{
    float **weights;
    float **biases;
    float **neurons;
} Gradients;

void fc_back_prop(float *input_gradient, float *output_gradient, float *weights, float *biases,
                  int input_size, int output_size, ActivationFunc activation_func_,
                  float *gradient_weights, float *gradient_biases);