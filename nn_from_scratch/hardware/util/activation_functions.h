#ifndef ACTIVATION_TYPE_H
#define ACTIVATION_TYPE_H
enum ActivationType
{
    LINEAR,
    RELU
};

typedef float (*ActivationFunc)(float);

float relu(float x);
float linear(float x);
float relu_deriv(float x);
float linear_deriv(float x);
ActivationFunc get_activation_func(enum ActivationType activationType);
ActivationFunc get_activation_func_deriv(enum ActivationType activationType);
#endif
/*
#define linear(x) (x)
#define relu(x) ((x > 0) ? x : 0)
#define linear_deriv(x) (1)
#define relu_deriv(x) ((x > 0) ? 1 : 0)
// Define macros for each activation function
#define ACTIVATION_MACRO_LIST \
    X(LINEAR, linear)         \
    X(RELU, relu)

#define ACTIVATION_DERIV_MACRO_LIST \
    X(LINEAR, linear_deriv)         \
    X(RELU, relu_deriv)
*/
