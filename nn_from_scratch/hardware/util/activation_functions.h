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
