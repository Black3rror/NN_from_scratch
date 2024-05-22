#include "activation_functions.h"
#include <stdlib.h>
#include <stdio.h>

/* Activation functions */
float relu(float x)
{
    return (x > 0) ? x : 0;
}
float linear(float x)
{
    return x;
}

float relu_deriv(float x)
{
    return (x > 0) ? 1 : 0;
}
float linear_deriv(__attribute__((unused)) float x)
{
    return 1;
}

ActivationFunc get_activation_func(enum ActivationType activationType)
{
    switch (activationType)
    {
    case RELU:
        return relu;

    case LINEAR:
        return linear;
    default:
        printf("Error unknown activation type: defaulting to LINEAR\n");
        return linear;
    }
}

ActivationFunc get_activation_func_deriv(enum ActivationType activationType)
{
    switch (activationType)
    {
    case RELU:
        return relu_deriv;

    case LINEAR:
        return linear_deriv;
    default:
        printf("Error unknown activation type: defaulting to LINEAR DERIVATIVE\n");
        return linear_deriv;
    }
}