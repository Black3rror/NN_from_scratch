#include "activation_functions.h"
#include <stdlib.h>
#include <stdio.h>

/* Activation functions */
float relu(float x, int derivative_flag)
{
    if (derivative_flag)
    {
        return (x > 0) ? 1 : 0;
    }
    else
        return (x > 0) ? x : 0;
}
float linear(float x, int derivative_flag)
{
    if (derivative_flag)
    {
        return 1;
    }
    else
        return x;
}


ActivationFunc getActivationFunc(enum ActivationType activationType)
{
    switch (activationType)
    {
    case RELU:
        return relu;

    case LINEAR:
        return linear;
    default:
        printf("Error unknown activation type... defaulting to LINEAR\n");
        return linear;
    }
}
