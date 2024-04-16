

#ifndef ACTIVATION_TYPE_H
#define ACTIVATION_TYPE_H
enum ActivationType
{
    LINEAR,
    RELU
};
#endif

typedef float(*ActivationFunc)(float, int);
float relu(float x, int derivative_flag);
float linear(float x, int derivative_flag);
ActivationFunc getActivationFunc(enum ActivationType activationType);
