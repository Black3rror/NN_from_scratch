#include "util/forward_prop.h"
#include "util/activation_functions.h"
#include "util/back_prop.h"
#include "util/loss_functions.h"

typedef struct
{
    float **weights;
    float **biases;
    float **neurons;
} Gradients;

typedef struct
{
    int n_layers;
    int *layers_size;
    float **layers_weights;
    float **layers_biases;
    ActivationFunc *layers_activation;
} ModelPtr;

void setModel(ModelPtr *model, int n_layers, int *layers_size, float **layers_weights,
              float **layers_biases);