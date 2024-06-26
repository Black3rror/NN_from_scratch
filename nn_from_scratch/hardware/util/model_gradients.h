
#ifndef MODEL_GRADIENTS_H
#define MODEL_GRADIENTS_H
#include "activation_functions.h"
#include <stdint.h>
#include "model_binding.h"
typedef struct
{
    float **weights;
    float **biases;
    float **net_inputs;
    uint8_t **deriv_activations;
} Gradients;

typedef struct
{
    float *weights;
    float *biases;
    float *net_input;
    uint8_t **deriv_activations;
} PartialGradients;

Gradients *allocate_gradients(Model *model);
void free_gradients(Gradients *gradients, Model *model);

PartialGradients *allocate_partial_gradients(Model *model, int target_layer, int n_neurons);
void free_partial_gradients(PartialGradients *gradients, Model *model, int target_layer);

#endif