#ifndef MODEL_BINDING_H
#define MODEL_BINDING_H
#include "activation_functions.h"
#include <stdint.h>
typedef struct
{
    int n_layers;
    int input_size;
    int output_size;
    int *layers_size;
    float **layers_weights;
    float **layers_biases;
    enum ActivationType *layers_activation;
} Model;

void setModel(Model *model, int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
              float **layers_biases, enum ActivationType *layers_activation);

Model *createAndSetModel(int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
                         float **layers_biases, enum ActivationType *layers_activation);

void freeModel(Model *model);

#endif
