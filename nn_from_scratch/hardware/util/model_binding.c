#include "model_binding.h"
#include <stdlib.h>
#include <stdio.h>
/*
    Model binding is excluded from memory tracking.

*/
/* binds values for a model to a modelPtr */
void setModel(ModelPtr *model, int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
              float **layers_biases, enum ActivationType *layers_activation)
{
    model->layers_biases = layers_biases;
    model->layers_size = layers_size;
    model->layers_weights = layers_weights;
    model->n_layers = n_layers;
    model->input_size = input_size;
    model->layers_activation = layers_activation;
    model->output_size = output_size;
}

/* Create ModelPtr and sets the model*/
ModelPtr *createAndSetModel(int n_layers, int input_size, int output_size, int *layers_size, float **layers_weights,
                            float **layers_biases, enum ActivationType *layers_activation)
{
    ModelPtr *model = (ModelPtr *)malloc(sizeof(ModelPtr));
    setModel(model, n_layers, input_size, output_size, layers_size, layers_weights, layers_biases, layers_activation);

    return model;
}

/* Frees a model, should especially be used when tracking memory. As the model binding is excluded from memory tracking */
void freeModel(ModelPtr *model)
{
    free(model);
}


