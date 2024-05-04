#include "model_binding.h"
#include <stdlib.h>

#include "config.h"
#ifdef TRACK_MEMORY
#include "track_memory.h"
#endif
/* Sets values for a model */
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

/* Frees a model, together with all the layers*/
void destroyModel(ModelPtr *model)
{
    // Free memory for layers
    free(model->layers_size);
    free(model->layers_weights);
    free(model->layers_biases);
    free(model->layers_activation);

    // Free memory for the ModelPtr struct
    free(model);
}