#include <stdlib.h>
#include "config.h"
#include "model_gradients.h"
Gradients *allocate_gradients(Model *model)
{
    Gradients *gradients = (Gradients *)malloc(sizeof(Gradients));

    gradients->biases = (float **)malloc(model->n_layers * sizeof(float *));
    gradients->weights = (float **)malloc(model->n_layers * sizeof(float *));
    gradients->net_inputs = (float **)malloc(model->n_layers * sizeof(float *));

    for (int i = 0; i < model->n_layers; i++)
    {
        gradients->biases[i] = (float *)calloc(model->layers_size[i], sizeof(float));
        gradients->net_inputs[i] = (float *)malloc(model->layers_size[i] * sizeof(float));
        // size of weights:
        if (i == 0)
        {
            gradients->weights[i] = (float *)calloc(model->layers_size[i] * model->input_size, sizeof(float));
        }
        else
        {
            gradients->weights[i] = (float *)calloc(model->layers_size[i] * model->layers_size[i - 1], sizeof(float));
        }
        // neurons will be set when forward propagating
    }

    return gradients;
}
/* Free's allocated memory for gradient*/
void free_gradients(Gradients *gradients, Model *model)
{

    for (int i = 0; i < model->n_layers; i++)
    {
        free(gradients->biases[i]);
        free(gradients->weights[i]);
        free(gradients->net_inputs[i]);
    }
    free(gradients->biases);
    free(gradients->weights);
    free(gradients->net_inputs);
    free(gradients);
}

PartialGradients *allocate_partial_gradients(Model *model, int target_layer, int n_neurons)
{
    PartialGradients *gradients = (PartialGradients *)malloc(sizeof(PartialGradients));

    gradients->biases = (float *)calloc(model->layers_size[target_layer], sizeof(float)); // biases updated for the targets and for prev layer
    gradients->deriv_activations = (uint8_t **)malloc((model->n_layers - target_layer) * sizeof(uint8_t *));
    gradients->weights = (float *)calloc(n_neurons * model->layers_size[target_layer], sizeof(float));
    gradients->net_input = (float *)malloc(n_neurons * sizeof(float));

    // neurons will be set when forward propagating
    for (int i = 0; i < model->n_layers - target_layer; i++)
    {
        gradients->deriv_activations[i] = (uint8_t *)malloc(model->layers_size[i + target_layer] * sizeof(uint8_t));
    }

    return gradients;
}

void free_partial_gradients(PartialGradients *gradients, Model *model, int target_layer)
{
    free(gradients->biases);
    free(gradients->weights);
    for (int i = 0; i < model->n_layers - target_layer; i++)
    {
        free(gradients->deriv_activations[i]);
    }
    free(gradients->deriv_activations);
    free(gradients->net_input);
    free(gradients);
}