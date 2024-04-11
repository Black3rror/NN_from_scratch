
#include "model_fc.h"
// std
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// for print
#include <stdio.h>

#include "data/eqcheck_data.h"

/*
    if training flag is set and gradient is not NULL,
              gradients will be updated for training

*/

float *fc_model(float *input_, int size, ModelPtr *model, int training_flag, float *predicted, Gradients *gradients)
{
    // forward propagate through each layer
    float *input = malloc(size * sizeof(float));
    memcpy(input, input_, size * sizeof(float)); // copy input, so the initial input does not get free'd
    for (int i = 0; i < model->n_layers; i++)
    {
        float *output = fc_forward_prop(input, model->layers_weights[i], model->layers_biases[i],
                                        size, model->layers_size[i], model->layers_activation[i], NULL);

        if (!training_flag)
        {
            free(input);
        }

        input = output;
        size = model->layers_size[i];
    }
    // if not training return output (input from prev loop)
    if (!training_flag)
    {
        return input;
    }
    // if training flag do backpropagate
    float loss_deriv = MSE_Derivative(predicted, input, model->n_layers);
    for (int i = model->n_layers - 1; i > 0; i--)
    {
        fc_back_prop(gradients->neurons[i], gradients->neurons[i - 1], model->layers_weights[i], model->layers_biases[i],
                     model->layers_size[i], model->layers_size[i - 1], model->layers_activation[i], gradients->weights[i], gradients->biases[i]);
    }
    return input;
}

/* train fully connected layer for x amount of samples*/
void fc_model_training(ModelPtr *model, float **samples_x, float **samples_y, int input_size, int n_samples, float step_rate)
{
    /* create gradient struct*/
    Gradients gradients;
    for (int i = 0; i < n_samples; i++)
    {
        float *output = fc_model(samples_x[i], input_size, model, 1, samples_y[i], &gradients);
        free(output);
    }
    // aver the gradients and apply step
}
float *fc_model_predict(ModelPtr *model, float *input, int input_size)
{
    float *output = fc_model(input, input_size, model, 0, NULL, NULL);
    return output;
}
void setModel(ModelPtr *model, int n_layers, int *layers_size, float **layers_weights,
              float **layers_biases)
{
    model->layers_biases = layers_biases;
    model->layers_size = layers_size;
    model->layers_weights = layers_weights;
    model->n_layers = n_layers;
    // malloc actionfunc list
    ActivationFunc *ActivationFuncLayers = (ActivationFunc *)malloc(sizeof(ActivationFunc) * n_layers);
}

void registerActivationFuncToLayer(ModelPtr *model, ActivationFunc activationFunc, int layer)
{
    model->layers_activation[layer] = activationFunc;
}