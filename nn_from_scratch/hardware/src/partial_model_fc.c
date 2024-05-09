
#include <stdlib.h>
#include <string.h>
#include "../util/forward_prop.h"
#include "../util/activation_functions.h"
#include "../util/back_prop.h"
#include "../util/loss_functions.h"
#include "partial_model_fc.h"
#include "../util/config.h"
#include <stdio.h>

/*
    Calculates partial gradients, meaning it will store the gradients for the target layer and activation for the previous neurons
    given n_weights and offset.
*/
void partial_calc_gradients(float *input, ModelPtr *model, int target_layer, int n_weights, int offset, float *actual, PartialGradients *gradients)
{

    float *curr_in = input;
    float *output;
    int size = model->input_size;
    ActivationFunc func = &linear; // input activation func is set to linear
    size = model->layers_size[0];

    for (int i = 0; i < model->n_layers; i++)
    {
        output = (float *)malloc(model->layers_size[i] * sizeof(float)); // allocate output

        /* forward propagate, store needed data otherwise free */
        output = fc_forward_prop_t(curr_in, size, output, model->layers_size[i], model->layers_weights[i], model->layers_biases[i], func);

        if (i == target_layer) // store neuron if at target layer
        {
            memcpy(gradients->net_input, curr_in + offset, n_weights * sizeof(float));
            if (target_layer != 0)
            {
                free(curr_in);
            }
        }
        else if (i != 0 && (i < target_layer || i > target_layer)) // free if below target layer or larget than target, since not needed
        {
            free(curr_in);
        }

        if (i >= target_layer)
        { // else only store derivative of the input
            for (int j = 0; j < model->layers_size[i]; j++)
            {
                gradients->deriv_activations[i - target_layer][j] = (uint8_t)func(output[j], 1);
            }
        }
        curr_in = output;
        size = model->layers_size[i];
        func = getActivationFunc(model->layers_activation[i]);
    }
    float loss_deriv = MSE_Derivative(curr_in, actual, model->layers_size[model->n_layers - 1]);

    /* use new backprop until target layer the use normal backprop for that layer only.*/

    for (int i = 0; i < model->layers_size[model->n_layers - 1]; i++)
    {
        curr_in[i] = loss_deriv * func(curr_in[i], 1);
    }
    // perform packprop using the backprop that uses the stored derivative activation values until target layer
    for (int i = model->n_layers - 1; i > target_layer; i--)
    {
        float *output = light_fc_back_prop(curr_in, model->layers_weights[i], model->layers_size[i],
                                           model->layers_size[i - 1], gradients->deriv_activations[i - target_layer - 1]);
        free(curr_in);
        curr_in = output;
    }
    // Apply last backprop, using 'normal backprop' to calculate the gradient to target weights.
    func = &linear;
    if (target_layer != 0)
    {
        func = getActivationFunc(model->layers_activation[target_layer - 1]);
    }
    specific_fc_back_prop(curr_in, gradients->net_input, model->layers_size[target_layer],
                          func, gradients->weights, gradients->biases, n_weights, offset);

    free(curr_in);

    return;
}

/* Apply gradients to a layer, given specific neurons*/
void fc_apply_specific_gradients(ModelPtr *model, int layer, int layer_size, int prev_layer_size, int n_weights, int offset, PartialGradients *gradients)
{
    for (int i = 0; i < layer_size; i++)
    {
        model->layers_biases[layer][i] -= LEARNING_RATE * (gradients->biases[i] / BATCH_SIZE);
        for (int j = 0; j < n_weights; j++)
        {
            // indexing for correct weight
            model->layers_weights[layer][i + (j + offset) * layer_size] -= LEARNING_RATE * (gradients->weights[i + j * layer_size] / BATCH_SIZE);
        }
    }
}

/* train a part of a layer - stated by target layer, the number of weights and the offset
    this will result in each given neurons incomming weight being trained
    etc. n_weights = 1 and offset =1, will result in each neurons second weight being trained
 */
void fc_model_train_partial_layer(ModelPtr *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                                  int target_layer, int n_weights, int offset)
{
    if ((target_layer == 0 && n_weights != 1 && offset != 0) || target_layer < 0 || offset < 0 || n_weights < 1)
    {
        printf("Invalid arguments for partial layer training! \n");
        return;
    }
    else if (n_weights + offset > model->layers_size[target_layer - 1])
    {
        printf("Invalid arguments for partial layer training! \n");
        return;
    }

    PartialGradients *gradients = (PartialGradients *)allocate_partial_gradients(model, target_layer, n_weights);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        partial_calc_gradients(samples_x[BATCH_SIZE + i], model, target_layer, n_weights, offset, samples_y[BATCH_SIZE + i], gradients);
    }

    // apply the calculated gradient to the specific layer
    if (target_layer == 0)
    {
        fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], model->input_size, n_weights, offset, gradients);
    }
    else
    {
        fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], model->layers_size[target_layer - 1], n_weights, offset, gradients);
    }
    free_partial_gradients(gradients, model, target_layer);
}

/* train a specific layer*/
void fc_model_train_layer(ModelPtr *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                          int target_layer)
{

    int offset = 0;
    int n_weights;
    if (target_layer == 0)
    {
        n_weights = 1;
    }
    else
    {
        n_weights = model->layers_size[target_layer - 1];
    }

    PartialGradients *gradients = (PartialGradients *)allocate_partial_gradients(model, target_layer, n_weights);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        partial_calc_gradients(samples_x[BATCH_SIZE + i], model, target_layer, n_weights, offset, samples_y[BATCH_SIZE + i], gradients);
    }

    // apply the calculated gradient to the specific layer
    if (target_layer == 0)
    {
        fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], model->input_size, n_weights, offset, gradients);
    }
    else
    {
        fc_apply_specific_gradients(model, target_layer, model->layers_size[target_layer], model->layers_size[target_layer - 1], n_weights, offset, gradients);
    }
    free_partial_gradients(gradients, model, target_layer);
}