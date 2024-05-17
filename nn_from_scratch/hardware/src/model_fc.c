#include <string.h>
#include <stdlib.h>
#include "model_fc.h"
#include "../util/forward_prop.h"
#include "../util/activation_functions.h"
#include "../util/back_prop.h"
#include "../util/loss_functions.h"
#include "../util/config.h"
#include <stdio.h>
/* Fully connected model functionality, does forward propagation,
    backpropagation with training to calculate gradients into gradient structure.
    @return nothing.
*/
void fc_calc_gradients(Model *model, float *input, float *actual, Gradients *gradients)
{
    float *curr_in = input;
    int size = model->input_size;
    ActivationFunc func = &linear;             // input activation func is set to linear
    ActivationFunc func_deriv = &linear_deriv; // input activation func is set to linear

    // forward propagate through each layer
    for (int i = 0; i < model->n_layers; i++)
    {

        curr_in = fc_forward_prop_t(curr_in, size, gradients->net_inputs[i],
                                    model->layers_size[i], model->layers_weights[i], model->layers_biases[i], func);
        size = model->layers_size[i];
        func = get_activation_func(model->layers_activation[i]);
        func_deriv = get_activation_func_deriv(model->layers_activation[i]);
    } // if not training return output (input from prev loop with activation func applied)
    for (int i = 0; i < model->layers_size[model->n_layers - 1]; i++)
    {
        curr_in[i] = func(curr_in[i]);
    }
    // if training flag do backpropagate
    // start getting error derivative and overwrite last layer net_inputs with gradients
    float loss_deriv = MSE_derivative(curr_in, actual, model->layers_size[model->n_layers - 1]); // last layer size is output size
    for (int i = 0; i < model->layers_size[model->n_layers - 1]; i++)
    {

        gradients->net_inputs[model->n_layers - 1][i] = loss_deriv * func_deriv(gradients->net_inputs[model->n_layers - 1][i]);
    }
    // perform backprop
    for (int i = model->n_layers - 1; i > 0; i--)
    {
        fc_back_prop(gradients->net_inputs[i], gradients->net_inputs[i - 1], model->layers_weights[i],
                     model->layers_size[i], model->layers_size[i - 1], get_activation_func(model->layers_activation[i - 1]), get_activation_func_deriv(model->layers_activation[i - 1]), gradients->weights[i], gradients->biases[i]);
    }

    // edge case for input to first layer
    curr_in = input;
    fc_back_prop(gradients->net_inputs[0], curr_in, model->layers_weights[0],
                 model->layers_size[0], model->input_size, linear, linear_deriv, gradients->weights[0], gradients->biases[0]);
    return;
}

/* Applies gradients for a fully connected layer*/
void fc_apply_gradient(Model *model, int layer, int layer_size, int prev_layer_size, Gradients *gradients)
{
    for (int i = 0; i < layer_size; i++)
    {
        // get sum
        printf("bias gradient %d = %f \n", i, gradients->biases[layer][i]);
        model->layers_biases[layer][i] -= LEARNING_RATE * (gradients->biases[layer][i] / BATCH_SIZE);
        for (int j = 0; j < prev_layer_size; j++)
        {
            // indexing for correct weight
            printf("weight gradient %d = %f \n", i + j * layer_size, gradients->weights[layer][i + j * layer_size]);
            model->layers_weights[layer][i + j * layer_size] -= LEARNING_RATE * (gradients->weights[layer][i + j * layer_size] / BATCH_SIZE);
        }
    }
    printf("\n");
}

/* train fully connected layer for batch_size amount of samples*/
void fc_model_train(Model *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size])
{
    /* create gradient struct*/
    Gradients *gradients = (Gradients *)allocate_gradients(model);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        fc_calc_gradients(model, samples_x[i], samples_y[i], gradients);
    }
    // loop over the gradients and apply step
    int size = model->input_size;
    for (int i = 0; i < model->n_layers; i++)
    {
        fc_apply_gradient(model, i, model->layers_size[i], size, gradients);
        size = model->layers_size[i];
    }

    free_gradients(gradients, model);
}

/* Function to calculated fully-connected model output */
float *fc_model_predict(Model *model, float *input)
{

    int size = model->input_size;
    ActivationFunc func = get_activation_func(model->layers_activation[0]);
    // forward propagate through each layer
    float *output = fc_forward_prop(input, model->layers_weights[0], model->layers_biases[0],
                                    size, model->layers_size[0], func);
    input = output;
    size = model->layers_size[0];
    func = get_activation_func(model->layers_activation[1]);

    for (int i = 1; i < model->n_layers; i++)
    {

        func = get_activation_func(model->layers_activation[i]);
        output = fc_forward_prop(input, model->layers_weights[i], model->layers_biases[i],
                                 size, model->layers_size[i], func);
        free(input);
        input = output;
        size = model->layers_size[i];
    }
    return output;
}
