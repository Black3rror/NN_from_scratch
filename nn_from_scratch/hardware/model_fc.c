
#include <string.h>
#include <stdlib.h>
#include "model_fc.h"
#include "util/config.h"

/* Fully connected model functionality, does forward propagation,
    backpropagation with training flag is set
    @return predicted output or NULL if training.
*/
float *fc_model(float *input_, ModelPtr *model, int training_flag, float *predicted, Gradients *gradients)
{
    float *input;
    if (!training_flag) // copy input, so the initial input does not get free'd
    {
        input = malloc(model->input_size * sizeof(float));
        memcpy(input, input_, model->input_size * sizeof(float));
    }
    else // Otherwise allocating memory is not needed
    {
        input = input_;
    }

    int size = model->input_size;
    ActivationFunc func = &linear; // input activation func is set to linear

    // forward propagate through each layer
    for (int i = 0; i < model->n_layers; i++)
    {

        float *output = fc_forward_prop(input, model->layers_weights[i], model->layers_biases[i],
                                        size, model->layers_size[i], func);
        if (!training_flag)
        {
            free(input);
        }
        else
        {
            gradients->neurons[i] = output;
        }
        input = output;
        size = model->layers_size[i];
        func = getActivationFunc(model->layers_activation[i]);
    } // if not training return output (input from prev loop with activation func applied)
    for (int i = 0; i < model->layers_size[model->n_layers - 1]; i++)
    {
        input[i] = func(input[i], 0);
    }
    if (!training_flag)
    {
        return input;
    }

    // if training flag do backpropagate
    // start getting error derivative and overwrite last layer neurons with gradients
    float loss_deriv = MSE_Derivative(predicted, input, model->layers_size[model->n_layers - 1]); // last layer size is output size

    for (int i = 0; i < model->layers_size[model->n_layers - 1]; i++)
    {

        gradients->neurons[model->n_layers - 1][i] = loss_deriv * func(gradients->neurons[model->n_layers - 1][i], 1);
    }
    // perform backprop
    for (int i = model->n_layers - 1; i > 0; i--)
    {
        fc_back_prop(gradients->neurons[i], gradients->neurons[i - 1], model->layers_weights[i], model->layers_biases[i],
                     model->layers_size[i], model->layers_size[i - 1], getActivationFunc(model->layers_activation[i - 1]), gradients->weights[i], gradients->biases[i]);
    }
    // edge case for input to first layer
    input = malloc(model->input_size * sizeof(float));
    memcpy(input, input_, model->input_size * sizeof(float)); // copy input, so the initial input does not get free'd or modified

    fc_back_prop(gradients->neurons[0], input, model->layers_weights[0], model->layers_biases[0],
                 model->layers_size[0], model->input_size, linear, gradients->weights[0], gradients->biases[0]);
    free(input);
    return NULL;
}

/* Helper function for fc_model_training for applying the change*/
void fc_apply_gradient(ModelPtr *model, int layer, int input_size, int output_size, Gradients *gradients, int n_samples, float step_rate)
{
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        model->layers_biases[layer][i] += step_rate * (gradients->biases[layer][i] / n_samples);
        for (int j = 0; j < input_size; j++)
        {
            // indexing for correct weight
            model->layers_weights[layer][i + j * output_size] += step_rate * (gradients->weights[layer][i + j * output_size] / n_samples);
        }
    }
}
Gradients *allocate_gradients(ModelPtr *model)
{
    Gradients *gradients = (Gradients *)malloc(sizeof(Gradients));

    gradients->biases = (float **)malloc(model->n_layers * sizeof(float *));
    gradients->weights = (float **)malloc(model->n_layers * sizeof(float *));
    gradients->neurons = (float **)malloc(model->n_layers * sizeof(float *));

    for (int i = 0; i < model->n_layers; i++)
    {
        gradients->biases[i] = (float *)calloc(model->layers_size[i], sizeof(float));

        // size of weights:
        if (i == 0)
        {
            gradients->weights[i] = (float *)calloc(model->layers_size[i] * model->input_size, sizeof(float));
        }
        else
        {
            gradients->weights[i] = (float *)calloc(model->layers_size[i] * model->layers_size[i - 1], sizeof(float));
        }
        // neurons will be sit when forward propagating
    }

    return gradients;
}
/* Free's allocated memory for gradient*/
void free_gradients(Gradients *gradients, ModelPtr *model)
{

    for (int i = 0; i < model->n_layers; i++)
    {
        free(gradients->biases[i]);
        free(gradients->weights[i]);
        // free(gradients->neurons[i]);
    }
    free(gradients->biases);
    free(gradients->weights);
    free(gradients->neurons);
    free(gradients);
}
void free_neurons(Gradients *gradients, ModelPtr *model)
{
    for (int i = 0; i < model->n_layers; i++)
    {
        free(gradients->neurons[i]);
    }
}
/* train fully connected layer for x amount of samples*/
void fc_model_training(ModelPtr *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size], int n_samples, float step_rate)
{
    /* create gradient struct*/
    Gradients *gradients = (Gradients *)allocate_gradients(model);

    for (int i = 0; i < n_samples; i++)
    {
        fc_model(samples_x[i], model, 1, samples_y[i], gradients);
        free_neurons(gradients, model);
    }
    // loop over the gradients and apply step
    int size = model->input_size;
    for (int i = 0; i < model->n_layers; i++)
    {
        fc_apply_gradient(model, i, size, model->layers_size[i], gradients, n_samples, step_rate);
        size = model->layers_size[i];
    }
    free_gradients(gradients, model);
}

/* Model prediction */
float *fc_model_predict(ModelPtr *model, float *input)
{
    float *output = fc_model(input, model, 0, NULL, NULL);

    return output;
}
