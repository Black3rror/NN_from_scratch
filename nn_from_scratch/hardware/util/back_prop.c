/*
    C file for back propagation

    Idea to reuse Neuron map for activations for gradients when backpropagating.
    Overwrite the activations with the gradients when going back.
*/
#include "back_prop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/* Back propagation function for one layer */
void fc_back_prop(float *input_gradient, float *output, float *weights, float *biases,
                  int input_size, int output_size, ActivationFunc activation_func,
                  float *gradient_weights, float *gradient_biases)
{
    float *temp = calloc(output_size, sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        float gradient = input_gradient[i];
        // add gradient to bias array
        gradient_biases[i] += 1 * gradient;

        // weight gradients
        for (int j = 0; j < output_size; j++)
        {
            // adding gradient weights
            gradient_weights[i + j * input_size] += gradient * activation_func(output[j], 0);
            // gradient neuron

            temp[j] += weights[i + j * input_size] * gradient;
        }
    }

    // gradients for next layer
    for (int j = 0; j < output_size; j++)
    {
        output[j] = temp[j] * activation_func(output[j], 1);
    }


    free(temp);
}