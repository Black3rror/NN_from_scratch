/*
    C file for back propagation

    Idea to reuse Neuron map for activations for gradients when backpropagating.
    Overwrite the activations with the gradients when going back.
*/
#include "back_prop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/* Back propagation function for one layer

    @param input: input activations for layer
    @param output: output activations layer (should be previous layer from the input layer)
    @param weights: weights for the layer between input and output
    @param biases: biases for the layer between input and output
    @param input_size: size of input layer
    @param output_size: size of output layer
    @param activation_func: activation function for the layer, flag = 1 for derivative
    @param gradient_weights: pointer to store the gradients for the weights
    @param gradient_biases: pointer to store the gradients for the biases

    @return void, the gradient_weights and gradient_biases will have been updated.
*/
void fc_back_prop(float *input, float *output, float *weights, float *biases,
                  int input_size, int output_size, ActivationFunc activation_func,
                  float *gradient_weights, float *gradient_biases)
{
    for (int i = 0; i < input_size; i++)
    {
        float gradient = activation_func(input[i], 1);
        // add gradient to bias array
        gradient_biases[i] += 1 * gradient;

        // weight gradients
        for (int j = 0; j < output_size; j++)
        {
            // adding gradient weights
            gradient_weights[i + j * input_size] += gradient * activation_func(output[j], 0);
            // summing the gradients, if j==0 overwrite the output activation with the gradient
            if (j == 0)
            {
                output[j] = gradient * weights[i + j * input_size];
            }
            output[j] += gradient * weights[i + j * input_size];
        }
    }
}