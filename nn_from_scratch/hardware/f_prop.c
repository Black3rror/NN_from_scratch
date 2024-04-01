#include <stdint.h>
#include "f_prop.h"
#include <stdlib.h>
#include <string.h>
// for print
#include <stdio.h>

/* Activation functions */
float relu(float x)
{
    return (x > 0) ? x : 0;
}
float linear(float x)
{
    return x;
}
/* Apply correct activation function*/
float activation_func(enum ActivationType activation, float x)
{
    switch (activation)
    {
    case RELU:
        return relu(x);
    case LINEAR:
        return linear(x);
    default:
        fprintf(stderr, "An error occurred: Unkown activation type \n");
        exit(EXIT_FAILURE); // Indicates failure to the operating system
        break;
    }
}

/* Dense computation for a single layer */
float *dense(float *input, float *weights, float *biases, float *output, int input_size, int output_size, enum ActivationType activation)
{
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // Index expression to access correct weight connected to neuron
            
            //printf("input %f, weight %f \n", input[j],weights[i+j*(output_size)]);

            // indexing for correct wiehgt
            sum+=input[j]* weights[i+j*(output_size)]; 
            //sum += input[j] * weights[(i * input_size) + j];
        }
        // add bias
        sum += biases[i];
        // apply activation
        output[i] = activation_func(activation, sum);
        //printf("Neuron: %d, value: %f \n", i, output[i]);
    }
    return output;
}

/* Helper function to f_prop */
float *f_prop_helper(float *input, int input_size, int i, int N)
{
    if (i == N)
    {
        // stop
        return input; // Input is last layers output (the final result).
    }
    // allocate output
    float *output = (float *)malloc(layers_size[i] * sizeof(float));
    output = dense(input, layers_weights[i], layers_biases[i], output, input_size, layers_size[i], layers_activation[i]);
    //printf("-----\n");
    free(input); // free input and use output as input for the next layer

    return f_prop_helper(output, layers_size[i], i + 1, N);
}
/* forward propagation */
float *f_prop(float *input, int input_size, int N)
{
    // copy input into start value, so input value does not get free'd
    float *start_input = (float *)malloc(sizeof(float) * input_size);
    memcpy(start_input, input, sizeof(float) * input_size);
    float *output = f_prop_helper(start_input, input_size, 0, N);
    return output;
}

