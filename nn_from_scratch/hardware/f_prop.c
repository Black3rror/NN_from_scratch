#include <stdint.h>
#include "model/convert/c_templates/model.h"

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
        return relu(x);
    default:
        break;
    }
}

/*
  dense function
  Output_layer_x = layer_activation(layer_x_weight * input + layer_x_bias)

  get output after doing dense for each layer
*/

// Dense computation for a single layer
void dense(float *input, float *weights, float *biases, float *output, int input_size, int output_size, enum ActivationType activation)
{
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // Index expression to access correct weight connected to neuron
            // TODO: Need double-check if correct
            sum += input[j] * weights[i * input_size + j];
        }
        sum += biases[i];
        // apply activation
        output[i] = activation_func(activation, sum);
    }
}

// main function for now
int main()
{

    /*  NOTES:
     * create loop to apply dense computation for all layers
     * correctly allocate and free memory - use inplace computation to minimize memory usage
     * test with a small input
     */
}
