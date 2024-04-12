#include <stdint.h>
#include "forward_prop.h"
#include <stdlib.h>
#include <string.h>
// for print
#include <stdio.h>

/* forward propagation
    @result returns output for layer, without activation function applied
*/
float *fc_forward_prop(float *input, float *weights, float *biases,
                    int input_size, int output_size, ActivationFunc activation_func, float* neurons)
{
    float *output = (float *)malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // indexing for correct weight
            sum += input[j] * weights[i + j * output_size];
        }
        // add bias
        sum += biases[i];
        output[i] = activation_func(sum, 0);

        if (neurons != NULL){
            neurons[i] = sum;
        }
    }
    return output;
}


