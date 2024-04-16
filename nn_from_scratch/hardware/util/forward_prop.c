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
                    int input_size, int output_size, ActivationFunc activation_func)
{
    float *output = (float *)malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
    {
        // get sum
        float sum = 0;
        for (int j = 0; j < input_size; j++)
        {
            // indexing for correct weight
            sum += activation_func(input[j],0) * weights[i + j * output_size];
        }
        // add bias
        sum += biases[i];
        output[i] = sum;

    }
    return output;
}


