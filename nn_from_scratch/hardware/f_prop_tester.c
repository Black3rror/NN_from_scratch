#include <stdint.h>
#include "f_prop.h"
#include "eqcheck/eqcheck_data.h"
#include <stdlib.h>
#include <string.h>
// for print
#include <stdio.h>
#include <math.h>

int main()
{
    // Test data input to output from eqcheck

    for (int i = 0; i < N_SAMPLES; i++)
    {
        float *input = eqcheck_samples_x[i];
        float *output = f_prop(input, INPUT_SIZE, N_LAYERS);
        // check output
        printf("input: %f \n", *input);
        float tolerance = 0.0001;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            if (fabs(output[j] - eqcheck_samples_y[i][j]) < tolerance)
            {
                printf("MATCH\n");
            }
            else
            {
                printf("output: %f, did not match eqcheck: %f \n", output[j], eqcheck_samples_y[i][j]);
            }
        }
    }
    printf("done.");
    return 0;
}