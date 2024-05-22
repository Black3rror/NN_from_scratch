

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "include/nn_from_scratch.h"
// #include "model/model.h"
#include "model/simple_model.h"
#include "data/eqcheck_data.h"
#include "data/ft_data.h"
#include "data/true_data.h"
#include "data/simple_data.h"
void compare_true(Model *model)
{

    float sum = 0;
    for (int i = 0; i < FT_N_SAMPLES; i++)
    {
        float *input = ft_samples_x[i];
        float *output = fc_model_predict(model, input);
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            float t = (output[j] - ft_samples_y[i][j]);
            // printf("%f  - %f \n", output[j], ft_samples_y[i][j]);
            t *= t;
            sum += t;
        }
        free(output);
    }
    printf("MSE error: %f \n", sum / FT_N_SAMPLES);
}

void tester(Model *model)
{
    /* First test error for current model */

    compare_true(model);
    printf("Beginning training.. \n");
    // batch size training

    fc_model_train(model, ft_samples_x, ft_samples_y);

    printf("After training: \n");
    compare_true(model);
    // test error after training
}

void tester_layer(Model *model, int layer)
{
    /* First test error for current model */

    compare_true(model);
    printf("Beginning training.. \n");
    // batch size training

    fc_model_train_layer(model, ft_samples_x, ft_samples_y, layer);

    printf("After training: \n");
    compare_true(model);
    // test error after training
}

void eqcheck(Model *model)
{
    printf("start eqcheck..\n");

    for (int i = 0; i < EQCHECK_N_SAMPLES; i++)
    {
        float *input = eqcheck_samples_x[i];
        float *output = fc_model_predict(model, input);
        // check output
        float tolerance = 0.0001;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            if (fabs(output[j] - eqcheck_samples_y[i][j]) > tolerance)
            {
                printf("FAILED: eqcheck for sample, expected: %f but predicted: %f\n", eqcheck_samples_y[i][j], output[j]);
                break;
                // printf("output: %f, did not match eqcheck: %f \n", output[j], eqcheck_samples_y[i][j]);
            }
        }
        free(output);
    }
    printf("eqcheck completed! \n");
}
void memory_tester(Model *model)
{

    reset_memory_tracking();
    printf("Memory stats for training for the whole network \n");
    fc_model_train(model, ft_samples_x, ft_samples_y);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the first layer \n");
    fc_model_train_layer(model, ft_samples_x, ft_samples_y, 0);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the last layer \n");
    fc_model_train_layer(model, ft_samples_x, ft_samples_y, model->n_layers - 1);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the first (0) layer, first weights \n");
    fc_model_train_partial_layer(model, ft_samples_x, ft_samples_y, 0, 1, 0);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("Memory stats for training for the second (1) layer, two last weights \n");
    fc_model_train_partial_layer(model, ft_samples_x, ft_samples_y, 1, 2, 2);
    print_memory();
    reset_memory_tracking();
    printf("\n \n");

    printf("\n Completed memory test \n");
    return;
}
void test_simple(Model *model)
{
    float *input = simple_samples_x[0];
    float *output = fc_model_predict(model, input);
    printf("output: %f \n", output[0]);
    fc_model_train(model, simple_samples_x, simple_samples_y);
}
int main()
{
    printf("starting.. \n");
    // Test data input to output from eqcheck
    Model *model = createAndSetModel(N_LAYERS, INPUT_SIZE, OUTPUT_SIZE, layers_size, layers_weights, layers_biases, layers_activation);
    printf("Set model \n");
    test_simple(model);
    // compare_true(model);
    //   eqcheck(model);

    // memory_tester(model);
    // compare_true(model);
    //   tester(model);

    return 0;
}