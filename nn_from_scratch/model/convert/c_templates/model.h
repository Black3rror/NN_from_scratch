#include <stdint.h>

#define INPUT_SIZE 1
#define N_LAYERS 4

#define LAYER_0_SIZE 16
#define LAYER_1_SIZE 8
#define LAYER_2_SIZE 2
#define LAYER_3_SIZE 1


enum ActivationType {
    LINEAR,
    RELU
};

extern int layers_size[N_LAYERS];
extern float* layers_weights[N_LAYERS];     // shape: (n_layers)(input_size * output_size)
extern float* layers_biases[N_LAYERS];      // shape: (n_layers)(output_size)
extern enum ActivationType layers_activation[N_LAYERS];
