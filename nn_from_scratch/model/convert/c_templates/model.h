#ifndef MODEL_H
#define MODEL_H

#include <cstdint>

#define INPUT_SIZE {input_size}
#define OUTPUT_SIZE {output_size}
#define N_LAYERS {n_layers}

{layers_size}

enum ActivationType {
    LINEAR,
    RELU
};

extern int layers_size[N_LAYERS];
extern float* layers_weights[N_LAYERS];     // shape: (n_layers)(input_size * output_size)
extern float* layers_biases[N_LAYERS];      // shape: (n_layers)(output_size)
extern enum ActivationType layers_activation[N_LAYERS];

#endif
