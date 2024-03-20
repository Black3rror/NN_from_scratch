#include <stdint.h>
#include "model.h"

float layer_0_weights[] = {0.57565004, 0.25474462, -0.3095492, 0.8526079};
float layer_1_weights[] = {0.7308921, 0.45773426, 0.27694538, 1.1145339, 0.63199854, -0.7626424, 0.15427063, 0.5997223};
float layer_2_weights[] = {0.7494462, -0.6259115};

float layer_0_biases[] = {0.059576977, -0.3204974, 0.0, -0.23192757};
float layer_1_biases[] = {0.22568879, -0.71715707};
float layer_2_biases[] = {0.35364622};


int layers_size[N_LAYERS] = {LAYER_0_SIZE, LAYER_1_SIZE, LAYER_2_SIZE};
float* layers_weights[N_LAYERS] = {layer_0_weights, layer_1_weights, layer_2_weights};
float* layers_biases[N_LAYERS] = {layer_0_biases, layer_1_biases, layer_2_biases};
enum ActivationType layers_activation[N_LAYERS] = {RELU, RELU, LINEAR};
