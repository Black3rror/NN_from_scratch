

#ifndef PARTIAL_MODEL_FC_H
#define PARTIAL_MODEL_FC_H
#include "../util/model_binding.h"
#include "../util/model_gradients.h"

void partial_calc_gradients(float *input, ModelPtr *model, int target_layer, int n_weights, int offset, float *actual, PartialGradients *gradients);

void fc_model_train_partial_layer(ModelPtr *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                                  int target_layer, int n_neurons, int offset);

void fc_model_train_layer(ModelPtr *model, float (*samples_x)[model->output_size], float (*samples_y)[model->output_size],
                          int target_layer);

#endif