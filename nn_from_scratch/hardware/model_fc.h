#include "util/forward_prop.h"
#include "util/activation_functions.h"
#include "util/back_prop.h"
#include "util/loss_functions.h"
#include "util/model_binding.h"

void fc_model_training(ModelPtr *model, float (*samples_x)[model->n_layers], float (*samples_y)[model->n_layers], int n_samples, float step_rate);
float *fc_model_predict(ModelPtr *model, float *input);
