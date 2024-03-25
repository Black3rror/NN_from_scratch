#include "model/model.h"

extern float relu(float x);
extern float linear(float x);
extern float activation_func(enum ActivationType activation, float x);
extern float *dense(float *input, float *weights, float *biases, float *output, int input_size, int output_size, enum ActivationType activation);
extern float *f_prop_helper(float *input, int input_size, int i, int N);
extern float *f_prop(float *input, int input_size, int N);