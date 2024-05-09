
#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H
extern float MSE(float *predicted, float *actual, int size);

extern float MSE_Derivative(float *predicted, float *actual, int size);
#endif