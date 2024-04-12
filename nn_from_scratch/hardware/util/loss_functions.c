/* Loss function MSE*/
float MSE(float *predicted, float *actual, int size)
{
    float error = 0.0;
    for (int i = 0; i < size; i++)
    {
        float diff = predicted[i] - actual[i];
        error += diff * diff;
    }
    return error / size;
}
float MSE_Derivative(float *predicted, float *actual, int size)
{
    float error = 0.0;
    for (int i = 0; i < size; i++)
    {
        float diff = predicted[i] - actual[i];
        error += 2 * diff;
    }
    return error / size;
}
