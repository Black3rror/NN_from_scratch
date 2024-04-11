
typedef float(*ActivationFunc)(float, int);
extern float relu(float x, int derivative_flag);
extern float linear(float x, int derivative_flag);
