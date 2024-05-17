#include <stdio.h>
enum Activation
{
    LINEAR,
    RELU
};
#define num_v(macro)  \
    int num_##macro() \
    {                 \
        return macro; \
    }
#define linear(x) (x)
#define relu(x) (x)
// Define macros for each activation function
#define ACTIVATION_MACRO_LIST \
    X(LINEAR, linear)         \
    X(RELU, relu)

// Define a macro to generate function variants for each activation function
#define GENERATE_FUNCTION_VARIANTS(act, function) \
    int num_##act(int x) { return function(x); }

// Generate function variants for all activation functions
#define X(act, func) GENERATE_FUNCTION_VARIANTS(act, func)
ACTIVATION_MACRO_LIST
#undef X

int main()
{
    printf("linear() returns: %d\n", num_LINEAR(2));
    printf("relu() returns: %d\n", num_RELU(3));
    return 0;
}