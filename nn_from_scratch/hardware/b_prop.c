/*
    C file for back propagation
*/

/* Back propagation function for a single layer*/
void b_prop(float* input, int input_size, int i)
{
    // Compute loss
    // Compute gradient
    // use Chain rule
    // Update parameters
    return 0;
}
//maybe not needed
void b_prop_helper(float* input, int input_size, int i){

    return 0;
}

/* Loss function */

void loss(float predicted, float target)
{
    return 0;
}

/* Compute gradient*/

void gradient(float predicted, float target, float input)
{
    float grad = input * (predicted - target); //maybe normalize?
    
    return grad;

}

/* Update function to update weights (use gradient decent)*/
// inputs: weight, gradient, learning
void update_weights(float *weight, float grad, float rate)
{
    *weight -= rate * grad;
    return 0;
}
/* Main function for testing*/
void main()
{

    return 0;
}