#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "layers/dense.h"

int main() {
    rand_init();
    Dense d(6, 10);
    double *input = (double*)malloc(sizeof(double)*6);
    for (int i=0; i<6; i++) input[i] = randn();

    d.dump();
    double *output = d.forward(input);

    printf("Input: ");
    for (int i=0; i<6; i++) printf("%9.6lf ", input[i]);
    printf("\nOutput: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    printf("\n");

    for (int i=0; i<10; i++) output[i] = randn();
    printf("\nLoss: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    d.backward(output, 0.1);
    d.dump();

    free(input);
    return 0;
}