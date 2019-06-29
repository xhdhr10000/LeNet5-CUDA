#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "layers/dense.h"

int main() {
    Dense d(6, 10);
    d.dump();

    double *input = (double*)malloc(sizeof(double)*6);
    for (int i=0; i<6; i++) input[i] = 1;
    double *dinput;
    cudaMalloc(&dinput, sizeof(double) * 6);
    cudaMemcpy(dinput, input, sizeof(double) * 6, cudaMemcpyHostToDevice);

    double *doutput = d.forward(dinput);
    double *output = (double*)malloc(sizeof(double) * 10);
    cudaMemcpy(output, doutput, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int i=0; i<6; i++) printf("%9.6lf ", input[i]);
    printf("\nOutput: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    printf("\n");

    double *loss;
    cudaMalloc(&loss, sizeof(double) * 10);
    for (int i=0; i<10; i++) output[i] = 2;
    printf("\nLoss: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    cudaMemcpy(loss, output, sizeof(double) * 10, cudaMemcpyHostToDevice);

    d.backward(loss, 0.1);
    d.dump();

    free(input);
    free(output);
    cudaFree(dinput);
    cudaFree(loss);
    return 0;
}
