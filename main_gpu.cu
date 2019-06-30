#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "layers/conv.h"
#include "layers/dense.h"
#include "layers/pooling.h"

void test_dense(double *input) {
    Dense d(6, 10);
    d.dump();

    double *doutput = d.forward(input);
    double *output = (double*)malloc(sizeof(double) * 10);
    cudaMemcpy(output, doutput, sizeof(double) * 10, cudaMemcpyDeviceToHost);

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

    free(output);
    cudaFree(loss);
}

void test_pooling(double *input, int c, int h, int w, int s) {
    Pooling p(c, h, w, s);
    p.dump();

    double *doutput = p.forward(input);
    double *output = (double*)malloc(sizeof(double) * c*h/s*w/s);
    cudaMemcpy(output, doutput, sizeof(double) * c*h/s*w/s, cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h/s*w/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    double *loss;
    cudaMalloc(&loss, sizeof(double) * c*h/s*w/s);
    for (int i=0; i<c*h/s*w/s; i++) output[i] = 99;
    printf("\nLoss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h*w/s/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }
    cudaMemcpy(loss, output, sizeof(double) * c*h/s*w/s, cudaMemcpyHostToDevice);

    double *dd = p.backward(loss, 0.1);
    double *d = (double*)malloc(sizeof(double)*c*h*w);
    cudaMemcpy(d, dd, sizeof(double)*c*h*w, cudaMemcpyDeviceToHost);

    printf("Output loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    free(d);
    free(output);
    cudaFree(loss);
}

void test_conv(double *input, int c, int h, int w, int oc, int k, int s, int p) {
    Conv c(c, h, w, oc, k, s, p);
    c.dump();

    int oh = (h+2*p-k)/s+1;
    int ow = (w+2*p-k)/s+1;
    double *doutput = c.forward(input);
    double *output = (double*)malloc(sizeof(double) * oc*oh*ow);
    cudaMemcpy(output, doutput, sizeof(double) * oc*oh*ow, cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }

    double *loss;
    cudaMalloc(&loss, sizeof(double) * oc*oh*ow);
    for (int i=0; i<oc*oh*ow; i++) output[i] = 2;
    printf("\nLoss:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }
    cudaMemcpy(loss, output, sizeof(double) * oc*oh*ow, cudaMemcpyHostToDevice);

    double *dd = c.backward(loss, 0.1);
    c.dump();
    double *d = (double*)malloc(sizeof(double)*c*h*w);
    cudaMemcpy(d, dd, sizeof(double)*c*h*w, cudaMemcpyDeviceToHost);

    printf("Output loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    free(d);
    free(output);
    cudaFree(loss);
}

int main() {
    rand_init();
    int c = 1, h = 5, w = 5;
    double *input = (double*)malloc(sizeof(double) * c*h*w);
    for (int i=0; i<c*h*w; i++) input[i] = 1;//randn();
    double *dinput;
    cudaMalloc(&dinput, sizeof(double) * c*h*w);
    cudaMemcpy(dinput, input, sizeof(double) * c*h*w, cudaMemcpyHostToDevice);

    printf("Input:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6lf ", input[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    // test_pooling(dinput, c, h, w, 3);
    test_conv(input, c, h, w, 1, 3, 1, 0);

    free(input);
    cudaFree(dinput);
    return 0;
}
