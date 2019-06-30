#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "layers/conv.h"
#include "layers/dense.h"
#include "layers/pooling.h"

void test_dense(float *input, int input_channel, int output_channel) {
    printf("****** Testing dense layer ******\n");
    printf("*** Step 1: init dense layer ***\n");
    Dense d(input_channel, output_channel);
    d.dump();

    printf("*** Step 2: forward ***\n");
    float *doutput = d.forward(input);
    float *output = (float*)malloc(sizeof(float) * output_channel);
    cudaMemcpy(output, doutput, sizeof(float) * output_channel, cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i=0; i<output_channel; i++) printf("%9.6f ", output[i]);
    printf("\n\n");

    for (int i=0; i<10; i++) output[i] = randn();
    printf("Loss: ");
    for (int i=0; i<10; i++) printf("%9.6f ", output[i]);
    printf("\n\n");
    float *loss;
    cudaMalloc(&loss, sizeof(float) * output_channel);
    cudaMemcpy(loss, output, sizeof(float) * 10, cudaMemcpyHostToDevice);

    printf("*** Step 3: backward ***\n");
    float *ddelta = d.backward(loss, 0.1);
    float *delta = (float*)malloc(sizeof(float) * output_channel);
    cudaMemcpy(delta, ddelta, sizeof(float) * output_channel, cudaMemcpyDeviceToHost);
    printf("Back-prop delta: ");
    for (int i=0; i<input_channel; i++) printf("%9.6f ", delta[i]);
    printf("\n\n");
    d.dump();

    free(output);
    free(delta);
    cudaFree(loss);
}

void test_pooling(float *input, int c, int h, int w, int s) {
    printf("***** Testing pooling layer ******\n");
    printf("*** Step 1: init pooling layer ***\n");
    Pooling p(c, h, w, s);
    p.dump();
    printf("\n");

    printf("*** Step 2: forward ***\n");
    float *doutput = p.forward(input);
    float *output = (float*)malloc(sizeof(float) * c*h/s*w/s);
    cudaMemcpy(output, doutput, sizeof(float) * c*h/s*w/s, cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h/s*w/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    for (int i=0; i<c*h/s*w/s; i++) output[i] = randn();
    printf("\nLoss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h*w/s/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }
    float *loss;
    cudaMalloc(&loss, sizeof(float) * c*h/s*w/s);
    cudaMemcpy(loss, output, sizeof(float) * c*h/s*w/s, cudaMemcpyHostToDevice);

    printf("*** Step 3: backward ***\n");
    float *dd = p.backward(loss, 0.1);
    float *d = (float*)malloc(sizeof(float)*c*h*w);
    cudaMemcpy(d, dd, sizeof(float)*c*h*w, cudaMemcpyDeviceToHost);

    printf("Back-prop delta:\n");
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

void test_conv(float *input, int c, int h, int w, int oc, int k, int s, int p) {
    printf("***** Testing conv layer ******\n");
    printf("*** Step 1: init conv layer ***\n");
    Conv conv(c, h, w, oc, k, s, p);
    conv.dump();

    printf("*** Step 2: forward ***\n");
    int oh = (h+2*p-k)/s+1;
    int ow = (w+2*p-k)/s+1;
    float *doutput = conv.forward(input);
    float *output = (float*)malloc(sizeof(float) * oc*oh*ow);
    cudaMemcpy(output, doutput, sizeof(float) * oc*oh*ow, cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }

    for (int i=0; i<oc*oh*ow; i++) output[i] = 2;
    printf("\nLoss:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }
    float *loss;
    cudaMalloc(&loss, sizeof(float) * oc*oh*ow);
    cudaMemcpy(loss, output, sizeof(float) * oc*oh*ow, cudaMemcpyHostToDevice);

    printf("*** Step 3: backward ***\n");
    float *dd = conv.backward(loss, 0.1);
    float *d = (float*)malloc(sizeof(float)*c*h*w);
    cudaMemcpy(d, dd, sizeof(float)*c*h*w, cudaMemcpyDeviceToHost);
    printf("Back-prop delta:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }
    conv.dump();

    free(d);
    free(output);
    cudaFree(loss);
}

int main() {
    rand_init();
    int c = 3, h = 10, w = 10;
    float *input = (float*)malloc(sizeof(float) * c*h*w);
    for (int i=0; i<c*h*w; i++) input[i] = randn();
    float *dinput;
    cudaMalloc(&dinput, sizeof(float) * c*h*w);
    cudaMemcpy(dinput, input, sizeof(float) * c*h*w, cudaMemcpyHostToDevice);

    printf("Input:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", input[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    test_conv(dinput, c, h, w, 5, 3, 2, 2);
    test_pooling(dinput, c, h, w, 2);
    test_dense(dinput, c, 10);

    free(input);
    cudaFree(dinput);
    return 0;
}
