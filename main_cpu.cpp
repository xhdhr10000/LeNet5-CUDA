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
    float *output = d.forward(input);
    printf("Output: ");
    for (int i=0; i<output_channel; i++) printf("%9.6f ", output[i]);
    printf("\n\n");

    for (int i=0; i<output_channel; i++) output[i] = randn();
    printf("Loss: ");
    for (int i=0; i<output_channel; i++) printf("%9.6f ", output[i]);
    printf("\n\n");

    printf("*** Step 3: backward ***\n");
    float *delta = d.backward(output, 0.1);
    printf("Back-prop delta: ");
    for (int i=0; i<input_channel; i++) printf("%9.6f ", delta[i]);
    printf("\n\n");
    d.dump();
}

void test_pooling(float *input, int c, int h, int w, int s) {
    printf("***** Testing pooling layer ******\n");
    printf("*** Step 1: init pooling layer ***\n");
    Pooling p(c, h, w, s);
    p.dump();
    printf("\n");

    printf("*** Step 2: forward ***\n");
    float *output = p.forward(input);
    printf("Output:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h/s*w/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    for (int i=0; i<c*h*w/s/s; i++) output[i] = randn();
    printf("Loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h*w/s/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    printf("*** Step 3: backward ***\n");
    float *d = p.backward(output, 0.1);
    printf("Back-prop loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }
}

void test_conv(float *input, int c, int h, int w, int oc, int k, int s, int p) {
    printf("***** Testing conv layer ******\n");
    printf("*** Step 1: init conv layer ***\n");
    Conv conv(c, h, w, oc, k, s, p);
    conv.dump();

    printf("*** Step 2: forward ***\n");
    float *output = conv.forward(input);
    int oh = (h+2*p-k)/s+1;
    int ow = (w+2*p-k)/s+1;
    printf("Output:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }

    for (int i=0; i<oc*oh*ow; i++) output[i] = randn();
    printf("Loss:\n");
    for (int i=0; i<oc; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) printf("%9.6f ", output[i*oh*ow + j*ow + k]);
            printf("\n");
        }
        printf("\n");
    }

    printf("*** Step 3: backward ***\n");
    float *d = conv.backward(output, 0.1);
    printf("Back-prop delta:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }
    conv.dump();
}

int main() {
    rand_init();
    int c = 3, w = 10, h = 10;
    float *input = (float*)malloc(sizeof(float)*(c*h*w));
    for (int i=0; i<c*h*w; i++) input[i] = randn();
    printf("Input:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", input[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    test_conv(input, c, h, w, 5, 3, 2, 2);
    test_pooling(input, c, h, w, 2);
    test_dense(input, c, 10);

    free(input);
    return 0;
}