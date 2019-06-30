#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "layers/conv.h"
#include "layers/dense.h"
#include "layers/pooling.h"

void test_dense(double *input) {
    Dense d(6, 10);
    d.dump();
    double *output = d.forward(input);

    printf("Output: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    printf("\n");

    for (int i=0; i<10; i++) output[i] = randn();
    printf("\nLoss: ");
    for (int i=0; i<10; i++) printf("%9.6lf ", output[i]);
    d.backward(output, 0.1);
    d.dump();
}

void test_pooling(double *input, int c, int h, int w, int s) {
    Pooling p(c, h, w, s);
    p.dump();
    double *output = p.forward(input);

    printf("Output:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h/s*w/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    for (int i=0; i<c*h*w/s/s; i++) output[i] = randn();
    printf("\nLoss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h/s; j++) {
            for (int k=0; k<w/s; k++) printf("%9.6f ", output[i*h*w/s/s + j*w/s + k]);
            printf("\n");
        }
        printf("\n");
    }

    double *d = p.backward(output, 0.1);
    printf("Output loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }
}

void test_conv(double *input, int c, int h, int w, int oc, int k, int s, int p) {
    Conv conv(c, h, w, oc, k, s, p);
    conv.dump();

    double *output = conv.forward(input);
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

    for (int i=0; i<oc*oh*ow; i++) output[i] = 2;
    double *d = conv.backward(output, 0.1);
    conv.dump();
    printf("Output loss:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6f ", d[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    rand_init();
    int c = 1, w = 5, h = 5;
    double *input = (double*)malloc(sizeof(double)*(c*h*w));
    for (int i=0; i<c*h*w; i++) input[i] = 1;//randn();
    printf("Input:\n");
    for (int i=0; i<c; i++) {
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) printf("%9.6lf ", input[i*h*w + j*w + k]);
            printf("\n");
        }
        printf("\n");
    }

    test_conv(input, c, h, w, 1, 3, 2, 1);

    free(input);
    return 0;
}