#include <stdio.h>
// #include <cuda_runtime.h>

class Conv {
private:
    int ic, oc, k, s, p;
    double *w, *b;

public:
    Conv(int input_channels, int output_channels, int kernel, int stride, int padding) {
        this->ic = input_channels;
        this->oc = output_channels;
        this->k = kernel;
        this->s = stride;
        this->p = padding;
    }

    void forward(double *input, int w, int h, double *output) {
    }

    void backward(double *delta) {
    }
};