#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dense.h"
#include "../common.h"

#ifdef CUDA
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif /* CUDA */

#ifdef CUDA
__global__ void dense_init_params(double *w, double *b, int ic) {
    int oi = threadIdx.x;
    curandState t;
    rand_init(&t, oi);
    for (int i=0; i<ic; i++) w[oi*ic + i] = randn(&t, oi) / sqrt((double)ic);
    b[oi] = 0;
}

__global__ void dense_forward(double *x, double *y, double *w, double *b, int ic) {
    int i = threadIdx.x;
    y[i] = 0;
    for (int j=0; j<ic; j++) {
        y[i] += w[i*ic + j] * x[j];
    }
    y[i] += b[i];
}

__global__ void dense_backward(double *delta, double *d, double *dw, double *db, double *w, double *b, double *x, int ic, double lr) {
    int i = threadIdx.x;
    db[i] = delta[i];
    for (int j=0; j<ic; j++) {
        dw[i*ic+j] = delta[i] * x[j];
        d[j] += delta[i] * w[i*ic+j];
    }
    __syncthreads();

    b[i] -= lr * db[i];
    for (int j=0; j<ic; j++) w[i*ic+j] -= lr * dw[i*ic+j];
}
#endif

Dense::Dense(int input_channels, int output_channels) {
    this->ic = input_channels;
    this->oc = output_channels;

    this->init_params();
}

void Dense::init_params(int idx) {
#ifdef CUDA
    cudaMalloc(&w, sizeof(double) * (ic*oc));
    cudaMalloc(&b, sizeof(double) * oc);
    cudaMalloc(&d, sizeof(double) * ic);
    cudaMalloc(&dw, sizeof(double) * (ic*oc));
    cudaMalloc(&db, sizeof(double) * oc);
    cudaMalloc(&y, sizeof(double) * oc);

    dense_init_params<<<1, oc>>>(w, b, ic);
#else
    this->w = (double*)malloc(sizeof(double) * (ic*oc));
    this->b = (double*)malloc(sizeof(double) * oc);
    this->d = (double*)malloc(sizeof(double) * ic);
    this->dw = (double*)malloc(sizeof(double) * (ic*oc));
    this->db = (double*)malloc(sizeof(double) * oc);
    this->y = (double*)malloc(sizeof(double) * oc);

    for (int i=0; i<ic*oc; i++) w[i] = randn() / sqrt(ic);
    for (int i=0; i<oc; i++) b[i] = 0;
#endif
}

double* Dense::forward(double *input) {
    x = input;
#ifdef CUDA
    dense_forward<<<1, oc>>>(x, y, w, b, ic);
#else
    for (int i=0; i<oc; i++) {
        y[i] = 0;
        for (int j=0; j<ic; j++) {
            y[i] += w[i*ic + j] * x[j];
        }
        y[i] += b[i];
    }
#endif
    return y;
}

double* Dense::backward(double *delta, double lr) {
#ifdef CUDA
    cudaMemset(d, 0, sizeof(double) * ic);
    dense_backward<<<1, oc>>>(delta, d, dw, db, w, b, x, ic, lr);
#else
    memset(d, 0, sizeof(double) * ic);
    for (int i=0; i<oc; i++) {
        db[i] = delta[i];
        for (int j=0; j<ic; j++) {
            dw[i*ic+j] = delta[i] * x[j];
            d[j] += delta[i] * w[i*ic+j];
        }
    }

    for (int i=0; i<oc; i++) {
        b[i] -= lr * db[i];
        for (int j=0; j<ic; j++) w[i*ic+j] -= lr * dw[i*ic+j];
    }
#endif
    return d;
}

void Dense::dump() {
#ifdef CUDA
    double w[ic*oc], b[oc];
    cudaMemcpy(w, this->w, sizeof(double)*ic*oc, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, this->b, sizeof(double)*oc, cudaMemcpyDeviceToHost);
#endif
    printf("\nDense_%d_%d:\n", ic, oc);
    int n = ic*5;
    for (int i=0; i<n; i++) printf("-");
    printf(" w ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n");
    for (int i=0; i<oc; i++) {
        printf("| ");
        for (int j=0; j<ic; j++) printf("%9.6lf ", w[i*ic+j]);
        printf("|\n");
    }
    for (int i=0; i<ic*10+3; i++) printf("-");
    printf("\n");

    n = oc*5;
    for (int i=0; i<n; i++) printf("-");
    printf(" b ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n| ");
    for (int i=0; i<oc; i++) printf("%9.6lf ", b[i]);
    printf("|\n");
    for (int i=0; i<oc*10+3; i++) printf("-");
    printf("\n\n");
}

Dense::~Dense() {
#ifdef CUDA
    cudaFree(w);
    cudaFree(b);
    cudaFree(y);
    cudaFree(d);
    cudaFree(dw);
    cudaFree(db);
#else
    free(w);
    free(b);
    free(y);
    free(d);
    free(dw);
    free(db);
#endif
}
