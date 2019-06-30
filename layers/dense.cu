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
__global__ void dense_init_params(float *w, float *b, int ic) {
    int oi = threadIdx.x;
    curandState t;
    rand_init_gpu(&t, oi);
    for (int i=0; i<ic; i++) w[oi*ic + i] = randn_gpu(&t, oi) / sqrt((float)ic);
    b[oi] = 0;
}

__global__ void dense_forward(float *x, float *y, float *w, float *b, int ic) {
    int i = threadIdx.x;
    y[i] = 0;
    for (int j=0; j<ic; j++) {
        y[i] += w[i*ic + j] * x[j];
    }
    y[i] += b[i];
}

__global__ void dense_backward(float *delta, float *d, float *dw, float *db, float *w, float *b, float *x, int ic, float lr) {
    int i = threadIdx.x;
    db[i] = delta[i];
    for (int j=0; j<ic; j++) {
        dw[i*ic+j] = delta[i] * x[j];
        atomicAdd(&d[j], delta[i] * w[i*ic+j]);
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

void Dense::init_params() {
#ifdef CUDA
    cudaMalloc(&w, sizeof(float) * (ic*oc));
    cudaMalloc(&b, sizeof(float) * oc);
    cudaMalloc(&d, sizeof(float) * ic);
    cudaMalloc(&dw, sizeof(float) * (ic*oc));
    cudaMalloc(&db, sizeof(float) * oc);
    cudaMalloc(&y, sizeof(float) * oc);

    dense_init_params<<<1, oc>>>(w, b, ic);
#else
    this->w = (float*)malloc(sizeof(float) * (ic*oc));
    this->b = (float*)malloc(sizeof(float) * oc);
    this->d = (float*)malloc(sizeof(float) * ic);
    this->dw = (float*)malloc(sizeof(float) * (ic*oc));
    this->db = (float*)malloc(sizeof(float) * oc);
    this->y = (float*)malloc(sizeof(float) * oc);

    rand_init();
    for (int i=0; i<ic*oc; i++) w[i] = randn() / sqrt(ic);
    for (int i=0; i<oc; i++) b[i] = 0;
#endif
}

float* Dense::forward(float *input) {
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

float* Dense::backward(float *delta, float lr) {
#ifdef CUDA
    cudaMemset(d, 0, sizeof(float) * ic);
    dense_backward<<<1, oc>>>(delta, d, dw, db, w, b, x, ic, lr);
#else
    memset(d, 0, sizeof(float) * ic);
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
    float w[ic*oc], b[oc];
    cudaMemcpy(w, this->w, sizeof(float)*ic*oc, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, this->b, sizeof(float)*oc, cudaMemcpyDeviceToHost);
#endif
    printf("Dense_%d_%d:\n", ic, oc);
    int n = ic*5;
    for (int i=0; i<n; i++) printf("-");
    printf(" w ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n");
    for (int i=0; i<oc; i++) {
        printf("| ");
        for (int j=0; j<ic; j++) printf("%9.6f ", w[i*ic+j]);
        printf("|\n");
    }
    for (int i=0; i<ic*10+3; i++) printf("-");
    printf("\n");

    n = oc*5;
    for (int i=0; i<n; i++) printf("-");
    printf(" b ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n| ");
    for (int i=0; i<oc; i++) printf("%9.6f ", b[i]);
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
