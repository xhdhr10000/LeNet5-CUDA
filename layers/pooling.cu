#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#include "pooling.h"
#include "../common.h"

#ifdef CUDA
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif /* CUDA */

#ifdef CUDA
__global__ void pooling_forward(double *x, double *y, int *p, int c, int ih, int iw, int oh, int ow, int s) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= oh || col >= ow) return;
    for (int i=0; i<c; i++) {
        y[i*oh*ow + row*ow + col] = -FLT_MAX;
        for (int j=0; j<s; j++) {
            for (int k=0; k<s; k++) {
                if (x[i*ih*iw + (row*s+j)*iw + col*s+k] > y[i*oh*ow + row*ow + col]) {
                    y[i*oh*ow + row*ow + col] = x[i*ih*iw + (row*s+j)*iw + col*s+k];
                    p[i*oh*ow + row*ow + col] = j*s+k;
                }
            }
        }
    }
}

__global__ void pooling_backward(double *delta, double *d, int *p, int c, int ih, int iw, int oh, int ow, int s) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= oh || col >= ow) return;
    for (int i=0; i<c; i++) {
        d[i*ih*iw + (row*s + p[i*oh*ow+row*ow+col]/s)*iw + col*s + p[i*oh*ow+row*ow+col]%s] = delta[i*oh*ow + row*ow + col];
    }
}
#endif

Pooling::Pooling(int channels, int height, int width, int stride) {
    this->c = channels;
    this->ih = height;
    this->iw = width;
    this->s = stride;
    this->oh = this->ih / s;
    this->ow = this->iw / s;

    this->init_params();
}

void Pooling::init_params(int idx) {
#ifdef CUDA
    cudaMalloc(&y, sizeof(double) * (c*oh*ow));
    cudaMalloc(&d, sizeof(double) * (c*ih*iw));
    cudaMalloc(&p, sizeof(int) * (c*oh*ow));
#else
    this->y = (double*)malloc(sizeof(double) * (c*oh*ow));
    this->d = (double*)malloc(sizeof(double) * (c*ih*iw));
    this->p = (int*)malloc(sizeof(int) * (c*oh*ow));
#endif
}

double* Pooling::forward(double *input) {
    x = input;
#ifdef CUDA
    cudaMemset(p, 0, sizeof(int)*(c*oh*ow));
    const int TILE = 32/s/s*s;
    dim3 blocks((ow-1)/TILE+1, (oh-1)/TILE+1), threads(TILE, TILE);
    pooling_forward<<<blocks, threads>>>(x, y, p, c, ih, iw, oh, ow, s);
#else
    memset(p, 0, sizeof(int)*(c*oh*ow));
    for (int i=0; i<c; i++) {
        for (int j=0; j<ih; j+=s) {
            for (int k=0; k<iw; k+=s) {
                y[oi(i, j/s, k/s)] = -MAXFLOAT;
                for (int m=0; m<s; m++) {
                    for (int n=0; n<s; n++) {
                        if (x[ii(i, j+m, k+n)] > y[oi(i, j/s, k/s)]) {
                            y[oi(i, j/s, k/s)] = x[ii(i, j+m, k+n)];
                            p[oi(i, j/s, k/s)] = m*s+n;
                        }
                    }
                }
            }
        }
    }
#endif
    return y;
}

double* Pooling::backward(double *delta, double lr) {
#ifdef CUDA
    cudaMemset(d, 0, sizeof(double) * (c*ih*iw));
    const int TILE = 32/s/s*s;
    dim3 blocks((ow-1)/TILE+1, (oh-1)/TILE+1), threads(TILE, TILE);
    pooling_backward<<<blocks, threads>>>(delta, d, p, c, ih, iw, oh, ow, s);
#else
    memset(d, 0, sizeof(double) * (c*ih*iw));
    for (int i=0; i<c; i++) {
        for (int j=0; j<oh; j++) {
            for (int k=0; k<ow; k++) {
                d[ii(i, j*s + p[oi(i,j,k)]/s, k*s + p[oi(i,j,k)]%s)] = delta[oi(i,j,k)];
            }
        }
    }
#endif
    return d;
}

void Pooling::dump() {
    printf("Pooling_(%d,%d,%d)_%d\n", c, iw, ih, s);
}

Pooling::~Pooling() {
#ifdef CUDA
    cudaFree(y);
    cudaFree(d);
    cudaFree(p);
#else
    free(y);
    free(d);
    free(p);
#endif
}
