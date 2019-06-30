#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "conv.h"
#include "../common.h"

#ifdef CUDA
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif /* CUDA */

#ifdef CUDA
__global__ void conv_init_params(double *w, double *b, int ic, int k) {
    int oc = threadIdx.x;
    curandState t;
    rand_init_gpu(&t, oc);
    for (int i=0; i<ic*k*k; i++) w[oc*ic*k*k + i] = 1;//randn_gpu(&t, oc) / sqrt((double)ic);
    b[oc] = 0;
}

__global__ void conv_forward(double *x, double *y, double *w, double *b, int ic, int ih, int iw, int oc, int oh, int ow, int k, int s, int p) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int chn = blockIdx.z*blockDim.z + threadIdx.z;
    if (row >= oh || col >= ow || chn >= oc) return;

    //__shared__ float sw[ic*k*k];
    for (int iic=0; iic<ic; iic++) {
        /*
        if (row < k && col < k)
            sw[iic*k*k + row*k + col] = w[chn*ic*k*k + iic*k*k + row*k + col];
        __syncthreads();
        */

        for (int iih=-p; iih<ih+p; iih+=s) {
            if (iih+k >= iw) break;
            for (int iiw=-p; iiw<iw+p; iiw+=s) {
                if (iiw+k >= iw) break;
                for (int wh=0; wh<k; wh++) {
                    if (iih+wh < 0 || iih+wh >= ih) continue;
                    for (int ww=0; ww<k; ww++) {
                        if (iiw+ww < 0 || iiw+ww >= iw) continue;
                        y[chn*oh*ow + row*ow + col] += w[chn*ic*k*k + iic*k*k + wh*k + ww] * x[iic*ih*iw + (iih+wh)*iw + iiw+ww];
                    }
                }
            }
        }
        __syncthreads();
        y[chn*oh*ow + row*ow + col] += b[chn];
    }
}

__global__ void conv_backward(double *delta, double *d, double *dw, double *db, double *w, double *b, double *x,
  int oc, int oh, int ow, int ic, int ih, int iw, int k, int s, int p, double lr) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int chn = blockIdx.z*blockDim.z + threadIdx.z;
    if (row >= oh || col >= ow || chn >= oc) return;

    // Calculate db
    db[chn] += delta[chn*oh*ow + row*ow + col];

    // Calculate dw
    for (int iih=-p; iih<ih+p; iih+=s) {
        if (iih+k >= ih) break;
        for (int iiw=-p; iiw<iw+p; iiw+=s) {
            if (iiw+k >= iw) break;
            for (int iic=0; iic<ic; iic++) {
                for (int dh=0; dh<k; dh++) {
                    if (iih+dh < 0 || iih+dh >= ih) continue;
                    for (int ddw=0; ddw<k; ddw++) {
                        if (iiw+ddw < 0 || iiw+ddw >= iw) continue;
                        dw[chn*ic*ih*iw + iic*ih*iw + dh*iw + ddw] += delta[chn*oh*ow + row*ow + col] * x[iic*ih*iw + (iih+dh)*iw + iiw+ddw];
                    }
                }
            }
        }
    }

    // Calculate d
    //__shared__ float sw[ic*k*k];
    for (int iic=0; iic<ic; iic++) {
        /*
        if (row < k && col < k)
            sw[iic*k*k + row*k + col] = w[chn*ic*k*k + iic*k*k + row*k + col];
        __syncthreads();
        */
        for (int iih=-p; iih<ih+p; iih+=s) {
            if (iih+k >= ih) break;
            for (int iiw=-p; iiw<iw+p; iiw+=s) {
                if (iiw+k >= iw) break;
                for (int wh=0; wh<k; wh++) {
                    if (iih+wh < 0 || iih+wh >= ih) continue;
                    for (int ww=0; ww<k; ww++) {
                        if (iiw+ww < 0 || iiw+ww >= iw) continue;
                        d[iic*ih*iw + (iih+wh)*iw + iiw+ww] += w[chn*ic*k*k + iic*k*k + wh*k + ww] * delta[chn*oh*ow + row*ow + col];
                    }
                }
            }
        }
    }
    __syncthreads();

    // Update w & b
    b[chn] -= lr * db[chn];
    for (int iic=0; iic<ic; iic++) {
        for (int wh=0; wh<k; wh++) {
            for (int ww=0; ww<k; ww++) {
                w[chn*ic*k*k + iic*k*k + wh*k + ww] -= lr * dw[chn*ic*k*k + iic*k*k + wh*k + ww];
            }
        }
    }
}
#endif

Conv::Conv(int input_channels, int input_height, int input_width, int output_channels, int kernel, int stride, int padding) {
    this->ic = input_channels;
    this->ih = input_height;
    this->iw = input_width;
    this->oc = output_channels;
    this->k = kernel;
    this->s = stride;
    this->p = padding;
    this->oh = (ih+2*p-k)/s+1;
    this->ow = (iw+2*p-k)/s+1;

    this->init_params();
}

void Conv::init_params() {
#ifdef CUDA
    cudaMalloc(&w, sizeof(double) * oc*ic*k*k);
    cudaMalloc(&b, sizeof(double) * oc);
    cudaMalloc(&d, sizeof(double) * ic*ih*iw);
    cudaMalloc(&dw, sizeof(double) * oc*ic*k*k);
    cudaMalloc(&db, sizeof(double) * oc);
    cudaMalloc(&y, sizeof(double) * oc*oh*ow);

    conv_init_params<<<1, oc>>>(w, b, ic, k);
#else
    this->w = (double*)malloc(sizeof(double) * oc*ic*k*k);
    this->b = (double*)malloc(sizeof(double) * oc);
    this->d = (double*)malloc(sizeof(double) * ic*ih*iw);
    this->dw = (double*)malloc(sizeof(double) * oc*ic*k*k);
    this->db = (double*)malloc(sizeof(double) * oc);
    this->y = (double*)malloc(sizeof(double) * oc*oh*ow);

    rand_init();
    for (int i=0; i<oc*ic*k*k; i++) w[i] = 1;//randn() / sqrt(ic);
    for (int i=0; i<oc; i++) b[i] = 0;
#endif
}

double* Conv::forward(double *input) {
    x = input;
#ifdef CUDA
    cudaMemset(y, 0, sizeof(double) * oc*oh*ow);
    const int TILE = 16, TILEZ = 3;
    dim3 blocks((ow-1)/TILE+1, (oh-1)/TILE+1, (oc-1)/TILEZ+1), threads(TILE, TILE, TILEZ);
    conv_forward<<<blocks, threads>>>(x, y, w, b, ic, ih, iw, oc, oh, ow, k, s, p);
#else
    memset(y, 0, sizeof(double) * oc*oh*ow);
    for (int oc=0; oc<this->oc; oc++) {
        int oh = 0, ow = 0;
        for (int ih=-p; ih<this->ih+p; ih+=s) {
            for (int iw=-p; iw<this->iw+p; iw+=s) {
                for (int ic=0; ic<this->ic; ic++) {
                    for (int wh=0; wh<k; wh++) {
                        if (ih+wh < 0 || ih+wh >= this->ih) continue;
                        for (int ww=0; ww<k; ww++) {
                            if (iw+ww < 0 || iw+ww >= this->iw) continue;
                            // printf("(%d,%d,%d) += (%d,%d,%d,%d) * (%d,%d,%d)\n", oc, oh, ow, oc, ic, wh, ww, ic, ih+wh, iw+ww);
                            y[oi(oc, oh, ow)] += w[wi(oc, ic, wh, ww)] * x[ii(ic, ih+wh, iw+ww)];
                        }
                    }
                }
                y[oi(oc, oh, ow)] += b[oc];
                ow++;
                if (ow >= this->ow) break;
            }
            ow = 0;
            oh++;
            if (oh >= this->oh) break;
        }
    }
#endif
    return y;
}

double* Conv::backward(double *delta, double lr) {
#ifdef CUDA
    cudaMemset(d, 0, sizeof(double) * ic*ih*iw);
    cudaMemset(dw, 0, sizeof(double) * oc*ic*k*k);
    cudaMemset(db, 0, sizeof(double) * oc);
    const int TILE = 16, TILEZ = 3;
    dim3 blocks((ow-1)/TILE+1, (oh-1)/TILE+1, (oc-1)/TILEZ+1), threads(TILE, TILE, TILEZ);
    conv_backward<<<blocks, threads>>>(delta, d, dw, db, w, b, x, oc, oh, ow, ic, ih, iw, k, s, p, lr);
#else
    memset(d, 0, sizeof(double) * ic*ih*iw);
    memset(dw, 0, sizeof(double) * oc*ic*k*k);
    memset(db, 0, sizeof(double) * oc);
    for (int oc=0; oc<this->oc; oc++) {
        // Calculate db
        for (int oh=0; oh<this->oh; oh++) {
            for (int ow=0; ow<this->ow; ow++) {
                db[oc] += delta[oi(oc, oh, ow)];
            }
        }

        // Calculate dw
        int oh = 0, ow = 0;
        for (int ih=-p; ih<this->ih+p; ih+=s) {
            for (int iw=-p; iw<this->iw+p; iw+=s) {
                for (int ic=0; ic<this->ic; ic++) {
                    for (int dh=0; dh<k; dh++) {
                        if (ih+dh < 0 || ih+dh >= this->ih) continue;
                        for (int dw=0; dw<k; dw++) {
                            if (iw+dw < 0 || iw+dw >= this->iw) continue;
                            // printf("(%d,%d,%d,%d) += (%d,%d,%d) * (%d,%d,%d)\n", oc, ic, dh, dw, oc, oh, ow, ic, ih+dh, iw+dw);
                            this->dw[wi(oc, ic, dh, dw)] += delta[oi(oc, oh, ow)] * x[ii(ic, ih+dh, iw+dw)];
                        }
                    }
                }
                ow++;
                if (ow >= this->ow) break;
            }
            ow = 0;
            oh++;
            if (oh >= this->oh) break;
        }
    }

    // Calculate d
    for (int oc=0; oc<this->oc; oc++) {
        int oh = 0, ow = 0;
        for (int ih=-p; ih<this->ih+p; ih+=s) {
            for (int iw=-p; iw<this->iw+p; iw+=s) {
                for (int ic=0; ic<this->ic; ic++) {
                    for (int wh=0; wh<k; wh++) {
                        if (ih+wh < 0 || ih+wh >= this->ih) continue;
                        for (int ww=0; ww<k; ww++) {
                            if (iw+ww < 0 || iw+ww >= this->iw) continue;
                            d[ii(ic, ih+wh, iw+ww)] += w[wi(oc, ic, wh, ww)] * delta[oi(oc, oh, ow)];
                        }
                    }
                }
                ow++;
                if (ow >= this->ow) break;
            }
            ow = 0;
            oh++;
            if (oh >= this->oh) break;
        }
    }

    // Update w & b
    for (int oc=0; oc<this->oc; oc++) {
        b[oc] -= lr * db[oc];
        for (int ic=0; ic<this->ic; ic++) {
            for (int wh=0; wh<k; wh++) {
                for (int ww=0; ww<k; ww++) {
                    w[wi(oc, ic, wh, ww)] -= lr * dw[wi(oc, ic, wh, ww)];
                }
            }
        }
    }
#endif
    return d;
}

void Conv::dump() {
#ifdef CUDA
    double w[oc*ic*k*k], b[oc];
    cudaMemcpy(w, this->w, sizeof(double)*oc*ic*k*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, this->b, sizeof(double)*oc, cudaMemcpyDeviceToHost);
#endif
    printf("Conv_(%d,%d,%d)_(%d,%d,%d)_%d_%d_%d:\n", ic, ih, iw, oc, oh, ow, k, s, p);
    int n = (k*10-3)/2;
    for (int i=0; i<n; i++) printf("-");
    printf(" w ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n");
    for (int oc=0; oc<this->oc; oc++) {
        printf("w%d:\n", oc);
        for (int ic=0; ic<this->ic; ic++) {
            for (int i=0; i<k; i++) {
                for (int j=0; j<k; j++) printf("%9.6lf ", w[wi(oc, ic, i, j)]);
                printf("\n");
            }
            printf("\n");
        }
    }

    n = (this->oc*10-3)/2;
    for (int i=0; i<n; i++) printf("-");
    printf(" b ");
    for (int i=0; i<n; i++) printf("-");
    printf("\n");
    for (int i=0; i<this->oc; i++) printf("%9.6lf ", b[i]);
    printf("\n");
    for (int i=0; i<this->oc*10; i++) printf("-");
    printf("\n\n");
}

Conv::~Conv() {
#ifdef CUDA
    cudaFree(w);
    cudaFree(b);
    cudaFree(d);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(y);
#else
    free(w);
    free(b);
    free(d);
    free(dw);
    free(db);
    free(y);
#endif
}
