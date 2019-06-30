#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "common.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ static void rand_init_gpu(curandState *t, int idx) {
    curand_init(clock64(), idx, 0, t);
}

__device__ static float randn_gpu(curandState *t, int idx) {
    return curand_normal(t);
}
#endif /* CUDA */

static void rand_init(void) {
    srand(clock());
}

static float randn(void) {
    float x = (float)rand() / RAND_MAX;
    float y = (float)rand() / RAND_MAX;
    return sqrt(-2 * log(x)) * cos(2 * M_PI * y);
}

#endif
