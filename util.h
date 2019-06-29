#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "common.h"

#ifdef CUDA
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif /* CUDA */

#ifdef CUDA
__device__ static void rand_init(curandState *t, int idx) {
    curand_init(clock64(), idx, 0, t);
}
#else
static void rand_init(void) {
    srand(clock());
}
#endif

#ifdef CUDA
__device__ static double randn(curandState *t, int idx) {
    return curand_normal(t);
}
#else
static double randn(void) {
    double x = (double)rand() / RAND_MAX;
    double y = (double)rand() / RAND_MAX;
    return sqrt(-2 * log(x)) * cos(2 * M_PI * y);
}
#endif

#endif
