#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorpy/compute.h"

static bool almostEqual(float a, float b) {
    float diff = fabsf(a - b);
    return diff < 1e-4f;
}

static bool assertPath(const char* label, TPComputePath got, TPComputePath expected) {
    if (got != expected) {
        fprintf(stderr, "%s path mismatch: got %s expected %s\n",
                label,
                tpComputePathName(got),
                tpComputePathName(expected));
        return false;
    }
    return true;
}

int main(void) {
    TPComputeContext compute;
    float* a;
    float* b;
    float* out;
    float sum;
    float dot;
    float expectedSum = 0.0f;
    float expectedDot = 0.0f;
    int count = 32768;
    int i;
    TPComputePath path;

    if (!tpComputeContextInit(&compute, 4, 1024)) {
        fprintf(stderr, "failed to initialize compute context\n");
        return 1;
    }

    a = (float*)malloc(sizeof(float) * (size_t)count);
    b = (float*)malloc(sizeof(float) * (size_t)count);
    out = (float*)malloc(sizeof(float) * (size_t)count);
    if (a == NULL || b == NULL || out == NULL) {
        fprintf(stderr, "failed to allocate buffers\n");
        tpComputeContextDestroy(&compute);
        free(a);
        free(b);
        free(out);
        return 1;
    }

    for (i = 0; i < count; i++) {
        a[i] = (float)(i % 13) * 0.5f;
        b[i] = (float)(i % 7) * 0.25f;
        expectedSum += a[i];
        expectedDot += a[i] * b[i];
    }

    if (!tpComputeFillF32(&compute, out, count, 3.5f, TP_COMPUTE_MODE_SCALAR, &path) ||
        !assertPath("fill-scalar", path, TP_COMPUTE_PATH_SCALAR)) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        if (!almostEqual(out[i], 3.5f)) {
            fprintf(stderr, "fill mismatch at %d\n", i);
            return 1;
        }
    }

    if (!tpComputeAddF32(&compute, a, b, out, count, TP_COMPUTE_MODE_SIMD, &path)) {
        fprintf(stderr, "simd add failed\n");
        return 1;
    }
    if (tpComputeSimdAvailable()) {
        if (!assertPath("add-simd", path, TP_COMPUTE_PATH_SIMD)) {
            return 1;
        }
    }
    for (i = 0; i < count; i++) {
        if (!almostEqual(out[i], a[i] + b[i])) {
            fprintf(stderr, "add mismatch at %d\n", i);
            return 1;
        }
    }

    if (!tpComputeMulF32(&compute, a, b, out, count, TP_COMPUTE_MODE_THREADED, &path) ||
        !assertPath("mul-threaded", path, TP_COMPUTE_PATH_THREADED)) {
        return 1;
    }
    for (i = 0; i < count; i++) {
        if (!almostEqual(out[i], a[i] * b[i])) {
            fprintf(stderr, "mul mismatch at %d\n", i);
            return 1;
        }
    }

    if (!tpComputeSumF32(&compute, a, count, &sum, TP_COMPUTE_MODE_THREADED, &path) ||
        !assertPath("sum-threaded", path, TP_COMPUTE_PATH_THREADED)) {
        return 1;
    }
    if (!almostEqual(sum, expectedSum)) {
        fprintf(stderr, "sum mismatch: %.6f expected %.6f\n", sum, expectedSum);
        return 1;
    }

    if (!tpComputeDotF32(&compute, a, b, count, &dot, TP_COMPUTE_MODE_THREADED, &path) ||
        !assertPath("dot-threaded", path, TP_COMPUTE_PATH_THREADED)) {
        return 1;
    }
    if (!almostEqual(dot, expectedDot)) {
        fprintf(stderr, "dot mismatch: %.6f expected %.6f\n", dot, expectedDot);
        return 1;
    }

    free(a);
    free(b);
    free(out);
    tpComputeContextDestroy(&compute);
    printf("compute cpu smoke test passed\n");
    return 0;
}
