#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TP_HAS_NEON 1
#else
#define TP_HAS_NEON 0
#endif

#include "tensorpy/compute.h"

typedef struct {
    float* out;
    const float* a;
    const float* b;
    int start;
    int count;
    float scalar;
} TPUnaryBinaryTask;

typedef struct {
    const float* values;
    const float* a;
    const float* b;
    float* partials;
    int baseStart;
    int grainSize;
} TPReduceTask;

static int maxInt(int a, int b) {
    return a > b ? a : b;
}

static int minInt(int a, int b) {
    return a < b ? a : b;
}

int tpComputeSimdWidthFloats(void) {
#if TP_HAS_NEON
    return 4;
#else
    return 1;
#endif
}

bool tpComputeSimdAvailable(void) {
    return tpComputeSimdWidthFloats() > 1;
}

int tpComputeDefaultParallelThreshold(void) {
    return 1 << 14;
}

bool tpComputeContextInit(TPComputeContext* context, int threadCount, int parallelThreshold) {
    if (context == NULL) {
        return false;
    }

    memset(context, 0, sizeof(*context));
    context->simdWidthFloats = tpComputeSimdWidthFloats();
    context->parallelThreshold = parallelThreshold > 0
        ? parallelThreshold
        : tpComputeDefaultParallelThreshold();

    if (threadCount <= 0) {
        threadCount = platformHardwareThreadCount();
    }

    context->threadCount = threadCount;
    context->pool = platformThreadPoolCreate(threadCount);
    if (context->pool == NULL) {
        context->threadCount = 0;
        return false;
    }

    context->threadCount = platformThreadPoolThreadCount(context->pool);
    context->ownsPool = true;
    return true;
}

void tpComputeContextDestroy(TPComputeContext* context) {
    if (context == NULL) {
        return;
    }

    if (context->ownsPool && context->pool != NULL) {
        platformThreadPoolDestroy(context->pool);
    }

    memset(context, 0, sizeof(*context));
}

TPComputePath tpComputeResolvePath(const TPComputeContext* context,
                                   TPComputeMode mode,
                                   int count) {
    bool simdAvailable = tpComputeSimdAvailable();
    bool threadedAvailable = context != NULL &&
                             context->pool != NULL &&
                             context->threadCount > 1;
    int threshold = context != NULL ? context->parallelThreshold : tpComputeDefaultParallelThreshold();

    if (mode == TP_COMPUTE_MODE_SCALAR) {
        return TP_COMPUTE_PATH_SCALAR;
    }

    if (mode == TP_COMPUTE_MODE_SIMD) {
        return simdAvailable ? TP_COMPUTE_PATH_SIMD : TP_COMPUTE_PATH_SCALAR;
    }

    if (mode == TP_COMPUTE_MODE_THREADED) {
        if (threadedAvailable) {
            return TP_COMPUTE_PATH_THREADED;
        }
        return simdAvailable ? TP_COMPUTE_PATH_SIMD : TP_COMPUTE_PATH_SCALAR;
    }

    if (threadedAvailable && count >= threshold) {
        return TP_COMPUTE_PATH_THREADED;
    }

    if (simdAvailable && count >= tpComputeSimdWidthFloats() * 2) {
        return TP_COMPUTE_PATH_SIMD;
    }

    return TP_COMPUTE_PATH_SCALAR;
}

const char* tpComputePathName(TPComputePath path) {
    switch (path) {
        case TP_COMPUTE_PATH_SCALAR:
            return "scalar";
        case TP_COMPUTE_PATH_SIMD:
            return "simd";
        case TP_COMPUTE_PATH_THREADED:
            return "threaded";
    }
    return "unknown";
}

static void tpFillScalarF32(float* out, int count, float value) {
    int i;
    for (i = 0; i < count; i++) {
        out[i] = value;
    }
}

static void tpAddScalarF32(const float* a, const float* b, float* out, int count) {
    int i;
    for (i = 0; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

static void tpMulScalarF32(const float* a, const float* b, float* out, int count) {
    int i;
    for (i = 0; i < count; i++) {
        out[i] = a[i] * b[i];
    }
}

static float tpSumScalarF32(const float* values, int count) {
    float total = 0.0f;
    int i;
    for (i = 0; i < count; i++) {
        total += values[i];
    }
    return total;
}

static float tpDotScalarF32(const float* a, const float* b, int count) {
    float total = 0.0f;
    int i;
    for (i = 0; i < count; i++) {
        total += a[i] * b[i];
    }
    return total;
}

#if TP_HAS_NEON
static void tpFillSimdF32(float* out, int count, float value) {
    int i = 0;
    float32x4_t lane = vdupq_n_f32(value);
    for (; i + 4 <= count; i += 4) {
        vst1q_f32(out + i, lane);
    }
    for (; i < count; i++) {
        out[i] = value;
    }
}

static void tpAddSimdF32(const float* a, const float* b, float* out, int count) {
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(av, bv));
    }
    for (; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

static void tpMulSimdF32(const float* a, const float* b, float* out, int count) {
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(av, bv));
    }
    for (; i < count; i++) {
        out[i] = a[i] * b[i];
    }
}

static float tpSumSimdF32(const float* values, int count) {
    int i = 0;
    float total = 0.0f;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (; i + 4 <= count; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(values + i));
    }

    total += vaddvq_f32(acc);
    for (; i < count; i++) {
        total += values[i];
    }
    return total;
}

static float tpDotSimdF32(const float* a, const float* b, int count) {
    int i = 0;
    float total = 0.0f;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (; i + 4 <= count; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    total += vaddvq_f32(acc);
    for (; i < count; i++) {
        total += a[i] * b[i];
    }
    return total;
}
#else
static void tpFillSimdF32(float* out, int count, float value) {
    tpFillScalarF32(out, count, value);
}

static void tpAddSimdF32(const float* a, const float* b, float* out, int count) {
    tpAddScalarF32(a, b, out, count);
}

static void tpMulSimdF32(const float* a, const float* b, float* out, int count) {
    tpMulScalarF32(a, b, out, count);
}

static float tpSumSimdF32(const float* values, int count) {
    return tpSumScalarF32(values, count);
}

static float tpDotSimdF32(const float* a, const float* b, int count) {
    return tpDotScalarF32(a, b, count);
}
#endif

static void tpFillRangeTask(int start, int end, void* context) {
    TPUnaryBinaryTask* task = (TPUnaryBinaryTask*)context;
    tpFillSimdF32(task->out + start, end - start, task->scalar);
}

static void tpAddRangeTask(int start, int end, void* context) {
    TPUnaryBinaryTask* task = (TPUnaryBinaryTask*)context;
    tpAddSimdF32(task->a + start, task->b + start, task->out + start, end - start);
}

static void tpMulRangeTask(int start, int end, void* context) {
    TPUnaryBinaryTask* task = (TPUnaryBinaryTask*)context;
    tpMulSimdF32(task->a + start, task->b + start, task->out + start, end - start);
}

static void tpSumRangeTask(int start, int end, void* context) {
    TPReduceTask* task = (TPReduceTask*)context;
    int chunk = (start - task->baseStart) / task->grainSize;
    task->partials[chunk] = tpSumSimdF32(task->values + start, end - start);
}

static void tpDotRangeTask(int start, int end, void* context) {
    TPReduceTask* task = (TPReduceTask*)context;
    int chunk = (start - task->baseStart) / task->grainSize;
    task->partials[chunk] = tpDotSimdF32(task->a + start, task->b + start, end - start);
}

static int tpThreadedGrainSize(const TPComputeContext* context, int count) {
    int threads = context != NULL ? maxInt(context->threadCount, 1) : 1;
    int grain = count / (threads * 4);
    if (grain < 256) {
        grain = 256;
    }
    return minInt(maxInt(grain, 1), maxInt(count, 1));
}

static bool tpRunParallel(const TPComputeContext* context,
                          int count,
                          PlatformParallelForFunction function,
                          void* taskContext) {
    int grain;

    if (context == NULL || context->pool == NULL || context->threadCount <= 1) {
        function(0, count, taskContext);
        return true;
    }

    grain = tpThreadedGrainSize(context, count);
    return platformThreadPoolParallelFor(context->pool, 0, count, grain, function, taskContext);
}

bool tpComputeFillF32(TPComputeContext* context,
                      float* out,
                      int count,
                      float value,
                      TPComputeMode mode,
                      TPComputePath* outPath) {
    TPComputePath path;
    TPUnaryBinaryTask task;

    if (out == NULL || count < 0) {
        return false;
    }

    path = tpComputeResolvePath(context, mode, count);
    if (outPath != NULL) {
        *outPath = path;
    }

    if (count == 0) {
        return true;
    }

    if (path == TP_COMPUTE_PATH_THREADED) {
        task.out = out;
        task.scalar = value;
        return tpRunParallel(context, count, tpFillRangeTask, &task);
    }

    if (path == TP_COMPUTE_PATH_SIMD) {
        tpFillSimdF32(out, count, value);
    } else {
        tpFillScalarF32(out, count, value);
    }
    return true;
}

bool tpComputeAddF32(TPComputeContext* context,
                     const float* a,
                     const float* b,
                     float* out,
                     int count,
                     TPComputeMode mode,
                     TPComputePath* outPath) {
    TPComputePath path;
    TPUnaryBinaryTask task;

    if (a == NULL || b == NULL || out == NULL || count < 0) {
        return false;
    }

    path = tpComputeResolvePath(context, mode, count);
    if (outPath != NULL) {
        *outPath = path;
    }

    if (count == 0) {
        return true;
    }

    if (path == TP_COMPUTE_PATH_THREADED) {
        task.a = a;
        task.b = b;
        task.out = out;
        return tpRunParallel(context, count, tpAddRangeTask, &task);
    }

    if (path == TP_COMPUTE_PATH_SIMD) {
        tpAddSimdF32(a, b, out, count);
    } else {
        tpAddScalarF32(a, b, out, count);
    }
    return true;
}

bool tpComputeMulF32(TPComputeContext* context,
                     const float* a,
                     const float* b,
                     float* out,
                     int count,
                     TPComputeMode mode,
                     TPComputePath* outPath) {
    TPComputePath path;
    TPUnaryBinaryTask task;

    if (a == NULL || b == NULL || out == NULL || count < 0) {
        return false;
    }

    path = tpComputeResolvePath(context, mode, count);
    if (outPath != NULL) {
        *outPath = path;
    }

    if (count == 0) {
        return true;
    }

    if (path == TP_COMPUTE_PATH_THREADED) {
        task.a = a;
        task.b = b;
        task.out = out;
        return tpRunParallel(context, count, tpMulRangeTask, &task);
    }

    if (path == TP_COMPUTE_PATH_SIMD) {
        tpMulSimdF32(a, b, out, count);
    } else {
        tpMulScalarF32(a, b, out, count);
    }
    return true;
}

bool tpComputeSumF32(TPComputeContext* context,
                     const float* values,
                     int count,
                     float* out,
                     TPComputeMode mode,
                     TPComputePath* outPath) {
    TPComputePath path;

    if (values == NULL || out == NULL || count < 0) {
        return false;
    }

    path = tpComputeResolvePath(context, mode, count);
    if (outPath != NULL) {
        *outPath = path;
    }

    if (count == 0) {
        *out = 0.0f;
        return true;
    }

    if (path == TP_COMPUTE_PATH_THREADED) {
        int grain = tpThreadedGrainSize(context, count);
        int chunkCount = (count + grain - 1) / grain;
        TPReduceTask task;
        float* partials = (float*)calloc((size_t)chunkCount, sizeof(float));
        int i;

        if (partials == NULL) {
            return false;
        }

        task.values = values;
        task.partials = partials;
        task.baseStart = 0;
        task.grainSize = grain;
        if (!platformThreadPoolParallelFor(context->pool, 0, count, grain, tpSumRangeTask, &task)) {
            free(partials);
            return false;
        }

        *out = 0.0f;
        for (i = 0; i < chunkCount; i++) {
            *out += partials[i];
        }
        free(partials);
        return true;
    }

    *out = path == TP_COMPUTE_PATH_SIMD
        ? tpSumSimdF32(values, count)
        : tpSumScalarF32(values, count);
    return true;
}

bool tpComputeDotF32(TPComputeContext* context,
                     const float* a,
                     const float* b,
                     int count,
                     float* out,
                     TPComputeMode mode,
                     TPComputePath* outPath) {
    TPComputePath path;

    if (a == NULL || b == NULL || out == NULL || count < 0) {
        return false;
    }

    path = tpComputeResolvePath(context, mode, count);
    if (outPath != NULL) {
        *outPath = path;
    }

    if (count == 0) {
        *out = 0.0f;
        return true;
    }

    if (path == TP_COMPUTE_PATH_THREADED) {
        int grain = tpThreadedGrainSize(context, count);
        int chunkCount = (count + grain - 1) / grain;
        TPReduceTask task;
        float* partials = (float*)calloc((size_t)chunkCount, sizeof(float));
        int i;

        if (partials == NULL) {
            return false;
        }

        task.a = a;
        task.b = b;
        task.partials = partials;
        task.baseStart = 0;
        task.grainSize = grain;
        if (!platformThreadPoolParallelFor(context->pool, 0, count, grain, tpDotRangeTask, &task)) {
            free(partials);
            return false;
        }

        *out = 0.0f;
        for (i = 0; i < chunkCount; i++) {
            *out += partials[i];
        }
        free(partials);
        return true;
    }

    *out = path == TP_COMPUTE_PATH_SIMD
        ? tpDotSimdF32(a, b, count)
        : tpDotScalarF32(a, b, count);
    return true;
}
