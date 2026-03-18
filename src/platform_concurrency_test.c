#include <stdio.h>
#include <stdlib.h>

#include "tensorpy/platform.h"

typedef struct {
    PlatformAtomicInt total;
} CounterContext;

typedef struct {
    CounterContext* counter;
    int iterations;
} IncrementTaskContext;

typedef struct {
    PlatformAtomicInt total;
} ParallelForContext;

static void incrementTask(void* context) {
    IncrementTaskContext* task = (IncrementTaskContext*)context;
    int i;
    for (i = 0; i < task->iterations; i++) {
        platformAtomicFetchAdd(&task->counter->total, 1);
    }
}

static void sumRangeTask(int start, int end, void* context) {
    ParallelForContext* task = (ParallelForContext*)context;
    int i;
    for (i = start; i < end; i++) {
        platformAtomicFetchAdd(&task->total, i);
    }
}

int main(void) {
    PlatformThreadPool* pool;
    PlatformTaskHandle handles[8];
    IncrementTaskContext tasks[8];
    CounterContext counter;
    ParallelForContext parallel;
    int i;
    int64_t expectedSum = 0;
    const int rangeEnd = 1000;

    pool = platformThreadPoolCreate(4);
    if (pool == NULL) {
        fprintf(stderr, "failed to create thread pool\n");
        return 1;
    }

    platformAtomicInit(&counter.total, 0);
    for (i = 0; i < 8; i++) {
        tasks[i].counter = &counter;
        tasks[i].iterations = 1000;
        if (!platformThreadPoolSubmit(pool, incrementTask, &tasks[i], &handles[i])) {
            fprintf(stderr, "submit failed: %s\n", platformThreadPoolLastError(pool));
            platformThreadPoolDestroy(pool);
            return 1;
        }
    }

    for (i = 0; i < 8; i++) {
        platformTaskHandleWait(&handles[i]);
        platformTaskHandleDestroy(&handles[i]);
    }

    if (platformAtomicLoad(&counter.total) != 8000) {
        fprintf(stderr, "unexpected task total: %lld\n",
                (long long)platformAtomicLoad(&counter.total));
        platformThreadPoolDestroy(pool);
        return 1;
    }

    platformAtomicInit(&parallel.total, 0);
    if (!platformThreadPoolParallelFor(pool, 0, rangeEnd, 32, sumRangeTask, &parallel)) {
        fprintf(stderr, "parallel_for failed: %s\n", platformThreadPoolLastError(pool));
        platformThreadPoolDestroy(pool);
        return 1;
    }

    for (i = 0; i < rangeEnd; i++) {
        expectedSum += i;
    }

    if (platformAtomicLoad(&parallel.total) != expectedSum) {
        fprintf(stderr, "unexpected parallel sum: %lld expected %lld\n",
                (long long)platformAtomicLoad(&parallel.total),
                (long long)expectedSum);
        platformThreadPoolDestroy(pool);
        return 1;
    }

    platformThreadPoolDestroy(pool);
    printf("platform concurrency smoke test passed\n");
    return 0;
}
