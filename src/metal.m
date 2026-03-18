#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

#include "tensorpy/metal.h"
#include "tensorpy/memory.h"

struct TPMetalBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> fillPipeline;
    id<MTLComputePipelineState> addPipeline;
    id<MTLComputePipelineState> mulPipeline;
    id<MTLComputePipelineState> addScalarPipeline;
    id<MTLComputePipelineState> mulScalarPipeline;
    id<MTLComputePipelineState> matmulPipeline;
    char lastError[256];
    bool available;
};

struct TPMetalBuffer {
    id<MTLBuffer> handle;
    size_t byteLength;
};

static void tpMetalSetError(TPMetalBackend* backend, const char* message) {
    size_t length;
    if (backend == NULL || message == NULL) {
        return;
    }
    length = strlen(message);
    if (length >= sizeof(backend->lastError)) {
        length = sizeof(backend->lastError) - 1;
    }
    memcpy(backend->lastError, message, length);
    backend->lastError[length] = '\0';
}

static bool tpMetalBuildPipeline(TPMetalBackend* backend,
                                 NSString* name,
                                 id<MTLComputePipelineState>* outPipeline) {
    NSError* error = nil;
    id<MTLFunction> function = [backend->library newFunctionWithName:name];
    if (function == nil) {
        tpMetalSetError(backend, "failed to find Metal kernel");
        return false;
    }

    *outPipeline = [backend->device newComputePipelineStateWithFunction:function error:&error];
    if (*outPipeline == nil) {
        tpMetalSetError(backend, error != nil ? [[error localizedDescription] UTF8String] : "failed to create Metal pipeline");
        return false;
    }
    return true;
}

TPMetalBackend* tpMetalBackendCreate(void) {
    static NSString* source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void fill_f32(device float* out [[buffer(0)]], constant float& value [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = value; }\n"
         "kernel void add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = a[gid] + b[gid]; }\n"
         "kernel void mul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = a[gid] * b[gid]; }\n"
         "kernel void add_scalar_f32(device const float* input [[buffer(0)]], constant float& scalar [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = input[gid] + scalar; }\n"
         "kernel void mul_scalar_f32(device const float* input [[buffer(0)]], constant float& scalar [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = input[gid] * scalar; }\n"
         "kernel void matmul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& p [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) { uint row = gid.y; uint col = gid.x; if (row >= m || col >= p) return; float total = 0.0f; for (uint k = 0; k < n; ++k) { total += a[row * n + k] * b[k * p + col]; } out[row * p + col] = total; }\n";
    TPMetalBackend* backend = (TPMetalBackend*)tpMemCalloc(1, sizeof(TPMetalBackend));
    NSError* error = nil;

    if (backend == NULL) {
        return NULL;
    }

    backend->device = MTLCreateSystemDefaultDevice();
    if (backend->device == nil) {
        tpMetalSetError(backend, "Metal device unavailable");
        return backend;
    }

    backend->queue = [backend->device newCommandQueue];
    if (backend->queue == nil) {
        tpMetalSetError(backend, "failed to create Metal command queue");
        return backend;
    }

    backend->library = [backend->device newLibraryWithSource:source options:nil error:&error];
    if (backend->library == nil) {
        tpMetalSetError(backend, error != nil ? [[error localizedDescription] UTF8String] : "failed to compile Metal library");
        return backend;
    }

    if (!tpMetalBuildPipeline(backend, @"fill_f32", &backend->fillPipeline) ||
        !tpMetalBuildPipeline(backend, @"add_f32", &backend->addPipeline) ||
        !tpMetalBuildPipeline(backend, @"mul_f32", &backend->mulPipeline) ||
        !tpMetalBuildPipeline(backend, @"add_scalar_f32", &backend->addScalarPipeline) ||
        !tpMetalBuildPipeline(backend, @"mul_scalar_f32", &backend->mulScalarPipeline) ||
        !tpMetalBuildPipeline(backend, @"matmul_f32", &backend->matmulPipeline)) {
        return backend;
    }

    backend->available = true;
    backend->lastError[0] = '\0';
    return backend;
}

void tpMetalBackendDestroy(TPMetalBackend* backend) {
    if (backend != NULL) {
        tpMemFree(backend);
    }
}

bool tpMetalBackendIsAvailable(TPMetalBackend* backend) {
    return backend != NULL && backend->available;
}

const char* tpMetalBackendLastError(TPMetalBackend* backend) {
    if (backend == NULL || backend->lastError[0] == '\0') {
        return NULL;
    }
    return backend->lastError;
}

TPMetalBuffer* tpMetalBufferCreate(TPMetalBackend* backend,
                                   size_t byteLength,
                                   const void* initialBytes) {
    TPMetalBuffer* buffer;

    if (backend == NULL || !backend->available) {
        return NULL;
    }

    buffer = (TPMetalBuffer*)tpMemCalloc(1, sizeof(TPMetalBuffer));
    if (buffer == NULL) {
        return NULL;
    }

    buffer->handle = [backend->device newBufferWithLength:byteLength options:MTLResourceStorageModeShared];
    if (buffer->handle == nil) {
        tpMemFree(buffer);
        return NULL;
    }
    buffer->byteLength = byteLength;

    if (initialBytes != NULL && byteLength > 0) {
        memcpy([buffer->handle contents], initialBytes, byteLength);
    }
    return buffer;
}

void tpMetalBufferDestroy(TPMetalBuffer* buffer) {
    if (buffer != NULL) {
        tpMemFree(buffer);
    }
}

bool tpMetalBufferWrite(TPMetalBuffer* buffer, const void* bytes, size_t byteLength) {
    if (buffer == NULL || buffer->handle == nil || byteLength > buffer->byteLength) {
        return false;
    }
    if (byteLength > 0 && bytes != NULL) {
        memcpy([buffer->handle contents], bytes, byteLength);
    }
    return true;
}

bool tpMetalBufferRead(TPMetalBuffer* buffer, void* outBytes, size_t byteLength) {
    if (buffer == NULL || buffer->handle == nil || outBytes == NULL || byteLength > buffer->byteLength) {
        return false;
    }
    if (byteLength > 0) {
        memcpy(outBytes, [buffer->handle contents], byteLength);
    }
    return true;
}

bool tpMetalFillF32(TPMetalBackend* backend,
                    TPMetalBuffer* out,
                    float value,
                    int count) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    NSUInteger width;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    uint32_t countValue = (uint32_t)count;

    if (backend == NULL || !backend->available || out == NULL || out->handle == nil) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:backend->fillPipeline];
    [encoder setBuffer:out->handle offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof(float) atIndex:1];
    [encoder setBytes:&countValue length:sizeof(uint32_t) atIndex:2];

    width = backend->fillPipeline.maxTotalThreadsPerThreadgroup;
    if (width > (NSUInteger)count && count > 0) {
        width = (NSUInteger)count;
    }
    if (width == 0) {
        width = 1;
    }
    gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    threadgroupSize = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}

static bool tpMetalRunBinaryF32(TPMetalBackend* backend,
                                id<MTLComputePipelineState> pipeline,
                                TPMetalBuffer* a,
                                TPMetalBuffer* b,
                                TPMetalBuffer* out,
                                int count) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    NSUInteger width;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    uint32_t countValue = (uint32_t)count;

    if (backend == NULL || !backend->available || a == NULL || b == NULL || out == NULL) {
        return false;
    }
    if (a->handle == nil || b->handle == nil || out->handle == nil) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a->handle offset:0 atIndex:0];
    [encoder setBuffer:b->handle offset:0 atIndex:1];
    [encoder setBuffer:out->handle offset:0 atIndex:2];
    [encoder setBytes:&countValue length:sizeof(uint32_t) atIndex:3];

    width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > (NSUInteger)count && count > 0) {
        width = (NSUInteger)count;
    }
    if (width == 0) {
        width = 1;
    }
    gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    threadgroupSize = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}

static bool tpMetalRunScalarBinaryF32(TPMetalBackend* backend,
                                      id<MTLComputePipelineState> pipeline,
                                      TPMetalBuffer* input,
                                      float scalar,
                                      TPMetalBuffer* out,
                                      int count) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    NSUInteger width;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    uint32_t countValue = (uint32_t)count;

    if (backend == NULL || !backend->available || input == NULL || out == NULL) {
        return false;
    }
    if (input->handle == nil || out->handle == nil) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input->handle offset:0 atIndex:0];
    [encoder setBytes:&scalar length:sizeof(float) atIndex:1];
    [encoder setBuffer:out->handle offset:0 atIndex:2];
    [encoder setBytes:&countValue length:sizeof(uint32_t) atIndex:3];

    width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > (NSUInteger)count && count > 0) {
        width = (NSUInteger)count;
    }
    if (width == 0) {
        width = 1;
    }
    gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    threadgroupSize = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}

bool tpMetalAddF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    return tpMetalRunBinaryF32(backend, backend->addPipeline, a, b, out, count);
}

bool tpMetalMulF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    return tpMetalRunBinaryF32(backend, backend->mulPipeline, a, b, out, count);
}

bool tpMetalAddScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count) {
    return tpMetalRunScalarBinaryF32(backend, backend->addScalarPipeline, input, scalar, out, count);
}

bool tpMetalMulScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count) {
    return tpMetalRunScalarBinaryF32(backend, backend->mulScalarPipeline, input, scalar, out, count);
}

bool tpMetalMatmulF32(TPMetalBackend* backend,
                      TPMetalBuffer* a,
                      TPMetalBuffer* b,
                      TPMetalBuffer* out,
                      int m,
                      int n,
                      int p) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    NSUInteger width;
    NSUInteger height;
    uint32_t mValue = (uint32_t)m;
    uint32_t nValue = (uint32_t)n;
    uint32_t pValue = (uint32_t)p;

    if (backend == NULL || !backend->available || a == NULL || b == NULL || out == NULL) {
        return false;
    }
    if (a->handle == nil || b->handle == nil || out->handle == nil) {
        return false;
    }
    if (m <= 0 || n <= 0 || p <= 0) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:backend->matmulPipeline];
    [encoder setBuffer:a->handle offset:0 atIndex:0];
    [encoder setBuffer:b->handle offset:0 atIndex:1];
    [encoder setBuffer:out->handle offset:0 atIndex:2];
    [encoder setBytes:&mValue length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&nValue length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&pValue length:sizeof(uint32_t) atIndex:5];

    width = backend->matmulPipeline.threadExecutionWidth;
    if (width == 0) {
        width = 8;
    }
    if (width > (NSUInteger)p) {
        width = (NSUInteger)p;
    }
    if (width == 0) {
        width = 1;
    }
    height = backend->matmulPipeline.maxTotalThreadsPerThreadgroup / width;
    if (height == 0) {
        height = 1;
    }
    if (height > (NSUInteger)m) {
        height = (NSUInteger)m;
    }

    gridSize = MTLSizeMake((NSUInteger)p, (NSUInteger)m, 1);
    threadgroupSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}
