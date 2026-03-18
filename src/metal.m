#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

#include "tensorpy/metal.h"

struct TPMetalBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> fillPipeline;
    id<MTLComputePipelineState> addPipeline;
    id<MTLComputePipelineState> mulPipeline;
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
         "kernel void mul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) out[gid] = a[gid] * b[gid]; }\n";
    TPMetalBackend* backend = (TPMetalBackend*)calloc(1, sizeof(TPMetalBackend));
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
        !tpMetalBuildPipeline(backend, @"mul_f32", &backend->mulPipeline)) {
        return backend;
    }

    backend->available = true;
    backend->lastError[0] = '\0';
    return backend;
}

void tpMetalBackendDestroy(TPMetalBackend* backend) {
    if (backend != NULL) {
        free(backend);
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

    buffer = (TPMetalBuffer*)calloc(1, sizeof(TPMetalBuffer));
    if (buffer == NULL) {
        return NULL;
    }

    buffer->handle = [backend->device newBufferWithLength:byteLength options:MTLResourceStorageModeShared];
    if (buffer->handle == nil) {
        free(buffer);
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
        free(buffer);
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

bool tpMetalAddF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    float* aData;
    float* bData;
    float* outData;
    int i;

    if (backend == NULL || !backend->available || a == NULL || b == NULL || out == NULL) {
        return false;
    }

    aData = (float*)[a->handle contents];
    bData = (float*)[b->handle contents];
    outData = (float*)[out->handle contents];
    for (i = 0; i < count; i++) {
        outData[i] = aData[i] + bData[i];
    }
    return true;
}

bool tpMetalMulF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    float* aData;
    float* bData;
    float* outData;
    int i;

    if (backend == NULL || !backend->available || a == NULL || b == NULL || out == NULL) {
        return false;
    }

    aData = (float*)[a->handle contents];
    bData = (float*)[b->handle contents];
    outData = (float*)[out->handle contents];
    for (i = 0; i < count; i++) {
        outData[i] = aData[i] * bData[i];
    }
    return true;
}
