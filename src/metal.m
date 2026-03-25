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
    id<MTLComputePipelineState> reluPipeline;
    id<MTLComputePipelineState> addBias2DPipeline;
    id<MTLComputePipelineState> matmulPipeline;
    id<MTLComputePipelineState> conv2dPipeline;
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
         "kernel void relu_f32(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < count) { float v = input[gid]; out[gid] = v > 0.0f ? v : 0.0f; } }\n"
         "kernel void add_bias_2d_f32(device const float* input [[buffer(0)]], device const float* bias [[buffer(1)]], device float* out [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) { uint col = gid.x; uint row = gid.y; if (row >= rows || col >= cols) return; out[row * cols + col] = input[row * cols + col] + bias[col]; }\n"
         "kernel void matmul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& p [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) { uint row = gid.y; uint col = gid.x; if (row >= m || col >= p) return; float total = 0.0f; for (uint k = 0; k < n; ++k) { total += a[row * n + k] * b[k * p + col]; } out[row * p + col] = total; }\n"
         "kernel void conv2d_f32(device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* bias [[buffer(2)]], device float* out [[buffer(3)]], constant uint& batch [[buffer(4)]], constant uint& inChannels [[buffer(5)]], constant uint& inHeight [[buffer(6)]], constant uint& inWidth [[buffer(7)]], constant uint& outChannels [[buffer(8)]], constant uint& outHeight [[buffer(9)]], constant uint& outWidth [[buffer(10)]], constant uint& kernelHeight [[buffer(11)]], constant uint& kernelWidth [[buffer(12)]], constant uint& hasBias [[buffer(13)]], constant uint& applyRelu [[buffer(14)]], uint3 gid [[thread_position_in_grid]]) { uint ox = gid.x; uint oy = gid.y; uint z = gid.z; if (ox >= outWidth || oy >= outHeight || z >= batch * outChannels) return; uint n = z / outChannels; uint oc = z % outChannels; float total = hasBias != 0 ? bias[oc] : 0.0f; uint inputBatchOffset = n * inChannels * inHeight * inWidth; uint weightOutOffset = oc * inChannels * kernelHeight * kernelWidth; for (uint ic = 0; ic < inChannels; ++ic) { uint inputChannelOffset = inputBatchOffset + ic * inHeight * inWidth; uint weightChannelOffset = weightOutOffset + ic * kernelHeight * kernelWidth; for (uint ky = 0; ky < kernelHeight; ++ky) { uint inputRowOffset = inputChannelOffset + (oy + ky) * inWidth + ox; uint weightRowOffset = weightChannelOffset + ky * kernelWidth; for (uint kx = 0; kx < kernelWidth; ++kx) { total += input[inputRowOffset + kx] * weight[weightRowOffset + kx]; } } } if (applyRelu != 0 && total < 0.0f) total = 0.0f; out[((n * outChannels + oc) * outHeight + oy) * outWidth + ox] = total; }\n";
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
        !tpMetalBuildPipeline(backend, @"relu_f32", &backend->reluPipeline) ||
        !tpMetalBuildPipeline(backend, @"add_bias_2d_f32", &backend->addBias2DPipeline) ||
        !tpMetalBuildPipeline(backend, @"matmul_f32", &backend->matmulPipeline) ||
        !tpMetalBuildPipeline(backend, @"conv2d_f32", &backend->conv2dPipeline)) {
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

bool tpMetalReluF32(TPMetalBackend* backend,
                    TPMetalBuffer* input,
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
    [encoder setComputePipelineState:backend->reluPipeline];
    [encoder setBuffer:input->handle offset:0 atIndex:0];
    [encoder setBuffer:out->handle offset:0 atIndex:1];
    [encoder setBytes:&countValue length:sizeof(uint32_t) atIndex:2];

    width = backend->reluPipeline.maxTotalThreadsPerThreadgroup;
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

bool tpMetalAddBias2DF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         TPMetalBuffer* bias,
                         TPMetalBuffer* out,
                         int rows,
                         int cols) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    NSUInteger width;
    NSUInteger height;
    uint32_t rowsValue = (uint32_t)rows;
    uint32_t colsValue = (uint32_t)cols;

    if (backend == NULL || !backend->available || input == NULL || bias == NULL || out == NULL) {
        return false;
    }
    if (input->handle == nil || bias->handle == nil || out->handle == nil) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:backend->addBias2DPipeline];
    [encoder setBuffer:input->handle offset:0 atIndex:0];
    [encoder setBuffer:bias->handle offset:0 atIndex:1];
    [encoder setBuffer:out->handle offset:0 atIndex:2];
    [encoder setBytes:&rowsValue length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&colsValue length:sizeof(uint32_t) atIndex:4];

    width = backend->addBias2DPipeline.threadExecutionWidth;
    if (width == 0) {
        width = 8;
    }
    if (width > (NSUInteger)cols) {
        width = (NSUInteger)cols;
    }
    if (width == 0) {
        width = 1;
    }
    height = backend->addBias2DPipeline.maxTotalThreadsPerThreadgroup / width;
    if (height == 0) {
        height = 1;
    }
    if (height > (NSUInteger)rows) {
        height = (NSUInteger)rows;
    }

    gridSize = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
    threadgroupSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
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

bool tpMetalConv2dF32(TPMetalBackend* backend,
                      TPMetalBuffer* input,
                      TPMetalBuffer* weight,
                      TPMetalBuffer* bias,
                      TPMetalBuffer* out,
                      int batch,
                      int inChannels,
                      int inHeight,
                      int inWidth,
                      int outChannels,
                      int outHeight,
                      int outWidth,
                      int kernelHeight,
                      int kernelWidth,
                      int applyRelu) {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    MTLSize gridSize;
    MTLSize threadgroupSize;
    NSUInteger width;
    NSUInteger height;
    uint32_t batchValue = (uint32_t)batch;
    uint32_t inChannelsValue = (uint32_t)inChannels;
    uint32_t inHeightValue = (uint32_t)inHeight;
    uint32_t inWidthValue = (uint32_t)inWidth;
    uint32_t outChannelsValue = (uint32_t)outChannels;
    uint32_t outHeightValue = (uint32_t)outHeight;
    uint32_t outWidthValue = (uint32_t)outWidth;
    uint32_t kernelHeightValue = (uint32_t)kernelHeight;
    uint32_t kernelWidthValue = (uint32_t)kernelWidth;
    uint32_t hasBiasValue = bias != NULL && bias->handle != nil ? 1u : 0u;
    uint32_t applyReluValue = (uint32_t)applyRelu;

    if (backend == NULL || !backend->available || input == NULL || weight == NULL || out == NULL) {
        return false;
    }
    if (input->handle == nil || weight->handle == nil || out->handle == nil) {
        return false;
    }

    commandBuffer = [backend->queue commandBuffer];
    encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:backend->conv2dPipeline];
    [encoder setBuffer:input->handle offset:0 atIndex:0];
    [encoder setBuffer:weight->handle offset:0 atIndex:1];
    [encoder setBuffer:bias != NULL ? bias->handle : nil offset:0 atIndex:2];
    [encoder setBuffer:out->handle offset:0 atIndex:3];
    [encoder setBytes:&batchValue length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&inChannelsValue length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&inHeightValue length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&inWidthValue length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&outChannelsValue length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&outHeightValue length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&outWidthValue length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&kernelHeightValue length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&kernelWidthValue length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&hasBiasValue length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&applyReluValue length:sizeof(uint32_t) atIndex:14];

    width = backend->conv2dPipeline.threadExecutionWidth;
    if (width == 0) {
        width = 8;
    }
    if (width > (NSUInteger)outWidth) {
        width = (NSUInteger)outWidth;
    }
    if (width == 0) {
        width = 1;
    }
    height = backend->conv2dPipeline.maxTotalThreadsPerThreadgroup / width;
    if (height == 0) {
        height = 1;
    }
    if (height > (NSUInteger)outHeight) {
        height = (NSUInteger)outHeight;
    }

    gridSize = MTLSizeMake((NSUInteger)outWidth, (NSUInteger)outHeight, (NSUInteger)(batch * outChannels));
    threadgroupSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}
