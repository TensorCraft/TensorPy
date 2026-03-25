#ifndef TENSORPY_METAL_H
#define TENSORPY_METAL_H

#include "common.h"

#ifndef TP_ENABLE_METAL
#define TP_ENABLE_METAL 1
#endif

typedef struct TPMetalBackend TPMetalBackend;
typedef struct TPMetalBuffer TPMetalBuffer;

TPMetalBackend* tpMetalBackendCreate(void);
void tpMetalBackendDestroy(TPMetalBackend* backend);
bool tpMetalBackendIsAvailable(TPMetalBackend* backend);
const char* tpMetalBackendLastError(TPMetalBackend* backend);

TPMetalBuffer* tpMetalBufferCreate(TPMetalBackend* backend,
                                   size_t byteLength,
                                   const void* initialBytes);
void tpMetalBufferDestroy(TPMetalBuffer* buffer);
bool tpMetalBufferWrite(TPMetalBuffer* buffer, const void* bytes, size_t byteLength);
bool tpMetalBufferRead(TPMetalBuffer* buffer, void* outBytes, size_t byteLength);

bool tpMetalFillF32(TPMetalBackend* backend,
                    TPMetalBuffer* out,
                    float value,
                    int count);
bool tpMetalAddF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count);
bool tpMetalAddScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count);
bool tpMetalMulF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count);
bool tpMetalMulScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count);
bool tpMetalReluF32(TPMetalBackend* backend,
                    TPMetalBuffer* input,
                    TPMetalBuffer* out,
                    int count);
bool tpMetalAddBias2DF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         TPMetalBuffer* bias,
                         TPMetalBuffer* out,
                         int rows,
                         int cols);
bool tpMetalMatmulF32(TPMetalBackend* backend,
                      TPMetalBuffer* a,
                      TPMetalBuffer* b,
                      TPMetalBuffer* out,
                      int m,
                      int n,
                      int p);
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
                      int applyRelu);

#endif // TENSORPY_METAL_H
