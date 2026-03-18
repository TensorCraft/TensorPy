#include "tensorpy/metal.h"

struct TPMetalBackend {
    int unused;
};

struct TPMetalBuffer {
    int unused;
};

TPMetalBackend* tpMetalBackendCreate(void) {
    return NULL;
}

void tpMetalBackendDestroy(TPMetalBackend* backend) {
    (void)backend;
}

bool tpMetalBackendIsAvailable(TPMetalBackend* backend) {
    (void)backend;
    return false;
}

const char* tpMetalBackendLastError(TPMetalBackend* backend) {
    (void)backend;
    return "Metal support is disabled at build time.";
}

TPMetalBuffer* tpMetalBufferCreate(TPMetalBackend* backend,
                                   size_t byteLength,
                                   const void* initialBytes) {
    (void)backend;
    (void)byteLength;
    (void)initialBytes;
    return NULL;
}

void tpMetalBufferDestroy(TPMetalBuffer* buffer) {
    (void)buffer;
}

bool tpMetalBufferWrite(TPMetalBuffer* buffer, const void* bytes, size_t byteLength) {
    (void)buffer;
    (void)bytes;
    (void)byteLength;
    return false;
}

bool tpMetalBufferRead(TPMetalBuffer* buffer, void* outBytes, size_t byteLength) {
    (void)buffer;
    (void)outBytes;
    (void)byteLength;
    return false;
}

bool tpMetalFillF32(TPMetalBackend* backend,
                    TPMetalBuffer* out,
                    float value,
                    int count) {
    (void)backend;
    (void)out;
    (void)value;
    (void)count;
    return false;
}

bool tpMetalAddF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    (void)backend;
    (void)a;
    (void)b;
    (void)out;
    (void)count;
    return false;
}

bool tpMetalMulF32(TPMetalBackend* backend,
                   TPMetalBuffer* a,
                   TPMetalBuffer* b,
                   TPMetalBuffer* out,
                   int count) {
    (void)backend;
    (void)a;
    (void)b;
    (void)out;
    (void)count;
    return false;
}

bool tpMetalAddScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count) {
    (void)backend;
    (void)input;
    (void)scalar;
    (void)out;
    (void)count;
    return false;
}

bool tpMetalMulScalarF32(TPMetalBackend* backend,
                         TPMetalBuffer* input,
                         float scalar,
                         TPMetalBuffer* out,
                         int count) {
    (void)backend;
    (void)input;
    (void)scalar;
    (void)out;
    (void)count;
    return false;
}

bool tpMetalMatmulF32(TPMetalBackend* backend,
                      TPMetalBuffer* a,
                      TPMetalBuffer* b,
                      TPMetalBuffer* out,
                      int m,
                      int n,
                      int p) {
    (void)backend;
    (void)a;
    (void)b;
    (void)out;
    (void)m;
    (void)n;
    (void)p;
    return false;
}
