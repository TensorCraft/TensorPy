#ifndef TENSORPY_API_H
#define TENSORPY_API_H

#include "common.h"

#define TENSORPY_API_VERSION 1
#define TENSORPY_EXTENSION_ABI_VERSION 1

typedef struct TPContext TPContext;
typedef struct TPModule TPModule;

typedef enum {
    TP_VALUE_NIL,
    TP_VALUE_BOOL,
    TP_VALUE_NUMBER,
    TP_VALUE_STRING,
    TP_VALUE_ERROR,
} TPValueType;

typedef enum {
    TP_OK = 0,
    TP_COMPILE_ERROR = 65,
    TP_RUNTIME_ERROR = 70,
} TPResult;

// TPValue is the scalar-only public exchange type for the Phase 1 embedding ABI.
//
// Ownership rules:
// - Call tpValueInit() before first use and tpValueFree() when finished.
// - string/exception payloads are owned by the TPValue and are released by
//   tpValueFree() or by another tpValueSet*() call on the same value.
// - setter helpers deep-copy caller-provided strings.
// - tpContextGetGlobal() and tpContextGetLastError() deep-copy runtime data into
//   the caller-provided TPValue.
// - tpContextSetGlobal() and tpModuleAddValue() copy the supplied scalar value
//   into the runtime; the caller retains ownership of the input TPValue.
typedef struct {
    TPValueType type;
    TPResult error;
    char* exceptionType;
    bool boolean;
    double number;
    char* string;
} TPValue;

// Host-native functions currently operate on scalar-only values.
//
// Callback rules:
// - args points to borrowed, read-only TPValue entries that are valid only for
//   the duration of the callback.
// - result is initialized by TensorPy before the callback and is cleaned up by
//   TensorPy after the callback returns.
// - return TP_OK with a nil/bool/number/string result for success.
// - return a non-TP_OK status or set result to TP_VALUE_ERROR to raise a
//   TensorPy exception back into the interpreter.
typedef TPResult (*TPNativeFn)(TPContext* context, int argCount, const TPValue* args, TPValue* result, void* userData);

void tpValueInit(TPValue* value);
void tpValueFree(TPValue* value);
void tpValueSetNil(TPValue* value);
void tpValueSetBool(TPValue* value, bool boolean);
void tpValueSetNumber(TPValue* value, double number);
bool tpValueSetString(TPValue* value, const char* string);
bool tpValueSetError(TPValue* value, TPResult error, const char* message);
bool tpValueSetException(TPValue* value, const char* exceptionType, const char* message);

TPContext* tpContextCreate(void);
void tpContextDestroy(TPContext* context);
TPResult tpContextInterpret(TPContext* context, const char* source, const char* filename);
bool tpContextGetGlobal(TPContext* context, const char* name, TPValue* out);
bool tpContextSetGlobal(TPContext* context, const char* name, const TPValue* value);
bool tpContextGetLastError(TPContext* context, TPValue* out);
TPModule* tpContextCreateModule(TPContext* context, const char* name);
void tpModuleDestroy(TPModule* module);
bool tpModuleAddValue(TPModule* module, const char* name, const TPValue* value);
bool tpModuleAddFunction(TPModule* module, const char* name, TPNativeFn function, void* userData);
int tpApiVersion(void);
int tpExtensionAbiVersion(void);

void tpInit(void);
void tpFree(void);
TPResult tpInterpret(const char* source, const char* filename);

#endif // TENSORPY_API_H
