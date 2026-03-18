#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorpy/api.h"
#include "tensorpy/object.h"
#include "tensorpy/table.h"
#include "tensorpy/vm.h"

struct TPContext {
    bool active;
};

typedef struct {
    TPContext* context;
    TPNativeFn function;
    void* userData;
} TPHostNative;

struct TPModule {
    TPContext* context;
    ObjInstance* module;
};

static TPContext* activeContext = NULL;
static TPContext* defaultContext = NULL;

static char* duplicateString(const char* source) {
    size_t length;
    char* copy;

    if (source == NULL) {
        return NULL;
    }

    length = strlen(source);
    copy = (char*)malloc(length + 1);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, source, length + 1);
    return copy;
}

static bool isContextUsable(TPContext* context) {
    return context != NULL && context == activeContext && context->active;
}

static bool formatExceptionValue(Value exception, TPValue* out) {
    Value message;
    Value messageKey;

    if (IS_INSTANCE(exception)) {
        ObjInstance* instance = AS_INSTANCE(exception);

        vm.gcPauseDepth++;
        messageKey = OBJ_VAL(copyString("message", 7));
        if (tableGet(&instance->fields, messageKey, &message) && IS_STRING(message) &&
            AS_STRING(message)->length > 0) {
            size_t total = (size_t)instance->klass->name->length + 2 + (size_t)AS_STRING(message)->length;
            char* buffer = (char*)malloc(total + 1);
            vm.gcPauseDepth--;
            if (buffer == NULL) {
                return false;
            }
            snprintf(buffer, total + 1, "%s: %s", instance->klass->name->chars, AS_STRING(message)->chars);
            tpValueSetNil(out);
            out->type = TP_VALUE_ERROR;
            out->error = TP_RUNTIME_ERROR;
            out->exceptionType = duplicateString(instance->klass->name->chars);
            out->string = buffer;
            return out->exceptionType != NULL;
        }
        vm.gcPauseDepth--;
        return tpValueSetException(out, instance->klass->name->chars, instance->klass->name->chars);
    }

    if (IS_CLASS(exception)) {
        return tpValueSetException(out, AS_CLASS(exception)->name->chars, AS_CLASS(exception)->name->chars);
    }

    if (IS_STRING(exception)) {
        return tpValueSetException(out, "RuntimeError", AS_STRING(exception)->chars);
    }

    return tpValueSetException(out, "RuntimeError", "RuntimeError");
}

static bool tpValueFromRuntimeValue(Value value, TPValue* out) {
    tpValueSetNil(out);

    if (IS_NIL(value)) {
        return true;
    }
    if (IS_BOOL(value)) {
        tpValueSetBool(out, AS_BOOL(value));
        return true;
    }
    if (IS_NUMBER(value)) {
        tpValueSetNumber(out, AS_NUMBER(value));
        return true;
    }
    if (IS_STRING(value)) {
        return tpValueSetString(out, AS_STRING(value)->chars);
    }

    return false;
}

static bool runtimeValueFromTPValue(const TPValue* value, Value* out) {
    vm.gcPauseDepth++;

    switch (value->type) {
        case TP_VALUE_NIL:
            *out = NIL_VAL;
            vm.gcPauseDepth--;
            return true;
        case TP_VALUE_BOOL:
            *out = BOOL_VAL(value->boolean);
            vm.gcPauseDepth--;
            return true;
        case TP_VALUE_NUMBER:
            *out = NUMBER_VAL(value->number);
            vm.gcPauseDepth--;
            return true;
        case TP_VALUE_STRING:
            if (value->string == NULL) {
                vm.gcPauseDepth--;
                return false;
            }
            *out = OBJ_VAL(copyString(value->string, (int)strlen(value->string)));
            vm.gcPauseDepth--;
            return true;
        case TP_VALUE_ERROR:
            vm.gcPauseDepth--;
            return false;
    }

    vm.gcPauseDepth--;
    return false;
}

static void freeHostNative(void* userData) {
    free(userData);
}

static void raiseHostNativeFailure(const TPValue* result, const char* fallbackMessage) {
    const char* exceptionType = "RuntimeError";
    const char* message = fallbackMessage;

    if (result != NULL) {
        if (result->exceptionType != NULL) {
            exceptionType = result->exceptionType;
        }
        if (result->string != NULL) {
            message = result->string;
        }
    }

    vmRaiseExceptionMessage(exceptionType, message);
}

static Value hostNativeAdapter(int argCount, Value* args) {
    TPHostNative* hostNative;
    TPValue* nativeArgs;
    TPValue result;
    TPResult status;
    Value runtimeResult;
    int i;

    if (vm.currentNative == NULL || vm.currentNative->userData == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Invalid host native call state.");
        return NIL_VAL;
    }

    hostNative = (TPHostNative*)vm.currentNative->userData;
    nativeArgs = (TPValue*)calloc((size_t)argCount, sizeof(TPValue));
    if (nativeArgs == NULL && argCount > 0) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while preparing host native arguments.");
        return NIL_VAL;
    }

    for (i = 0; i < argCount; i++) {
        tpValueInit(&nativeArgs[i]);
        if (!tpValueFromRuntimeValue(args[i], &nativeArgs[i])) {
            vmRaiseExceptionMessage("TypeError", "Host native functions currently accept only nil, bool, number, and string arguments.");
            while (i >= 0) {
                tpValueFree(&nativeArgs[i]);
                i--;
            }
            free(nativeArgs);
            return NIL_VAL;
        }
    }

    tpValueInit(&result);
    status = hostNative->function(hostNative->context, argCount, nativeArgs, &result, hostNative->userData);

    for (i = 0; i < argCount; i++) {
        tpValueFree(&nativeArgs[i]);
    }
    free(nativeArgs);

    if (status != TP_OK) {
        raiseHostNativeFailure(&result, "Host native function failed.");
        tpValueFree(&result);
        return NIL_VAL;
    }

    if (result.type == TP_VALUE_ERROR) {
        raiseHostNativeFailure(&result, "Host native function returned an error value.");
        tpValueFree(&result);
        return NIL_VAL;
    }

    if (!runtimeValueFromTPValue(&result, &runtimeResult)) {
        vmRaiseExceptionMessage("TypeError", "Host native functions currently return only nil, bool, number, and string values.");
        tpValueFree(&result);
        return NIL_VAL;
    }

    tpValueFree(&result);
    return runtimeResult;
}

static TPResult mapInterpretResult(InterpretResult result) {
    if (result == INTERPRET_COMPILE_ERROR) {
        return TP_COMPILE_ERROR;
    }
    if (result == INTERPRET_RUNTIME_ERROR) {
        return TP_RUNTIME_ERROR;
    }
    return TP_OK;
}

void tpValueInit(TPValue* value) {
    if (value == NULL) {
        return;
    }

    value->type = TP_VALUE_NIL;
    value->error = TP_OK;
    value->exceptionType = NULL;
    value->boolean = false;
    value->number = 0;
    value->string = NULL;
}

void tpValueFree(TPValue* value) {
    if (value == NULL) {
        return;
    }

    if (value->exceptionType != NULL) {
        free(value->exceptionType);
        value->exceptionType = NULL;
    }
    if (value->string != NULL) {
        free(value->string);
        value->string = NULL;
    }
    value->type = TP_VALUE_NIL;
    value->error = TP_OK;
    value->boolean = false;
    value->number = 0;
}

void tpValueSetNil(TPValue* value) {
    if (value == NULL) {
        return;
    }

    tpValueFree(value);
}

void tpValueSetBool(TPValue* value, bool boolean) {
    if (value == NULL) {
        return;
    }

    tpValueFree(value);
    value->type = TP_VALUE_BOOL;
    value->boolean = boolean;
}

void tpValueSetNumber(TPValue* value, double number) {
    if (value == NULL) {
        return;
    }

    tpValueFree(value);
    value->type = TP_VALUE_NUMBER;
    value->number = number;
}

bool tpValueSetString(TPValue* value, const char* string) {
    char* copy;

    if (value == NULL || string == NULL) {
        return false;
    }

    copy = duplicateString(string);
    if (copy == NULL) {
        return false;
    }

    tpValueFree(value);
    value->type = TP_VALUE_STRING;
    value->string = copy;
    return true;
}

bool tpValueSetError(TPValue* value, TPResult error, const char* message) {
    char* copy;

    if (value == NULL || message == NULL) {
        return false;
    }

    copy = duplicateString(message);
    if (copy == NULL) {
        return false;
    }

    tpValueFree(value);
    value->type = TP_VALUE_ERROR;
    value->error = error;
    value->string = copy;
    return true;
}

bool tpValueSetException(TPValue* value, const char* exceptionType, const char* message) {
    char* typeCopy;
    char* messageCopy;

    if (value == NULL || exceptionType == NULL || message == NULL) {
        return false;
    }

    typeCopy = duplicateString(exceptionType);
    if (typeCopy == NULL) {
        return false;
    }

    messageCopy = duplicateString(message);
    if (messageCopy == NULL) {
        free(typeCopy);
        return false;
    }

    tpValueFree(value);
    value->type = TP_VALUE_ERROR;
    value->error = TP_RUNTIME_ERROR;
    value->exceptionType = typeCopy;
    value->string = messageCopy;
    return true;
}

TPContext* tpContextCreate(void) {
    TPContext* context;

    if (activeContext != NULL) {
        return NULL;
    }

    context = (TPContext*)malloc(sizeof(TPContext));
    if (context == NULL) {
        return NULL;
    }

    initVM();
    context->active = true;
    activeContext = context;
    return context;
}

void tpContextDestroy(TPContext* context) {
    if (context == NULL) {
        return;
    }

    if (context == defaultContext) {
        defaultContext = NULL;
    }

    if (context == activeContext && context->active) {
        freeVM();
        activeContext = NULL;
    }

    free(context);
}

TPResult tpContextInterpret(TPContext* context, const char* source, const char* filename) {
    if (!isContextUsable(context)) {
        return TP_RUNTIME_ERROR;
    }

    return mapInterpretResult(interpret(source, filename));
}

bool tpContextGetGlobal(TPContext* context, const char* name, TPValue* out) {
    Value value;

    if (!isContextUsable(context) || name == NULL || out == NULL) {
        return false;
    }

    tpValueSetNil(out);
    vm.gcPauseDepth++;
    if (!vmGetGlobalValue(name, &value)) {
        vm.gcPauseDepth--;
        return false;
    }
    vm.gcPauseDepth--;

    return tpValueFromRuntimeValue(value, out);
}

bool tpContextSetGlobal(TPContext* context, const char* name, const TPValue* value) {
    Value runtimeValue;

    if (!isContextUsable(context) || name == NULL || value == NULL) {
        return false;
    }

    if (!runtimeValueFromTPValue(value, &runtimeValue)) {
        return false;
    }

    vm.gcPauseDepth++;
    vmSetGlobalValue(name, runtimeValue);
    vm.gcPauseDepth--;
    return true;
}

bool tpContextGetLastError(TPContext* context, TPValue* out) {
    Value exception;

    if (!isContextUsable(context) || out == NULL) {
        return false;
    }

    tpValueSetNil(out);
    if (!vmGetLastException(&exception)) {
        return false;
    }

    return formatExceptionValue(exception, out);
}

TPModule* tpContextCreateModule(TPContext* context, const char* name) {
    TPModule* module;
    ObjString* moduleName;

    if (!isContextUsable(context) || name == NULL) {
        return NULL;
    }

    module = (TPModule*)malloc(sizeof(TPModule));
    if (module == NULL) {
        return NULL;
    }

    vm.gcPauseDepth++;
    moduleName = copyString(name, (int)strlen(name));
    module->module = newInstance(vm.moduleClass);
    tableSet(&module->module->fields, OBJ_VAL(copyString("__name__", 8)), OBJ_VAL(moduleName));
    tableSet(&vm.modules, OBJ_VAL(moduleName), OBJ_VAL(module->module));
    vm.gcPauseDepth--;

    module->context = context;
    return module;
}

void tpModuleDestroy(TPModule* module) {
    if (module == NULL) {
        return;
    }

    free(module);
}

bool tpModuleAddValue(TPModule* module, const char* name, const TPValue* value) {
    Value runtimeValue;

    if (module == NULL || module->context == NULL || !isContextUsable(module->context) ||
        name == NULL || value == NULL) {
        return false;
    }

    if (!runtimeValueFromTPValue(value, &runtimeValue)) {
        return false;
    }

    vm.gcPauseDepth++;
    tableSet(&module->module->fields,
             OBJ_VAL(copyString(name, (int)strlen(name))),
             runtimeValue);
    vm.gcPauseDepth--;
    return true;
}

bool tpModuleAddFunction(TPModule* module, const char* name, TPNativeFn function, void* userData) {
    TPHostNative* hostNative;
    ObjNative* native;

    if (module == NULL || module->context == NULL || !isContextUsable(module->context) ||
        name == NULL || function == NULL) {
        return false;
    }

    hostNative = (TPHostNative*)malloc(sizeof(TPHostNative));
    if (hostNative == NULL) {
        return false;
    }

    hostNative->context = module->context;
    hostNative->function = function;
    hostNative->userData = userData;

    vm.gcPauseDepth++;
    native = newNativeWithFinalizer(hostNativeAdapter, hostNative, freeHostNative);
    tableSet(&module->module->fields,
             OBJ_VAL(copyString(name, (int)strlen(name))),
             OBJ_VAL(native));
    vm.gcPauseDepth--;
    return true;
}

int tpApiVersion(void) {
    return TENSORPY_API_VERSION;
}

int tpExtensionAbiVersion(void) {
    return TENSORPY_EXTENSION_ABI_VERSION;
}

void tpInit(void) {
    if (defaultContext != NULL) {
        return;
    }

    defaultContext = tpContextCreate();
}

void tpFree(void) {
    if (defaultContext == NULL) {
        return;
    }

    tpContextDestroy(defaultContext);
}

TPResult tpInterpret(const char* source, const char* filename) {
    if (defaultContext == NULL) {
        tpInit();
    }
    if (defaultContext == NULL) {
        return TP_RUNTIME_ERROR;
    }

    return tpContextInterpret(defaultContext, source, filename);
}
