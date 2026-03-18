#include <stdlib.h>

#include "tensorpy/api.h"
#include "tensorpy/vm.h"

struct TPContext {
    bool active;
};

static TPContext* activeContext = NULL;
static TPContext* defaultContext = NULL;

static TPResult mapInterpretResult(InterpretResult result) {
    if (result == INTERPRET_COMPILE_ERROR) {
        return TP_COMPILE_ERROR;
    }
    if (result == INTERPRET_RUNTIME_ERROR) {
        return TP_RUNTIME_ERROR;
    }
    return TP_OK;
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
    if (context == NULL || context != activeContext || !context->active) {
        return TP_RUNTIME_ERROR;
    }

    return mapInterpretResult(interpret(source, filename));
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
