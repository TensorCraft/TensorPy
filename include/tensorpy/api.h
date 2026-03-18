#ifndef TENSORPY_API_H
#define TENSORPY_API_H

#include "common.h"

typedef enum {
    TP_OK = 0,
    TP_COMPILE_ERROR = 65,
    TP_RUNTIME_ERROR = 70,
} TPResult;

void tpInit(void);
void tpFree(void);
TPResult tpInterpret(const char* source, const char* filename);

#endif // TENSORPY_API_H
