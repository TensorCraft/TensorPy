#include <stdlib.h>
#include <string.h>

#include "tensorpy/memory.h"

void* tpMemAlloc(size_t size) {
    if (size == 0) {
        size = 1;
    }
    return malloc(size);
}

void* tpMemCalloc(size_t count, size_t size) {
    if (count == 0) {
        count = 1;
    }
    if (size == 0) {
        size = 1;
    }
    return calloc(count, size);
}

void* tpMemRealloc(void* ptr, size_t size) {
    if (size == 0) {
        size = 1;
    }
    return realloc(ptr, size);
}

void tpMemFree(void* ptr) {
    free(ptr);
}

char* tpMemDup(const char* source) {
    size_t length;
    char* copy;

    if (source == NULL) {
        return NULL;
    }

    length = strlen(source);
    copy = (char*)tpMemAlloc(length + 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, source, length + 1);
    return copy;
}
