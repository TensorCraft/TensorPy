#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "tensorpy/platform.h"

char* platformReadTextFile(const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    fseek(file, 0L, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(fileSize + 1);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }

    size_t bytesRead = fread(buffer, sizeof(char), fileSize, file);
    fclose(file);
    if (bytesRead < fileSize) {
        free(buffer);
        return NULL;
    }

    buffer[bytesRead] = '\0';
    return buffer;
}

double platformClockSeconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

double platformRandomDouble(void) {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }
    return (double)rand() / (double)RAND_MAX;
}

char* platformGetCurrentDirectory(void) {
    size_t size = 256;
    for (;;) {
        char* buffer = (char*)malloc(size);
        if (buffer == NULL) {
            return NULL;
        }

        if (getcwd(buffer, size) != NULL) {
            return buffer;
        }

        free(buffer);
        size *= 2;
        if (size > 16384) {
            return NULL;
        }
    }
}

const char* platformName(void) {
    return "macos-darwin";
}
