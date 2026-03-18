#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

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

char** platformListDirectory(const char* path, int* count) {
    DIR* dir;
    struct dirent* entry;
    int capacity = 8;
    int entryCount = 0;
    char** entries;

    if (count == NULL) {
        return NULL;
    }

    dir = opendir(path);
    if (dir == NULL) {
        return NULL;
    }

    entries = (char**)malloc(sizeof(char*) * capacity);
    if (entries == NULL) {
        closedir(dir);
        return NULL;
    }

    while ((entry = readdir(dir)) != NULL) {
        char* copy;
        size_t length;

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        if (entryCount == capacity) {
            char** grown;
            capacity *= 2;
            grown = (char**)realloc(entries, sizeof(char*) * capacity);
            if (grown == NULL) {
                platformFreeDirectoryList(entries, entryCount);
                closedir(dir);
                return NULL;
            }
            entries = grown;
        }

        length = strlen(entry->d_name);
        copy = (char*)malloc(length + 1);
        if (copy == NULL) {
            platformFreeDirectoryList(entries, entryCount);
            closedir(dir);
            return NULL;
        }
        memcpy(copy, entry->d_name, length + 1);
        entries[entryCount++] = copy;
    }

    closedir(dir);
    *count = entryCount;
    return entries;
}

void platformFreeDirectoryList(char** entries, int count) {
    int i;

    if (entries == NULL) {
        return;
    }

    for (i = 0; i < count; i++) {
        free(entries[i]);
    }
    free(entries);
}

static bool platformStatPath(const char* path, struct stat* out) {
    return stat(path, out) == 0;
}

bool platformPathExists(const char* path) {
    struct stat info;
    return platformStatPath(path, &info);
}

bool platformPathIsDirectory(const char* path) {
    struct stat info;
    return platformStatPath(path, &info) && S_ISDIR(info.st_mode);
}

bool platformPathIsFile(const char* path) {
    struct stat info;
    return platformStatPath(path, &info) && S_ISREG(info.st_mode);
}

const char* platformName(void) {
    return "macos-darwin";
}
