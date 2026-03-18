#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <errno.h>
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

uint8_t* platformReadBinaryFile(const char* path, int* count) {
    FILE* file;
    long fileSize;
    uint8_t* buffer;
    size_t bytesRead;

    if (count == NULL) {
        return NULL;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    fileSize = ftell(file);
    if (fileSize < 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);

    buffer = (uint8_t*)malloc((size_t)fileSize);
    if (buffer == NULL && fileSize > 0) {
        fclose(file);
        return NULL;
    }

    bytesRead = fread(buffer, sizeof(uint8_t), (size_t)fileSize, file);
    fclose(file);
    if (bytesRead < (size_t)fileSize) {
        free(buffer);
        return NULL;
    }

    *count = (int)fileSize;
    return buffer;
}

bool platformWriteTextFile(const char* path, const char* text) {
    FILE* file;
    size_t length;
    size_t bytesWritten;

    if (path == NULL || text == NULL) {
        return false;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        return false;
    }

    length = strlen(text);
    bytesWritten = fwrite(text, sizeof(char), length, file);
    fclose(file);
    return bytesWritten == length;
}

bool platformWriteBinaryFile(const char* path, const uint8_t* bytes, int count) {
    FILE* file;
    size_t bytesWritten;

    if (path == NULL || count < 0 || (count > 0 && bytes == NULL)) {
        return false;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        return false;
    }

    bytesWritten = fwrite(bytes, sizeof(uint8_t), (size_t)count, file);
    fclose(file);
    return bytesWritten == (size_t)count;
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

bool platformCreateDirectory(const char* path) {
    if (path == NULL || path[0] == '\0') {
        return false;
    }

    return mkdir(path, 0777) == 0;
}

bool platformCreateDirectories(const char* path) {
    char* copy;
    size_t length;
    size_t i;

    if (path == NULL || path[0] == '\0') {
        return false;
    }

    if (platformPathIsDirectory(path)) {
        return true;
    }

    length = strlen(path);
    copy = (char*)malloc(length + 1);
    if (copy == NULL) {
        return false;
    }
    memcpy(copy, path, length + 1);

    for (i = 1; i < length; i++) {
        if (copy[i] != '/') {
            continue;
        }

        copy[i] = '\0';
        if (copy[0] != '\0' && !platformPathIsDirectory(copy) && mkdir(copy, 0777) != 0) {
            free(copy);
            return false;
        }
        copy[i] = '/';
    }

    if (!platformPathIsDirectory(copy) && mkdir(copy, 0777) != 0) {
        free(copy);
        return false;
    }

    free(copy);
    return true;
}

bool platformRemoveFile(const char* path) {
    if (path == NULL) {
        return false;
    }

    return unlink(path) == 0;
}

bool platformRemoveDirectory(const char* path) {
    if (path == NULL) {
        return false;
    }

    return rmdir(path) == 0;
}

bool platformRenamePath(const char* from, const char* to) {
    if (from == NULL || to == NULL) {
        return false;
    }

    return rename(from, to) == 0;
}

const char* platformName(void) {
    return "macos-darwin";
}
