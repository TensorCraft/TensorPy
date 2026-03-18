#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "tensorpy/memory.h"
#include "tensorpy/platform.h"

typedef pthread_t PlatformThreadHandle;
typedef pthread_mutex_t PlatformMutexHandle;
typedef pthread_cond_t PlatformCondVarHandle;
typedef atomic_int_fast64_t PlatformAtomicHandle;

typedef struct PlatformTaskState {
    PlatformAtomicInt refCount;
    PlatformMutex mutex;
    PlatformCondVar cond;
    bool completed;
} PlatformTaskState;

typedef struct {
    PlatformTaskFunction function;
    void* context;
    PlatformTaskState* state;
} PlatformQueuedTask;

struct PlatformThreadPool {
    PlatformMutex mutex;
    PlatformCondVar cond;
    PlatformCondVar idleCond;
    PlatformThread* threads;
    int threadCount;
    PlatformQueuedTask* queue;
    int queueCapacity;
    int queueHead;
    int queueTail;
    int queueCount;
    int activeCount;
    bool shuttingDown;
    char lastError[128];
};

typedef struct {
    PlatformParallelForFunction function;
    void* context;
    PlatformEvent done;
    PlatformAtomicInt remaining;
} PlatformParallelForState;

typedef struct {
    PlatformParallelForState* state;
    int start;
    int end;
} PlatformParallelForTask;

static PlatformThreadHandle* platformThreadHandle(PlatformThread* thread) {
    return (PlatformThreadHandle*)thread->impl;
}

static PlatformMutexHandle* platformMutexHandle(PlatformMutex* mutex) {
    return (PlatformMutexHandle*)mutex->impl;
}

static PlatformCondVarHandle* platformCondHandle(PlatformCondVar* cond) {
    return (PlatformCondVarHandle*)cond->impl;
}

static PlatformAtomicHandle* platformAtomicHandle(PlatformAtomicInt* value) {
    return (PlatformAtomicHandle*)value->impl;
}

static const PlatformAtomicHandle* platformAtomicHandleConst(const PlatformAtomicInt* value) {
    return (const PlatformAtomicHandle*)value->impl;
}

static void platformWriteLastError(PlatformThreadPool* pool, const char* message) {
    size_t length;

    if (pool == NULL || message == NULL) {
        return;
    }

    length = strlen(message);
    if (length >= sizeof(pool->lastError)) {
        length = sizeof(pool->lastError) - 1;
    }
    memcpy(pool->lastError, message, length);
    pool->lastError[length] = '\0';
}

static bool platformTaskStateCreate(PlatformTaskState** outState) {
    PlatformTaskState* state;

    if (outState == NULL) {
        return false;
    }

    state = (PlatformTaskState*)tpMemAlloc(sizeof(PlatformTaskState));
    if (state == NULL) {
        return false;
    }

    platformAtomicInit(&state->refCount, 1);
    state->completed = false;
    if (!platformMutexInit(&state->mutex) || !platformCondVarInit(&state->cond)) {
        platformCondVarDestroy(&state->cond);
        platformMutexDestroy(&state->mutex);
        tpMemFree(state);
        return false;
    }

    *outState = state;
    return true;
}

static void platformTaskStateRetain(PlatformTaskState* state) {
    if (state != NULL) {
        platformAtomicFetchAdd(&state->refCount, 1);
    }
}

static void platformTaskStateRelease(PlatformTaskState* state) {
    if (state == NULL) {
        return;
    }

    if (platformAtomicFetchSub(&state->refCount, 1) == 1) {
        platformCondVarDestroy(&state->cond);
        platformMutexDestroy(&state->mutex);
        tpMemFree(state);
    }
}

static void platformTaskStateComplete(PlatformTaskState* state) {
    if (state == NULL) {
        return;
    }

    platformMutexLock(&state->mutex);
    state->completed = true;
    platformCondVarBroadcast(&state->cond);
    platformMutexUnlock(&state->mutex);
}

static void platformTaskStateWait(PlatformTaskState* state) {
    if (state == NULL) {
        return;
    }

    platformMutexLock(&state->mutex);
    while (!state->completed) {
        platformCondVarWait(&state->cond, &state->mutex);
    }
    platformMutexUnlock(&state->mutex);
}

static bool platformThreadPoolQueueGrow(PlatformThreadPool* pool) {
    PlatformQueuedTask* grown;
    int i;
    int newCapacity;

    if (pool == NULL) {
        return false;
    }

    newCapacity = pool->queueCapacity < 16 ? 16 : pool->queueCapacity * 2;
    grown = (PlatformQueuedTask*)tpMemAlloc(sizeof(PlatformQueuedTask) * (size_t)newCapacity);
    if (grown == NULL) {
        platformWriteLastError(pool, "failed to grow task queue");
        return false;
    }

    for (i = 0; i < pool->queueCount; i++) {
        grown[i] = pool->queue[(pool->queueHead + i) % pool->queueCapacity];
    }

    tpMemFree(pool->queue);
    pool->queue = grown;
    pool->queueCapacity = newCapacity;
    pool->queueHead = 0;
    pool->queueTail = pool->queueCount;
    return true;
}

static bool platformThreadPoolQueuePush(PlatformThreadPool* pool,
                                        PlatformTaskFunction function,
                                        void* context,
                                        PlatformTaskState* state) {
    if (pool->queueCount == pool->queueCapacity && !platformThreadPoolQueueGrow(pool)) {
        return false;
    }

    pool->queue[pool->queueTail].function = function;
    pool->queue[pool->queueTail].context = context;
    pool->queue[pool->queueTail].state = state;
    pool->queueTail = (pool->queueTail + 1) % pool->queueCapacity;
    pool->queueCount++;
    return true;
}

static bool platformThreadPoolQueuePop(PlatformThreadPool* pool, PlatformQueuedTask* outTask) {
    if (pool == NULL || outTask == NULL || pool->queueCount == 0) {
        return false;
    }

    *outTask = pool->queue[pool->queueHead];
    pool->queueHead = (pool->queueHead + 1) % pool->queueCapacity;
    pool->queueCount--;
    return true;
}

static void* platformThreadPoolWorker(void* context) {
    PlatformThreadPool* pool = (PlatformThreadPool*)context;

    for (;;) {
        PlatformQueuedTask task;

        platformMutexLock(&pool->mutex);
        while (!pool->shuttingDown && pool->queueCount == 0) {
            platformCondVarWait(&pool->cond, &pool->mutex);
        }

        if (pool->shuttingDown && pool->queueCount == 0) {
            platformMutexUnlock(&pool->mutex);
            return NULL;
        }

        if (!platformThreadPoolQueuePop(pool, &task)) {
            platformMutexUnlock(&pool->mutex);
            continue;
        }
        pool->activeCount++;
        platformMutexUnlock(&pool->mutex);

        task.function(task.context);
        platformTaskStateComplete(task.state);
        platformTaskStateRelease(task.state);

        platformMutexLock(&pool->mutex);
        pool->activeCount--;
        if (pool->queueCount == 0 && pool->activeCount == 0) {
            platformCondVarBroadcast(&pool->idleCond);
        }
        platformMutexUnlock(&pool->mutex);
    }
}

static void platformParallelForRunRange(void* context) {
    PlatformParallelForTask* task = (PlatformParallelForTask*)context;
    task->state->function(task->start, task->end, task->state->context);
    if (platformAtomicFetchSub(&task->state->remaining, 1) == 1) {
        platformEventSignal(&task->state->done);
    }
    tpMemFree(task);
}

char* platformReadTextFile(const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    fseek(file, 0L, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* buffer = (char*)tpMemAlloc(fileSize + 1);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }

    size_t bytesRead = fread(buffer, sizeof(char), fileSize, file);
    fclose(file);
    if (bytesRead < fileSize) {
        tpMemFree(buffer);
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

    buffer = (uint8_t*)tpMemAlloc((size_t)fileSize);
    if (buffer == NULL && fileSize > 0) {
        fclose(file);
        return NULL;
    }

    bytesRead = fread(buffer, sizeof(uint8_t), (size_t)fileSize, file);
    fclose(file);
    if (bytesRead < (size_t)fileSize) {
        tpMemFree(buffer);
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

char* platformGetEnvironmentVariable(const char* name) {
    const char* value;

    if (name == NULL) {
        return NULL;
    }

    value = getenv(name);
    if (value == NULL) {
        return NULL;
    }

    return tpMemDup(value);
}

int platformSystemCommand(const char* command) {
    int status;

    if (command == NULL) {
        return -1;
    }

    status = system(command);
    if (status < 0) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    return status;
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
        char* buffer = (char*)tpMemAlloc(size);
        if (buffer == NULL) {
            return NULL;
        }

        if (getcwd(buffer, size) != NULL) {
            return buffer;
        }

        tpMemFree(buffer);
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

    entries = (char**)tpMemAlloc(sizeof(char*) * (size_t)capacity);
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
            grown = (char**)tpMemRealloc(entries, sizeof(char*) * (size_t)capacity);
            if (grown == NULL) {
                platformFreeDirectoryList(entries, entryCount);
                closedir(dir);
                return NULL;
            }
            entries = grown;
        }

        length = strlen(entry->d_name);
        copy = (char*)tpMemAlloc(length + 1);
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
        tpMemFree(entries[i]);
    }
    tpMemFree(entries);
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
    copy = (char*)tpMemAlloc(length + 1);
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
            tpMemFree(copy);
            return false;
        }
        copy[i] = '/';
    }

    if (!platformPathIsDirectory(copy) && mkdir(copy, 0777) != 0) {
        tpMemFree(copy);
        return false;
    }

    tpMemFree(copy);
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

int platformHardwareThreadCount(void) {
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count < 1) {
        return 1;
    }
    if (count > 1024) {
        return 1024;
    }
    return (int)count;
}

bool platformThreadCreate(PlatformThread* thread,
                          PlatformThreadFunction function,
                          void* context) {
    PlatformThreadHandle* handle;

    if (thread == NULL || function == NULL) {
        return false;
    }

    handle = (PlatformThreadHandle*)tpMemAlloc(sizeof(PlatformThreadHandle));
    if (handle == NULL) {
        return false;
    }

    thread->impl = handle;
    thread->started = false;
    if (pthread_create(handle, NULL, function, context) != 0) {
        tpMemFree(handle);
        thread->impl = NULL;
        return false;
    }
    thread->started = true;
    return true;
}

bool platformThreadJoin(PlatformThread* thread, void** result) {
    if (thread == NULL || !thread->started) {
        return false;
    }

    if (pthread_join(*platformThreadHandle(thread), result) != 0) {
        return false;
    }

    tpMemFree(thread->impl);
    thread->impl = NULL;
    thread->started = false;
    return true;
}

bool platformThreadDetach(PlatformThread* thread) {
    if (thread == NULL || !thread->started) {
        return false;
    }

    if (pthread_detach(*platformThreadHandle(thread)) != 0) {
        return false;
    }

    tpMemFree(thread->impl);
    thread->impl = NULL;
    thread->started = false;
    return true;
}

uint64_t platformThreadCurrentId(void) {
    pthread_t self = pthread_self();
    uint64_t out = 0;
    memcpy(&out, &self, sizeof(self) < sizeof(out) ? sizeof(self) : sizeof(out));
    return out;
}

bool platformMutexInit(PlatformMutex* mutex) {
    PlatformMutexHandle* handle;

    if (mutex == NULL) {
        return false;
    }

    handle = (PlatformMutexHandle*)tpMemAlloc(sizeof(PlatformMutexHandle));
    if (handle == NULL) {
        return false;
    }

    mutex->impl = handle;
    mutex->initialized = false;
    if (pthread_mutex_init(handle, NULL) != 0) {
        tpMemFree(handle);
        mutex->impl = NULL;
        return false;
    }
    mutex->initialized = true;
    return true;
}

void platformMutexDestroy(PlatformMutex* mutex) {
    if (mutex == NULL || !mutex->initialized) {
        return;
    }
    pthread_mutex_destroy(platformMutexHandle(mutex));
    tpMemFree(mutex->impl);
    mutex->impl = NULL;
    mutex->initialized = false;
}

void platformMutexLock(PlatformMutex* mutex) {
    if (mutex != NULL && mutex->initialized) {
        pthread_mutex_lock(platformMutexHandle(mutex));
    }
}

void platformMutexUnlock(PlatformMutex* mutex) {
    if (mutex != NULL && mutex->initialized) {
        pthread_mutex_unlock(platformMutexHandle(mutex));
    }
}

bool platformCondVarInit(PlatformCondVar* cond) {
    PlatformCondVarHandle* handle;

    if (cond == NULL) {
        return false;
    }

    handle = (PlatformCondVarHandle*)tpMemAlloc(sizeof(PlatformCondVarHandle));
    if (handle == NULL) {
        return false;
    }

    cond->impl = handle;
    cond->initialized = false;
    if (pthread_cond_init(handle, NULL) != 0) {
        tpMemFree(handle);
        cond->impl = NULL;
        return false;
    }
    cond->initialized = true;
    return true;
}

void platformCondVarDestroy(PlatformCondVar* cond) {
    if (cond == NULL || !cond->initialized) {
        return;
    }
    pthread_cond_destroy(platformCondHandle(cond));
    tpMemFree(cond->impl);
    cond->impl = NULL;
    cond->initialized = false;
}

void platformCondVarWait(PlatformCondVar* cond, PlatformMutex* mutex) {
    if (cond == NULL || mutex == NULL || !cond->initialized || !mutex->initialized) {
        return;
    }
    pthread_cond_wait(platformCondHandle(cond), platformMutexHandle(mutex));
}

void platformCondVarSignal(PlatformCondVar* cond) {
    if (cond != NULL && cond->initialized) {
        pthread_cond_signal(platformCondHandle(cond));
    }
}

void platformCondVarBroadcast(PlatformCondVar* cond) {
    if (cond != NULL && cond->initialized) {
        pthread_cond_broadcast(platformCondHandle(cond));
    }
}

void platformAtomicInit(PlatformAtomicInt* value, int64_t initialValue) {
    PlatformAtomicHandle* handle;

    if (value == NULL) {
        return;
    }

    handle = (PlatformAtomicHandle*)tpMemAlloc(sizeof(PlatformAtomicHandle));
    if (handle == NULL) {
        value->impl = NULL;
        return;
    }
    atomic_init(handle, initialValue);
    value->impl = handle;
}

int64_t platformAtomicLoad(const PlatformAtomicInt* value) {
    if (value == NULL || value->impl == NULL) {
        return 0;
    }
    return atomic_load(platformAtomicHandleConst(value));
}

void platformAtomicStore(PlatformAtomicInt* value, int64_t newValue) {
    if (value != NULL && value->impl != NULL) {
        atomic_store(platformAtomicHandle(value), newValue);
    }
}

int64_t platformAtomicFetchAdd(PlatformAtomicInt* value, int64_t delta) {
    if (value == NULL || value->impl == NULL) {
        return 0;
    }
    return atomic_fetch_add(platformAtomicHandle(value), delta);
}

int64_t platformAtomicFetchSub(PlatformAtomicInt* value, int64_t delta) {
    if (value == NULL || value->impl == NULL) {
        return 0;
    }
    return atomic_fetch_sub(platformAtomicHandle(value), delta);
}

bool platformAtomicCompareExchange(PlatformAtomicInt* value,
                                   int64_t* expected,
                                   int64_t desired) {
    if (value == NULL || expected == NULL) {
        return false;
    }
    if (value->impl == NULL) {
        return false;
    }
    return atomic_compare_exchange_strong(platformAtomicHandle(value), expected, desired);
}

bool platformEventInit(PlatformEvent* event, bool signaled) {
    if (event == NULL) {
        return false;
    }

    if (!platformMutexInit(&event->mutex)) {
        return false;
    }
    if (!platformCondVarInit(&event->cond)) {
        platformMutexDestroy(&event->mutex);
        return false;
    }
    event->signaled = signaled;
    return true;
}

void platformEventDestroy(PlatformEvent* event) {
    if (event == NULL) {
        return;
    }
    platformCondVarDestroy(&event->cond);
    platformMutexDestroy(&event->mutex);
}

void platformEventWait(PlatformEvent* event) {
    if (event == NULL) {
        return;
    }

    platformMutexLock(&event->mutex);
    while (!event->signaled) {
        platformCondVarWait(&event->cond, &event->mutex);
    }
    platformMutexUnlock(&event->mutex);
}

void platformEventSignal(PlatformEvent* event) {
    if (event == NULL) {
        return;
    }

    platformMutexLock(&event->mutex);
    event->signaled = true;
    platformCondVarBroadcast(&event->cond);
    platformMutexUnlock(&event->mutex);
}

void platformEventReset(PlatformEvent* event) {
    if (event == NULL) {
        return;
    }

    platformMutexLock(&event->mutex);
    event->signaled = false;
    platformMutexUnlock(&event->mutex);
}

PlatformThreadPool* platformThreadPoolCreate(int threadCount) {
    PlatformThreadPool* pool;
    int i;

    if (threadCount <= 0) {
        threadCount = platformHardwareThreadCount();
    }

    pool = (PlatformThreadPool*)tpMemCalloc(1, sizeof(PlatformThreadPool));
    if (pool == NULL) {
        return NULL;
    }

    if (!platformMutexInit(&pool->mutex) ||
        !platformCondVarInit(&pool->cond) ||
        !platformCondVarInit(&pool->idleCond)) {
        platformThreadPoolDestroy(pool);
        return NULL;
    }

    pool->queueCapacity = threadCount * 4;
    if (pool->queueCapacity < 16) {
        pool->queueCapacity = 16;
    }

    pool->queue = (PlatformQueuedTask*)tpMemCalloc((size_t)pool->queueCapacity, sizeof(PlatformQueuedTask));
    pool->threads = (PlatformThread*)tpMemCalloc((size_t)threadCount, sizeof(PlatformThread));
    if (pool->queue == NULL || pool->threads == NULL) {
        platformWriteLastError(pool, "failed to allocate thread pool resources");
        platformThreadPoolDestroy(pool);
        return NULL;
    }

    pool->threadCount = threadCount;
    pool->lastError[0] = '\0';
    for (i = 0; i < pool->threadCount; i++) {
        if (!platformThreadCreate(&pool->threads[i], platformThreadPoolWorker, pool)) {
            platformWriteLastError(pool, "failed to create worker thread");
            pool->shuttingDown = true;
            platformCondVarBroadcast(&pool->cond);
            while (i-- > 0) {
                platformThreadJoin(&pool->threads[i], NULL);
            }
            platformThreadPoolDestroy(pool);
            return NULL;
        }
    }

    return pool;
}

void platformThreadPoolDestroy(PlatformThreadPool* pool) {
    int i;

    if (pool == NULL) {
        return;
    }

    if (pool->threads != NULL) {
        platformMutexLock(&pool->mutex);
        pool->shuttingDown = true;
        platformCondVarBroadcast(&pool->cond);
        platformMutexUnlock(&pool->mutex);

        for (i = 0; i < pool->threadCount; i++) {
            if (pool->threads[i].started) {
                platformThreadJoin(&pool->threads[i], NULL);
            }
        }
    }

    if (pool->queue != NULL) {
        for (i = 0; i < pool->queueCount; i++) {
            PlatformQueuedTask* task = &pool->queue[(pool->queueHead + i) % pool->queueCapacity];
            platformTaskStateComplete(task->state);
            platformTaskStateRelease(task->state);
        }
        tpMemFree(pool->queue);
    }

    tpMemFree(pool->threads);
    platformCondVarDestroy(&pool->idleCond);
    platformCondVarDestroy(&pool->cond);
    platformMutexDestroy(&pool->mutex);
    tpMemFree(pool);
}

int platformThreadPoolThreadCount(PlatformThreadPool* pool) {
    if (pool == NULL) {
        return 0;
    }
    return pool->threadCount;
}

const char* platformThreadPoolLastError(PlatformThreadPool* pool) {
    if (pool == NULL || pool->lastError[0] == '\0') {
        return NULL;
    }
    return pool->lastError;
}

bool platformThreadPoolSubmit(PlatformThreadPool* pool,
                              PlatformTaskFunction function,
                              void* context,
                              PlatformTaskHandle* outHandle) {
    PlatformTaskState* state;

    if (pool == NULL || function == NULL || outHandle == NULL) {
        return false;
    }

    outHandle->state = NULL;
    if (!platformTaskStateCreate(&state)) {
        platformWriteLastError(pool, "failed to allocate task state");
        return false;
    }

    platformMutexLock(&pool->mutex);
    if (pool->shuttingDown) {
        platformMutexUnlock(&pool->mutex);
        platformWriteLastError(pool, "thread pool is shutting down");
        platformTaskStateRelease(state);
        return false;
    }

    platformTaskStateRetain(state);
    if (!platformThreadPoolQueuePush(pool, function, context, state)) {
        platformMutexUnlock(&pool->mutex);
        platformTaskStateRelease(state);
        platformTaskStateRelease(state);
        return false;
    }

    outHandle->state = state;
    platformCondVarSignal(&pool->cond);
    platformMutexUnlock(&pool->mutex);
    return true;
}

bool platformTaskHandleIsValid(const PlatformTaskHandle* handle) {
    return handle != NULL && handle->state != NULL;
}

bool platformTaskHandleIsDone(const PlatformTaskHandle* handle) {
    bool completed;

    if (!platformTaskHandleIsValid(handle)) {
        return true;
    }

    platformMutexLock(&handle->state->mutex);
    completed = handle->state->completed;
    platformMutexUnlock(&handle->state->mutex);
    return completed;
}

void platformTaskHandleWait(PlatformTaskHandle* handle) {
    if (!platformTaskHandleIsValid(handle)) {
        return;
    }
    platformTaskStateWait(handle->state);
}

void platformTaskHandleDestroy(PlatformTaskHandle* handle) {
    if (handle == NULL || handle->state == NULL) {
        return;
    }
    platformTaskStateRelease(handle->state);
    handle->state = NULL;
}

void platformThreadPoolWaitAll(PlatformThreadPool* pool) {
    if (pool == NULL) {
        return;
    }

    platformMutexLock(&pool->mutex);
    while (pool->queueCount > 0 || pool->activeCount > 0) {
        platformCondVarWait(&pool->idleCond, &pool->mutex);
    }
    platformMutexUnlock(&pool->mutex);
}

bool platformThreadPoolParallelFor(PlatformThreadPool* pool,
                                   int start,
                                   int end,
                                   int grainSize,
                                   PlatformParallelForFunction function,
                                   void* context) {
    PlatformParallelForState state;
    PlatformTaskHandle* handles;
    int taskCount;
    int index;
    int begin;

    if (function == NULL) {
        return false;
    }

    if (end <= start) {
        return true;
    }

    if (pool == NULL || pool->threadCount <= 0) {
        function(start, end, context);
        return true;
    }

    if (grainSize <= 0) {
        grainSize = 1;
    }

    taskCount = (end - start + grainSize - 1) / grainSize;
    if (taskCount <= 1) {
        function(start, end, context);
        return true;
    }

    handles = (PlatformTaskHandle*)tpMemCalloc((size_t)taskCount, sizeof(PlatformTaskHandle));
    if (handles == NULL) {
        platformWriteLastError(pool, "failed to allocate parallel_for handles");
        return false;
    }

    state.function = function;
    state.context = context;
    if (!platformEventInit(&state.done, false)) {
        platformWriteLastError(pool, "failed to initialize parallel_for event");
        tpMemFree(handles);
        return false;
    }
    platformAtomicInit(&state.remaining, taskCount);

    begin = start;
    for (index = 0; index < taskCount; index++) {
        PlatformParallelForTask* task = (PlatformParallelForTask*)tpMemAlloc(sizeof(PlatformParallelForTask));
        if (task == NULL) {
            platformWriteLastError(pool, "failed to allocate parallel_for task");
            break;
        }

        task->state = &state;
        task->start = begin;
        task->end = begin + grainSize;
        if (task->end > end) {
            task->end = end;
        }
        begin = task->end;

        if (!platformThreadPoolSubmit(pool, platformParallelForRunRange, task, &handles[index])) {
            tpMemFree(task);
            break;
        }
    }

    if (index != taskCount) {
        int j;
        for (j = 0; j < index; j++) {
            platformTaskHandleWait(&handles[j]);
            platformTaskHandleDestroy(&handles[j]);
        }
        platformEventDestroy(&state.done);
        tpMemFree(handles);
        return false;
    }

    platformEventWait(&state.done);

    for (index = 0; index < taskCount; index++) {
        platformTaskHandleWait(&handles[index]);
        platformTaskHandleDestroy(&handles[index]);
    }

    platformEventDestroy(&state.done);
    tpMemFree(handles);
    return true;
}
