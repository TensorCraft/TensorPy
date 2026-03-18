#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorpy/object.h"
#include "tensorpy/table.h"
#include "tensorpy/value.h"
#include "tensorpy/vm.h"

// In a real implementation, we'd have a memory manager
#define ALLOCATE_OBJ(type, objectType) \
    (type*)allocateObject(sizeof(type), objectType)

static Obj* allocateObject(size_t size, ObjType type) {
    if (vm.gcEnabled && vm.gcPauseDepth == 0 && vm.objectCount >= vm.nextGC) {
        gcCollect();
    }

    Obj* object = (Obj*)malloc(size);
    object->type = type;
    object->isMarked = false;
    object->next = vm.objects;
    vm.objects = object;
    vm.objectCount++;
    return object;
}

static uint32_t hashString(const char* key, int length) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < length; i++) {
        hash ^= (uint8_t)key[i];
        hash *= 16777619;
    }
    return hash;
}

static ObjString* allocateString(char* chars, int length, uint32_t hash) {
    ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
    string->length = length;
    string->chars = chars;
    string->hash = hash;
    return string;
}

int tensorElementCount(int rank, const int* shape) {
    int size = 1;
    int i;

    if (rank < 0) {
        return -1;
    }
    if (rank == 0) {
        return 1;
    }

    for (i = 0; i < rank; i++) {
        if (shape == NULL || shape[i] < 0) {
            return -1;
        }
        size *= shape[i];
    }
    return size;
}

static ObjList* makeIntList(const int* values, int count) {
    ObjList* list = newList();
    int i;

    for (i = 0; i < count; i++) {
        writeValueArray(&list->items, NUMBER_VAL((double)values[i]));
    }
    return list;
}

static void fillTensorStrides(const ObjTensor* tensor, int* outStrides) {
    int stride = 1;
    int i;

    for (i = tensor->rank - 1; i >= 0; i--) {
        outStrides[i] = stride;
        stride *= tensor->shape[i];
    }
}

static void freeObject(Obj* object) {
    switch (object->type) {
        case OBJ_STRING:
            free(((ObjString*)object)->chars);
            break;
        case OBJ_FUNCTION: {
            ObjFunction* function = (ObjFunction*)object;
            freeValueArray(&function->defaults);
            freeValueArray(&function->paramNames);
            freeValueArray(&function->localNames);
            freeChunk(&function->chunk);
            break;
        }
        case OBJ_CLOSURE:
        case OBJ_SLICE:
        case OBJ_ITERATOR:
        case OBJ_BOUND_METHOD:
            break;
        case OBJ_NATIVE: {
            ObjNative* native = (ObjNative*)object;
            if (native->freeUserData != NULL && native->userData != NULL) {
                native->freeUserData(native->userData);
            }
            break;
        }
        case OBJ_ENVIRONMENT: {
            ObjEnvironment* env = (ObjEnvironment*)object;
            if (env->ownsTable) {
                freeTable(env->table);
            }
            break;
        }
        case OBJ_SET:
            freeTable(&((ObjSet*)object)->table);
            break;
        case OBJ_LIST:
            freeValueArray(&((ObjList*)object)->items);
            break;
        case OBJ_DICT:
            freeTable(&((ObjDict*)object)->table);
            break;
        case OBJ_TUPLE:
            freeValueArray(&((ObjTuple*)object)->items);
            break;
        case OBJ_BYTES:
            free(((ObjBytes*)object)->bytes);
            break;
        case OBJ_DEVICE:
        case OBJ_DTYPE:
            break;
        case OBJ_TENSOR: {
            ObjTensor* tensor = (ObjTensor*)object;
            tpMetalBufferDestroy(tensor->metalBuffer);
            free(tensor->shape);
            free(tensor->data);
            break;
        }
        case OBJ_CLASS:
            freeTable(&((ObjClass*)object)->methods);
            break;
        case OBJ_INSTANCE:
            freeTable(&((ObjInstance*)object)->fields);
            break;
    }

    vm.objectCount--;
    free(object);
}

ObjString* copyString(const char* chars, int length) {
    uint32_t hash = hashString(chars, length);
    ObjString* interned = tableFindString(&vm.strings, chars, length, hash);
    if (interned != NULL) return interned;

    char* heapChars = malloc(length + 1);
    memcpy(heapChars, chars, length);
    heapChars[length] = '\0';
    ObjString* string = allocateString(heapChars, length, hash);
    tableSet(&vm.strings, OBJ_VAL(string), NIL_VAL);
    return string;
}

ObjString* takeString(char* chars, int length) {
    uint32_t hash = hashString(chars, length);
    ObjString* interned = tableFindString(&vm.strings, chars, length, hash);
    if (interned != NULL) {
        free(chars);
        return interned;
    }

    ObjString* string = allocateString(chars, length, hash);
    tableSet(&vm.strings, OBJ_VAL(string), NIL_VAL);
    return string;
}

ObjBytes* newBytes(int length, const uint8_t* source) {
    ObjBytes* bytes = ALLOCATE_OBJ(ObjBytes, OBJ_BYTES);
    bytes->length = length;
    bytes->bytes = (uint8_t*)malloc(length);
    if (source != NULL) {
        memcpy(bytes->bytes, source, length);
    }
    return bytes;
}

ObjSlice* newSlice(Value start, Value stop, Value step) {
    ObjSlice* slice = ALLOCATE_OBJ(ObjSlice, OBJ_SLICE);
    slice->start = start;
    slice->stop = stop;
    slice->step = step;
    return slice;
}

ObjIterator* newIterator(Value iterable) {
    ObjIterator* iterator = ALLOCATE_OBJ(ObjIterator, OBJ_ITERATOR);
    iterator->iterable = iterable;
    iterator->index = 0;
    return iterator;
}

ObjFunction* newFunction() {
    ObjFunction* function = ALLOCATE_OBJ(ObjFunction, OBJ_FUNCTION);
    function->arity = 0;
    function->defaultsCount = 0;
    function->maxSlots = 0;
    function->hasVarargs = false;
    initValueArray(&function->defaults);
    initValueArray(&function->paramNames);
    initValueArray(&function->localNames);
    function->name = NULL;
    initChunk(&function->chunk);
    return function;
}

ObjClosure* newClosure(ObjFunction* function, ObjEnvironment* env) {
    ObjClosure* closure = ALLOCATE_OBJ(ObjClosure, OBJ_CLOSURE);
    closure->function = function;
    closure->env = env;
    return closure;
}

ObjEnvironment* newEnvironment(ObjEnvironment* parent) {
    ObjEnvironment* env = ALLOCATE_OBJ(ObjEnvironment, OBJ_ENVIRONMENT);
    initTable(&env->values);
    env->table = &env->values;
    env->ownsTable = true;
    env->parent = parent;
    return env;
}

ObjEnvironment* newEnvironmentWithTable(ObjEnvironment* parent, Table* table) {
    ObjEnvironment* env = ALLOCATE_OBJ(ObjEnvironment, OBJ_ENVIRONMENT);
    initTable(&env->values);
    env->table = table;
    env->ownsTable = false;
    env->parent = parent;
    return env;
}

ObjNative* newNative(NativeFn function) {
    return newNativeWithFinalizer(function, NULL, NULL);
}

ObjNative* newNativeWithUserData(NativeFn function, void* userData) {
    return newNativeWithFinalizer(function, userData, NULL);
}

ObjNative* newNativeWithFinalizer(NativeFn function, void* userData, NativeFreeFn freeUserData) {
    ObjNative* native = ALLOCATE_OBJ(ObjNative, OBJ_NATIVE);
    native->function = function;
    native->userData = userData;
    native->freeUserData = freeUserData;
    return native;
}

ObjSet* newSet() {
    ObjSet* set = ALLOCATE_OBJ(ObjSet, OBJ_SET);
    initTable(&set->table);
    return set;
}

ObjList* newList() {
    ObjList* list = ALLOCATE_OBJ(ObjList, OBJ_LIST);
    initValueArray(&list->items);
    return list;
}

ObjDict* newDict() {
    ObjDict* dict = ALLOCATE_OBJ(ObjDict, OBJ_DICT);
    initTable(&dict->table);
    return dict;
}

ObjTuple* newTuple() {
    ObjTuple* tuple = ALLOCATE_OBJ(ObjTuple, OBJ_TUPLE);
    initValueArray(&tuple->items);
    return tuple;
}

ObjDevice* newDevice(ObjString* name, TPDeviceKind kind) {
    ObjDevice* device = ALLOCATE_OBJ(ObjDevice, OBJ_DEVICE);
    device->kind = kind;
    device->name = name;
    return device;
}

ObjDType* newDType(ObjString* name, TPDTypeKind kind) {
    ObjDType* dtype = ALLOCATE_OBJ(ObjDType, OBJ_DTYPE);
    dtype->kind = kind;
    dtype->name = name;
    return dtype;
}

ObjTensor* newTensor(int rank,
                     const int* shape,
                     ObjDType* dtype,
                     ObjDevice* device,
                     const float* source) {
    ObjTensor* tensor;
    int size;
    int* shapeCopy = NULL;
    float* dataCopy = NULL;
    TPMetalBuffer* metalBuffer = NULL;

    if (dtype == NULL || device == NULL) {
        return NULL;
    }

    size = tensorElementCount(rank, shape);
    if (size < 0) {
        return NULL;
    }

    if (rank > 0) {
        shapeCopy = (int*)malloc(sizeof(int) * (size_t)rank);
        if (shapeCopy == NULL) {
            return NULL;
        }
        memcpy(shapeCopy, shape, sizeof(int) * (size_t)rank);
    }

    if (size > 0) {
        dataCopy = (float*)malloc(sizeof(float) * (size_t)size);
        if (dataCopy == NULL) {
            free(shapeCopy);
            return NULL;
        }
        if (source != NULL) {
            memcpy(dataCopy, source, sizeof(float) * (size_t)size);
        } else {
            memset(dataCopy, 0, sizeof(float) * (size_t)size);
        }
    }

    if (device->kind == TP_DEVICE_METAL) {
        if (vm.metalBackend == NULL || !tpMetalBackendIsAvailable(vm.metalBackend)) {
            free(shapeCopy);
            free(dataCopy);
            return NULL;
        }
        metalBuffer = tpMetalBufferCreate(vm.metalBackend,
                                          sizeof(float) * (size_t)(size > 0 ? size : 1),
                                          dataCopy);
        if (metalBuffer == NULL) {
            free(shapeCopy);
            free(dataCopy);
            return NULL;
        }
    }

    tensor = ALLOCATE_OBJ(ObjTensor, OBJ_TENSOR);
    tensor->rank = rank;
    tensor->size = size;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->contiguous = true;
    tensor->shape = shapeCopy;
    tensor->data = dataCopy;
    tensor->metalBuffer = metalBuffer;
    tensor->requiresGrad = false;
    tensor->grad = NULL;
    tensor->parentA = NULL;
    tensor->parentB = NULL;
    tensor->gradOp = TP_AUTOGRAD_NONE;
    tensor->gradAux = 0.0f;
    return tensor;
}

ObjClass* newClass(ObjString* name) {
    ObjClass* klass = ALLOCATE_OBJ(ObjClass, OBJ_CLASS);
    klass->name = name;
    klass->superClass = NULL;
    initTable(&klass->methods);
    return klass;
}

ObjInstance* newInstance(ObjClass* klass) {
    ObjInstance* instance = ALLOCATE_OBJ(ObjInstance, OBJ_INSTANCE);
    instance->klass = klass;
    initTable(&instance->fields);
    return instance;
}

ObjBoundMethod* newBoundMethod(Value receiver, Value method) {
    ObjBoundMethod* bound = ALLOCATE_OBJ(ObjBoundMethod, OBJ_BOUND_METHOD);
    bound->receiver = receiver;
    bound->method = method;
    return bound;
}

void freeObjects(void) {
    Obj* object = vm.objects;
    while (object != NULL) {
        Obj* next = object->next;
        freeObject(object);
        object = next;
    }
    vm.objects = NULL;
}

int sweepUnmarkedObjects(void) {
    int freed = 0;
    Obj* previous = NULL;
    Obj* object = vm.objects;

    while (object != NULL) {
        if (object->isMarked) {
            object->isMarked = false;
            previous = object;
            object = object->next;
            continue;
        }

        Obj* unreached = object;
        object = object->next;
        if (previous != NULL) {
            previous->next = object;
        } else {
            vm.objects = object;
        }
        freeObject(unreached);
        freed++;
    }

    return freed;
}

int countObjects(void) {
    return (int)vm.objectCount;
}

bool getNativeObjectAttribute(Value object, ObjString* name, Value* result) {
    if (result == NULL || !IS_OBJ(object)) {
        return false;
    }

    if (IS_DEVICE(object)) {
        ObjDevice* device = AS_DEVICE(object);
        if (strcmp(name->chars, "name") == 0) {
            *result = OBJ_VAL(device->name);
            return true;
        }
        if (strcmp(name->chars, "kind") == 0) {
            *result = OBJ_VAL(device->name);
            return true;
        }
        return false;
    }

    if (IS_DTYPE(object)) {
        ObjDType* dtype = AS_DTYPE(object);
        if (strcmp(name->chars, "name") == 0) {
            *result = OBJ_VAL(dtype->name);
            return true;
        }
        return false;
    }

    if (IS_TENSOR(object)) {
        ObjTensor* tensor = AS_TENSOR(object);
        if (strcmp(name->chars, "shape") == 0) {
            *result = OBJ_VAL(makeIntList(tensor->shape, tensor->rank));
            return true;
        }
        if (strcmp(name->chars, "rank") == 0) {
            *result = NUMBER_VAL((double)tensor->rank);
            return true;
        }
        if (strcmp(name->chars, "size") == 0) {
            *result = NUMBER_VAL((double)tensor->size);
            return true;
        }
        if (strcmp(name->chars, "dtype") == 0) {
            *result = OBJ_VAL(tensor->dtype);
            return true;
        }
        if (strcmp(name->chars, "device") == 0) {
            *result = OBJ_VAL(tensor->device);
            return true;
        }
        if (strcmp(name->chars, "contiguous") == 0) {
            *result = BOOL_VAL(tensor->contiguous);
            return true;
        }
        if (strcmp(name->chars, "requires_grad") == 0) {
            *result = BOOL_VAL(tensor->requiresGrad);
            return true;
        }
        if (strcmp(name->chars, "grad") == 0) {
            *result = tensor->grad != NULL ? OBJ_VAL(tensor->grad) : NIL_VAL;
            return true;
        }
        if (strcmp(name->chars, "strides") == 0) {
            int* strides;
            ObjList* list;

            if (tensor->rank == 0) {
                *result = OBJ_VAL(newList());
                return true;
            }

            strides = (int*)malloc(sizeof(int) * (size_t)tensor->rank);
            if (strides == NULL) {
                return false;
            }
            fillTensorStrides(tensor, strides);
            list = makeIntList(strides, tensor->rank);
            free(strides);
            *result = OBJ_VAL(list);
            return true;
        }
        return false;
    }

    return false;
}

void appendNativeObjectDirEntries(ObjList* out, Value object) {
    if (out == NULL || !IS_OBJ(object)) {
        return;
    }

    if (IS_DEVICE(object)) {
        writeValueArray(&out->items, OBJ_VAL(copyString("name", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("kind", 4)));
        return;
    }

    if (IS_DTYPE(object)) {
        writeValueArray(&out->items, OBJ_VAL(copyString("name", 4)));
        return;
    }

    if (IS_TENSOR(object)) {
        writeValueArray(&out->items, OBJ_VAL(copyString("shape", 5)));
        writeValueArray(&out->items, OBJ_VAL(copyString("rank", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("size", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("dtype", 5)));
        writeValueArray(&out->items, OBJ_VAL(copyString("device", 6)));
        writeValueArray(&out->items, OBJ_VAL(copyString("contiguous", 10)));
        writeValueArray(&out->items, OBJ_VAL(copyString("requires_grad", 13)));
        writeValueArray(&out->items, OBJ_VAL(copyString("grad", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("strides", 7)));
        writeValueArray(&out->items, OBJ_VAL(copyString("reshape", 7)));
        writeValueArray(&out->items, OBJ_VAL(copyString("to", 2)));
        writeValueArray(&out->items, OBJ_VAL(copyString("astype", 6)));
        writeValueArray(&out->items, OBJ_VAL(copyString("item", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("tolist", 6)));
        writeValueArray(&out->items, OBJ_VAL(copyString("backward", 8)));
        writeValueArray(&out->items, OBJ_VAL(copyString("zero_grad", 9)));
        writeValueArray(&out->items, OBJ_VAL(copyString("sum", 3)));
        writeValueArray(&out->items, OBJ_VAL(copyString("mean", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("max", 3)));
        writeValueArray(&out->items, OBJ_VAL(copyString("matmul", 6)));
        writeValueArray(&out->items, OBJ_VAL(copyString("relu", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("tanh", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("sigmoid", 7)));
        writeValueArray(&out->items, OBJ_VAL(copyString("gelu", 4)));
        writeValueArray(&out->items, OBJ_VAL(copyString("softmax", 7)));
        writeValueArray(&out->items, OBJ_VAL(copyString("layernorm", 9)));
    }
}

void printObject(Value value) {
    switch (OBJ_TYPE(value)) {
        case OBJ_STRING:
            printf("%s", AS_CSTRING(value));
            break;
        case OBJ_NATIVE:
            printf("<native fn>");
            break;
        case OBJ_FUNCTION:
            if (AS_FUNCTION(value)->name == NULL) {
                printf("<script>");
            } else {
                printf("<fn %s>", AS_FUNCTION(value)->name->chars);
            }
            break;
        case OBJ_CLOSURE:
            if (AS_CLOSURE(value)->function->name == NULL) {
                printf("<script>");
            } else {
                printf("<fn %s>", AS_CLOSURE(value)->function->name->chars);
            }
            break;
        case OBJ_ENVIRONMENT:
            printf("<env>");
            break;
        case OBJ_SET: {
            printf("{");
            ObjSet* set = AS_SET(value);
            bool first = true;
            for (int i = 0; i < set->table.capacity; i++) {
                Entry* entry = &set->table.entries[i];
                if (IS_NIL(entry->key)) continue;
                if (!first) printf(", ");
                printValueRepr(entry->key);
                first = false;
            }
            printf("}");
            break;
        }
        case OBJ_LIST: {
            printf("[");
            ObjList* list = AS_LIST(value);
            for (int i = 0; i < list->items.count; i++) {
                printValueRepr(list->items.values[i]);
                if (i < list->items.count - 1) printf(", ");
            }
            printf("]");
            break;
        }
        case OBJ_DICT: {
            printf("{");
            ObjDict* dict = AS_DICT(value);
            bool first = true;
            for (int i = 0; i < dict->table.capacity; i++) {
                Entry* entry = &dict->table.entries[i];
                if (IS_NIL(entry->key)) continue;
                if (!first) printf(", ");
                printValueRepr(entry->key);
                printf(": ");
                printValueRepr(entry->value);
                first = false;
            }
            printf("}");
            break;
        }
        case OBJ_TUPLE: {
            printf("(");
            ObjTuple* tuple = AS_TUPLE(value);
            for (int i = 0; i < tuple->items.count; i++) {
                printValueRepr(tuple->items.values[i]);
                if (i < tuple->items.count - 1) printf(", ");
            }
            if (tuple->items.count == 1) printf(",");
            printf(")");
            break;
        }
        case OBJ_BYTES: {
            printf("b'");
            ObjBytes* b = AS_BYTES(value);
            for (int i = 0; i < b->length; i++) {
                printf("\\x%02x", b->bytes[i]);
            }
            printf("'");
            break;
        }
        case OBJ_CLASS:
            printf("<class %s>", AS_CLASS(value)->name->chars);
            break;
        case OBJ_INSTANCE:
            printf("<%s instance>", AS_INSTANCE(value)->klass->name->chars);
            break;
        case OBJ_BOUND_METHOD:
            if (IS_CLOSURE(AS_BOUND_METHOD(value)->method)) {
                ObjFunction* method = AS_CLOSURE(AS_BOUND_METHOD(value)->method)->function;
                if (method->name == NULL) {
                    printf("<bound method>");
                } else {
                    printf("<bound method %s>", method->name->chars);
                }
            } else if (IS_FUNCTION(AS_BOUND_METHOD(value)->method)) {
                ObjFunction* method = AS_FUNCTION(AS_BOUND_METHOD(value)->method);
                if (method->name == NULL) {
                    printf("<bound method>");
                } else {
                    printf("<bound method %s>", method->name->chars);
                }
            } else {
                printf("<bound method>");
            }
            break;
        case OBJ_SLICE: {
            ObjSlice* slice = AS_SLICE(value);
            printf("slice(");
            printValueRepr(slice->start);
            printf(", ");
            printValueRepr(slice->stop);
            printf(", ");
            printValueRepr(slice->step);
            printf(")");
            break;
        }
        case OBJ_ITERATOR:
            printf("<iterator>");
            break;
        case OBJ_DEVICE:
            printf("%s", AS_DEVICE(value)->name->chars);
            break;
        case OBJ_DTYPE:
            printf("%s", AS_DTYPE(value)->name->chars);
            break;
        case OBJ_TENSOR: {
            ObjTensor* tensor = AS_TENSOR(value);
            int limit = tensor->size < 8 ? tensor->size : 8;
            int i;
            printf("tensor(shape=[");
            for (i = 0; i < tensor->rank; i++) {
                printf("%d", tensor->shape[i]);
                if (i < tensor->rank - 1) {
                    printf(", ");
                }
            }
            printf("], dtype=%s, device=%s",
                   tensor->dtype->name->chars,
                   tensor->device->name->chars);
            if (tensor->requiresGrad) {
                printf(", requires_grad=True");
            }
            printf(", data=[");
            for (i = 0; i < limit; i++) {
                printf("%g", tensor->data[i]);
                if (i < limit - 1) {
                    printf(", ");
                }
            }
            if (tensor->size > limit) {
                printf(", ...");
            }
            printf("])");
            break;
        }
        default:
            printf("<unknown obj>");
            break;
    }
}
