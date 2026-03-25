#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "tensorpy/object.h"
#include "tensorpy/memory.h"
#include "tensorpy/table.h"
#include "tensorpy/value.h"
#include "tensorpy/vm.h"

// In a real implementation, we'd have a memory manager
#define ALLOCATE_OBJ(type, objectType) \
    (type*)allocateObject(sizeof(type), objectType)

static bool isZeroInt(const ObjInt* value);

static Obj* allocateObject(size_t size, ObjType type) {
    if (vm.gcEnabled && vm.gcPauseDepth == 0 && vm.objectCount >= vm.nextGC) {
        gcCollect();
    }

    Obj* object = (Obj*)tpMemAlloc(size);
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

#define TP_INT_BASE 1000000000u

static void trimInt(ObjInt* value) {
    while (value->length > 1 && value->digits[value->length - 1] == 0) {
        value->length--;
    }
    if (value->length == 1 && value->digits[0] == 0) {
        value->negative = false;
    }
}

static ObjInt* allocateInt(int length) {
    ObjInt* value = ALLOCATE_OBJ(ObjInt, OBJ_INT);
    value->negative = false;
    value->length = length;
    value->digits = (uint32_t*)tpMemCalloc((size_t)length, sizeof(uint32_t));
    if (value->digits == NULL) {
        tpMemFree(value);
        return NULL;
    }
    return value;
}

static ObjInt* copyInt(const ObjInt* source) {
    ObjInt* value = allocateInt(source->length);
    if (value == NULL) {
        return NULL;
    }
    value->negative = source->negative;
    memcpy(value->digits, source->digits, sizeof(uint32_t) * (size_t)source->length);
    return value;
}

static ObjInt* intFromUint64(uint64_t magnitude, bool negative) {
    ObjInt* value;
    int length = 0;
    uint64_t temp = magnitude;

    do {
        length++;
        temp /= TP_INT_BASE;
    } while (temp != 0);

    value = allocateInt(length);
    if (value == NULL) {
        return NULL;
    }
    value->negative = negative && magnitude != 0;
    temp = magnitude;
    for (int i = 0; i < length; i++) {
        value->digits[i] = (uint32_t)(temp % TP_INT_BASE);
        temp /= TP_INT_BASE;
    }
    trimInt(value);
    return value;
}

static int compareAbsInt(const ObjInt* a, const ObjInt* b) {
    if (a->length != b->length) {
        return a->length < b->length ? -1 : 1;
    }
    for (int i = a->length - 1; i >= 0; i--) {
        if (a->digits[i] != b->digits[i]) {
            return a->digits[i] < b->digits[i] ? -1 : 1;
        }
    }
    return 0;
}

static ObjInt* addAbsInt(const ObjInt* a, const ObjInt* b) {
    int max = a->length > b->length ? a->length : b->length;
    ObjInt* out = allocateInt(max + 1);
    uint64_t carry = 0;

    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < max; i++) {
        uint64_t sum = carry;
        if (i < a->length) sum += a->digits[i];
        if (i < b->length) sum += b->digits[i];
        out->digits[i] = (uint32_t)(sum % TP_INT_BASE);
        carry = sum / TP_INT_BASE;
    }
    out->digits[max] = (uint32_t)carry;
    out->length = max + (carry != 0 ? 1 : 0);
    trimInt(out);
    return out;
}

static ObjInt* subAbsInt(const ObjInt* a, const ObjInt* b) {
    ObjInt* out = allocateInt(a->length);
    int64_t borrow = 0;

    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < a->length; i++) {
        int64_t diff = (int64_t)a->digits[i] - (i < b->length ? b->digits[i] : 0) - borrow;
        if (diff < 0) {
            diff += TP_INT_BASE;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out->digits[i] = (uint32_t)diff;
    }
    trimInt(out);
    return out;
}

static ObjInt* mulAbsInt(const ObjInt* a, const ObjInt* b) {
    int length = a->length + b->length;
    uint64_t* accum = (uint64_t*)tpMemCalloc((size_t)length, sizeof(uint64_t));
    ObjInt* out;

    if (accum == NULL) {
        return NULL;
    }
    for (int i = 0; i < a->length; i++) {
        for (int j = 0; j < b->length; j++) {
            accum[i + j] += (uint64_t)a->digits[i] * (uint64_t)b->digits[j];
        }
    }
    out = allocateInt(length);
    if (out == NULL) {
        tpMemFree(accum);
        return NULL;
    }
    uint64_t carry = 0;
    for (int i = 0; i < length; i++) {
        uint64_t cur = accum[i] + carry;
        out->digits[i] = (uint32_t)(cur % TP_INT_BASE);
        carry = cur / TP_INT_BASE;
    }
    tpMemFree(accum);
    trimInt(out);
    return out;
}

static ObjInt* shiftAbsIntAndAddDigit(const ObjInt* value, uint32_t digit) {
    ObjInt* out;
    if (value->length == 1 && value->digits[0] == 0) {
        return intFromUint64((uint64_t)digit, false);
    }
    out = allocateInt(value->length + 1);
    if (out == NULL) {
        return NULL;
    }
    out->digits[0] = digit;
    memcpy(out->digits + 1, value->digits, sizeof(uint32_t) * (size_t)value->length);
    trimInt(out);
    return out;
}

static ObjInt* mulAbsIntSmall(const ObjInt* value, uint32_t factor) {
    ObjInt* out;
    uint64_t carry = 0;

    if (factor == 0) {
        return intFromUint64(0, false);
    }
    if (factor == 1) {
        return copyInt(value);
    }
    out = allocateInt(value->length + 1);
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < value->length; i++) {
        uint64_t cur = (uint64_t)value->digits[i] * factor + carry;
        out->digits[i] = (uint32_t)(cur % TP_INT_BASE);
        carry = cur / TP_INT_BASE;
    }
    out->digits[value->length] = (uint32_t)carry;
    trimInt(out);
    return out;
}

int intCompare(const ObjInt* a, const ObjInt* b) {
    if (a->negative != b->negative) {
        return a->negative ? -1 : 1;
    }
    int cmp = compareAbsInt(a, b);
    return a->negative ? -cmp : cmp;
}

ObjInt* intAdd(const ObjInt* a, const ObjInt* b) {
    if (a->negative == b->negative) {
        ObjInt* out = addAbsInt(a, b);
        if (out != NULL) out->negative = a->negative;
        return out;
    }
    int cmp = compareAbsInt(a, b);
    if (cmp == 0) {
        return intFromUint64(0, false);
    }
    if (cmp > 0) {
        ObjInt* out = subAbsInt(a, b);
        if (out != NULL) out->negative = a->negative;
        return out;
    }
    ObjInt* out = subAbsInt(b, a);
    if (out != NULL) out->negative = b->negative;
    return out;
}

ObjInt* intSub(const ObjInt* a, const ObjInt* b) {
    ObjInt negB = *b;
    negB.negative = !b->negative;
    return intAdd(a, &negB);
}

ObjInt* intMul(const ObjInt* a, const ObjInt* b) {
    ObjInt* out = mulAbsInt(a, b);
    if (out != NULL) {
        out->negative = (a->negative != b->negative) && !intIsZero(out);
    }
    return out;
}

bool intDivMod(const ObjInt* a, const ObjInt* b, ObjInt** outQuotient, ObjInt** outRemainder) {
    ObjInt absA = *a;
    ObjInt absB = *b;
    ObjInt* quotient;
    ObjInt* remainder;
    int cmp;

    if (outQuotient == NULL || outRemainder == NULL || intIsZero(b)) {
        return false;
    }

    vm.gcPauseDepth++;

    absA.negative = false;
    absB.negative = false;
    cmp = compareAbsInt(&absA, &absB);
    if (cmp < 0) {
        quotient = intFromUint64(0, false);
        remainder = copyInt(&absA);
        if (quotient == NULL || remainder == NULL) {
            vm.gcPauseDepth--;
            return false;
        }
    } else if (cmp == 0) {
        quotient = intFromUint64(1, false);
        remainder = intFromUint64(0, false);
        if (quotient == NULL || remainder == NULL) {
            vm.gcPauseDepth--;
            return false;
        }
    } else {
        quotient = allocateInt(absA.length);
        remainder = intFromUint64(0, false);
        if (quotient == NULL || remainder == NULL) {
            vm.gcPauseDepth--;
            return false;
        }

        for (int i = absA.length - 1; i >= 0; i--) {
            ObjInt* nextRemainder = shiftAbsIntAndAddDigit(remainder, absA.digits[i]);
            uint32_t low = 0;
            uint32_t high = TP_INT_BASE - 1;
            uint32_t qdigit = 0;
            remainder = nextRemainder;
            if (remainder == NULL) {
                vm.gcPauseDepth--;
                return false;
            }
            while (low <= high) {
                uint32_t mid = low + (high - low) / 2;
                ObjInt* product = mulAbsIntSmall(&absB, mid);
                int rel;
                if (product == NULL) {
                    vm.gcPauseDepth--;
                    return false;
                }
                rel = compareAbsInt(product, remainder);
                if (rel <= 0) {
                    qdigit = mid;
                    low = mid + 1;
                } else {
                    if (mid == 0) break;
                    high = mid - 1;
                }
            }
            quotient->digits[i] = qdigit;
            if (qdigit != 0) {
                ObjInt* product = mulAbsIntSmall(&absB, qdigit);
                ObjInt* next = subAbsInt(remainder, product);
                if (product == NULL || next == NULL) {
                    vm.gcPauseDepth--;
                    return false;
                }
                remainder = next;
            }
        }
        trimInt(quotient);
        trimInt(remainder);
    }

    if ((a->negative != b->negative) && !intIsZero(remainder)) {
        ObjInt* one = intFromUint64(1, false);
        ObjInt* qAdj = addAbsInt(quotient, one);
        ObjInt* rAdj = subAbsInt(&absB, remainder);
        if (one == NULL || qAdj == NULL || rAdj == NULL) {
            vm.gcPauseDepth--;
            return false;
        }
        quotient = qAdj;
        remainder = rAdj;
        quotient->negative = true;
        remainder->negative = b->negative;
    } else {
        quotient->negative = (a->negative != b->negative) && !intIsZero(quotient);
        remainder->negative = b->negative && !intIsZero(remainder);
    }

    vm.gcPauseDepth--;
    *outQuotient = quotient;
    *outRemainder = remainder;
    return true;
}

uint32_t intHash(const ObjInt* value) {
    uint32_t hash = value->negative ? 0x9e3779b9u : 0x7f4a7c15u;
    for (int i = 0; i < value->length; i++) {
        hash ^= value->digits[i] + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    }
    return hash;
}

double intToDouble(const ObjInt* value) {
    double result = 0.0;
    double place = 1.0;

    for (int i = 0; i < value->length; i++) {
        result += (double)value->digits[i] * place;
        place *= (double)TP_INT_BASE;
    }
    return value->negative ? -result : result;
}

static bool isZeroInt(const ObjInt* value) {
    return value->length == 1 && value->digits[0] == 0;
}

bool intIsZero(const ObjInt* value) {
    return isZeroInt(value);
}

bool intToInt64Exact(const ObjInt* value, int64_t* out) {
    uint64_t result = 0;
    uint64_t limit = value->negative ? ((uint64_t)INT64_MAX + 1u) : (uint64_t)INT64_MAX;

    if (out == NULL) {
        return false;
    }
    for (int i = value->length - 1; i >= 0; i--) {
        if (result > limit / TP_INT_BASE) {
            return false;
        }
        result *= TP_INT_BASE;
        if (result > limit - value->digits[i]) {
            return false;
        }
        result += value->digits[i];
    }
    if (value->negative) {
        if (result == (uint64_t)INT64_MAX + 1u) {
            *out = INT64_MIN;
        } else {
            *out = -(int64_t)result;
        }
    } else {
        *out = (int64_t)result;
    }
    return true;
}

static ObjInt* parseIntString(const char* chars, int length) {
    int start = 0;
    bool negative = false;
    ObjInt* result = intFromUint64(0, false);

    if (result == NULL) {
        return NULL;
    }
    vm.gcPauseDepth++;
    if (length > 0 && (chars[0] == '+' || chars[0] == '-')) {
        negative = chars[0] == '-';
        start = 1;
    }
    for (int i = start; i < length; i++) {
        char c = chars[i];
        if (c < '0' || c > '9') {
            vm.gcPauseDepth--;
            return NULL;
        }
        uint32_t digit = (uint32_t)(c - '0');
        ObjInt* ten = intFromUint64(10, false);
        ObjInt* digitInt = intFromUint64(digit, false);
        ObjInt* tmp = mulAbsInt(result, ten);
        ObjInt* next = addAbsInt(tmp, digitInt);
        if (next == NULL) {
            vm.gcPauseDepth--;
            return NULL;
        }
        result = next;
    }
    result->negative = negative && !isZeroInt(result);
    vm.gcPauseDepth--;
    return result;
}

static void intToCString(const ObjInt* value, char* buffer, size_t size) {
    int written = 0;
    if (value->negative && !isZeroInt(value)) {
        written += snprintf(buffer + written, size - (size_t)written, "-");
    }
    written += snprintf(buffer + written, size - (size_t)written, "%u", value->digits[value->length - 1]);
    for (int i = value->length - 2; i >= 0; i--) {
        written += snprintf(buffer + written, size - (size_t)written, "%09u", value->digits[i]);
    }
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
            tpMemFree(((ObjString*)object)->chars);
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
            tpMemFree(((ObjBytes*)object)->bytes);
            break;
        case OBJ_INT: {
            ObjInt* integer = (ObjInt*)object;
            tpMemFree(integer->digits);
            break;
        }
        case OBJ_DEVICE:
        case OBJ_DTYPE:
            break;
        case OBJ_TENSOR: {
            ObjTensor* tensor = (ObjTensor*)object;
            if (tensor->ownsMetalBuffer) {
                tpMetalBufferDestroy(tensor->metalBuffer);
            }
            tpMemFree(tensor->shape);
            if (tensor->ownsData) {
                tpMemFree(tensor->data);
            }
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
    tpMemFree(object);
}

ObjString* copyString(const char* chars, int length) {
    uint32_t hash = hashString(chars, length);
    ObjString* interned = tableFindString(&vm.strings, chars, length, hash);
    if (interned != NULL) return interned;

    char* heapChars = tpMemAlloc(length + 1);
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
        tpMemFree(chars);
        return interned;
    }

    ObjString* string = allocateString(chars, length, hash);
    tableSet(&vm.strings, OBJ_VAL(string), NIL_VAL);
    return string;
}

ObjBytes* newBytes(int length, const uint8_t* source) {
    ObjBytes* bytes = ALLOCATE_OBJ(ObjBytes, OBJ_BYTES);
    bytes->length = length;
    bytes->bytes = (uint8_t*)tpMemAlloc((size_t)length);
    if (source != NULL) {
        memcpy(bytes->bytes, source, length);
    }
    return bytes;
}

ObjInt* newIntFromInt64(int64_t value) {
    uint64_t magnitude;
    bool negative = value < 0;

    if (value == 0) {
        return intFromUint64(0, false);
    }
    magnitude = negative ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
    return intFromUint64(magnitude, negative);
}

ObjInt* newIntFromString(const char* chars, int length) {
    return parseIntString(chars, length);
}

ObjInt* newIntCopy(const ObjInt* source) {
    return copyInt(source);
}

void intToString(const ObjInt* value, char** outChars, int* outLength) {
    size_t size;
    char* chars;

    if (value == NULL || outChars == NULL || outLength == NULL) {
        return;
    }
    size = (size_t)value->length * 10 + 2;
    chars = (char*)tpMemAlloc(size);
    if (chars == NULL) {
        *outChars = NULL;
        *outLength = 0;
        return;
    }
    intToCString(value, chars, size);
    *outLength = (int)strlen(chars);
    *outChars = chars;
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
        shapeCopy = (int*)tpMemAlloc(sizeof(int) * (size_t)rank);
        if (shapeCopy == NULL) {
            return NULL;
        }
        memcpy(shapeCopy, shape, sizeof(int) * (size_t)rank);
    }

    if (size > 0) {
        dataCopy = (float*)tpMemAlloc(sizeof(float) * (size_t)size);
        if (dataCopy == NULL) {
            tpMemFree(shapeCopy);
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
            tpMemFree(shapeCopy);
            tpMemFree(dataCopy);
            return NULL;
        }
        metalBuffer = tpMetalBufferCreate(vm.metalBackend,
                                          sizeof(float) * (size_t)(size > 0 ? size : 1),
                                          dataCopy);
        if (metalBuffer == NULL) {
            tpMemFree(shapeCopy);
            tpMemFree(dataCopy);
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
    tensor->ownsData = true;
    tensor->ownsMetalBuffer = metalBuffer != NULL;
    tensor->cpuDirty = false;
    tensor->metalDirty = false;
    tensor->requiresGrad = false;
    tensor->grad = NULL;
    tensor->parentA = NULL;
    tensor->parentB = NULL;
    tensor->gradOp = TP_AUTOGRAD_NONE;
    tensor->gradAux = 0.0f;
    return tensor;
}

ObjTensor* newTensorView(int rank,
                         const int* shape,
                         ObjDType* dtype,
                         ObjDevice* device,
                         float* data,
                         TPMetalBuffer* metalBuffer) {
    ObjTensor* tensor;
    int size;
    int* shapeCopy = NULL;

    if (dtype == NULL || device == NULL) {
        return NULL;
    }

    size = tensorElementCount(rank, shape);
    if (size < 0) {
        return NULL;
    }

    if (rank > 0) {
        shapeCopy = (int*)tpMemAlloc(sizeof(int) * (size_t)rank);
        if (shapeCopy == NULL) {
            return NULL;
        }
        memcpy(shapeCopy, shape, sizeof(int) * (size_t)rank);
    }

    tensor = ALLOCATE_OBJ(ObjTensor, OBJ_TENSOR);
    tensor->rank = rank;
    tensor->size = size;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->contiguous = true;
    tensor->shape = shapeCopy;
    tensor->data = data;
    tensor->metalBuffer = metalBuffer;
    tensor->ownsData = false;
    tensor->ownsMetalBuffer = false;
    tensor->cpuDirty = false;
    tensor->metalDirty = false;
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
            *result = NUMBER_VAL((double)device->kind);
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
            {
                ObjInt* integer = newIntFromInt64(tensor->rank);
                *result = integer != NULL ? OBJ_VAL(integer) : NIL_VAL;
            }
            return true;
        }
        if (strcmp(name->chars, "size") == 0) {
            {
                ObjInt* integer = newIntFromInt64(tensor->size);
                *result = integer != NULL ? OBJ_VAL(integer) : NIL_VAL;
            }
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

            strides = (int*)tpMemAlloc(sizeof(int) * (size_t)tensor->rank);
            if (strides == NULL) {
                return false;
            }
            fillTensorStrides(tensor, strides);
            list = makeIntList(strides, tensor->rank);
            tpMemFree(strides);
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
        case OBJ_INT: {
            ObjInt* integer = AS_INT(value);
            size_t size = (size_t)integer->length * 10 + 2;
            char* chars = (char*)malloc(size);
            if (chars == NULL) {
                printf("0");
                break;
            }
            intToCString(integer, chars, size);
            printf("%s", chars);
            free(chars);
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
