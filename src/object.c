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
    Obj* object = (Obj*)malloc(size);
    object->type = type;
    object->next = vm.objects; 
    vm.objects = object;
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
    function->globals = NULL;
    initValueArray(&function->defaults);
    initValueArray(&function->paramNames);
    initValueArray(&function->localNames);
    function->name = NULL;
    initChunk(&function->chunk);
    return function;
}

ObjFunction* cloneFunction(ObjFunction* function) {
    ObjFunction* clone = ALLOCATE_OBJ(ObjFunction, OBJ_FUNCTION);
    *clone = *function;
    clone->globals = NULL;
    return clone;
}

ObjNative* newNative(NativeFn function) {
    ObjNative* native = ALLOCATE_OBJ(ObjNative, OBJ_NATIVE);
    native->function = function;
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

ObjBoundMethod* newBoundMethod(Value receiver, ObjFunction* method) {
    ObjBoundMethod* bound = ALLOCATE_OBJ(ObjBoundMethod, OBJ_BOUND_METHOD);
    bound->receiver = receiver;
    bound->method = method;
    return bound;
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
            if (AS_BOUND_METHOD(value)->method->name == NULL) {
                printf("<bound method>");
            } else {
                printf("<bound method %s>", AS_BOUND_METHOD(value)->method->name->chars);
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
        default:
            printf("<unknown obj>");
            break;
    }
}
