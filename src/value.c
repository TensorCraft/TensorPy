#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "tensorpy/value.h"
#include "tensorpy/memory.h"
#include "tensorpy/object.h"

bool isNumericValue(Value value) {
    return value.type == VAL_NUMBER || IS_INT(value);
}

double numericValue(Value value) {
    if (value.type == VAL_NUMBER) {
        return value.as.number;
    }
    if (IS_INT(value)) {
        return intToDouble(AS_INT(value));
    }
    return 0.0;
}

static void printEscapedString(ObjString* string) {
    printf("\"");
    for (int i = 0; i < string->length; i++) {
        char c = string->chars[i];
        switch (c) {
            case '\\':
                printf("\\\\");
                break;
            case '"':
                printf("\\\"");
                break;
            case '\n':
                printf("\\n");
                break;
            case '\r':
                printf("\\r");
                break;
            case '\t':
                printf("\\t");
                break;
            default:
                printf("%c", c);
                break;
        }
    }
    printf("\"");
}

void initValueArray(ValueArray* array) {
    array->values = NULL;
    array->capacity = 0;
    array->count = 0;
}

void writeValueArray(ValueArray* array, Value value) {
    if (array->capacity < array->count + 1) {
        int oldCapacity = array->capacity;
        array->capacity = oldCapacity < 8 ? 8 : oldCapacity * 2;
        array->values = realloc(array->values, sizeof(Value) * array->capacity);
    }

    array->values[array->count] = value;
    array->count++;
}

void freeValueArray(ValueArray* array) {
    free(array->values);
    initValueArray(array);
}

void printValue(Value value) {
    switch (value.type) {
        case VAL_BOOL:
            printf(AS_BOOL(value) ? "True" : "False");
            break;
        case VAL_NIL:
            printf("None");
            break;
        case VAL_NUMBER:
            printf("%g", AS_NUMBER(value));
            break;
        case VAL_OBJ:
            if (IS_INT(value)) {
                char* chars = NULL;
                int length = 0;
                intToString(AS_INT(value), &chars, &length);
                if (chars != NULL) {
                    printf("%s", chars);
                    tpMemFree(chars);
                } else {
                    printf("0");
                }
            } else {
                printObject(value);
            }
            break;
    }
}

void printValueRepr(Value value) {
    switch (value.type) {
        case VAL_BOOL:
            printf(AS_BOOL(value) ? "True" : "False");
            break;
        case VAL_NIL:
            printf("None");
            break;
        case VAL_NUMBER:
            printf("%g", AS_NUMBER(value));
            break;
        case VAL_OBJ:
            if (IS_STRING(value)) {
                printEscapedString(AS_STRING(value));
            } else if (IS_INT(value)) {
                char* chars = NULL;
                int length = 0;
                intToString(AS_INT(value), &chars, &length);
                if (chars != NULL) {
                    printf("%s", chars);
                    tpMemFree(chars);
                } else {
                    printf("0");
                }
            } else {
                printObject(value);
            }
            break;
    }
}

bool valuesEqual(Value a, Value b) {
    if (IS_NUMBER(a) && IS_NUMBER(b)) {
        return numericValue(a) == numericValue(b);
    }
    if (a.type != b.type) return false;
    switch (a.type) {
        case VAL_BOOL:   return AS_BOOL(a) == AS_BOOL(b);
        case VAL_NIL:    return true;
        case VAL_NUMBER: return AS_NUMBER(a) == AS_NUMBER(b);
        case VAL_OBJ: {
            if (IS_STRING(a) && IS_STRING(b)) {
                ObjString* sa = AS_STRING(a);
                ObjString* sb = AS_STRING(b);
                return sa->length == sb->length &&
                       memcmp(sa->chars, sb->chars, sa->length) == 0;
            }
            if (IS_INT(a) && IS_INT(b)) {
                ObjInt* ia = AS_INT(a);
                ObjInt* ib = AS_INT(b);
                if (ia->negative != ib->negative || ia->length != ib->length) return false;
                return memcmp(ia->digits, ib->digits, sizeof(uint32_t) * (size_t)ia->length) == 0;
            }
            return AS_OBJ(a) == AS_OBJ(b);
        }
    }
    return false;
}
