#include <stdio.h>
#include <time.h>
#include <string.h>

#include "tensorpy/common.h"
#include "tensorpy/object.h"
#include "tensorpy/value.h"
#include "tensorpy/vm.h"
#include "tensorpy/builtins.h"

// Built-in: print()
static Value printNative(int argCount, Value* args) {
    for (int i = 0; i < argCount; i++) {
        printValue(args[i]);
        if (i < argCount - 1) printf(" ");
    }
    printf("\n");
    fflush(stdout);
    return NIL_VAL;
}

// Built-in: clock() (useful for benchmarks)
static Value clockNative(int argCount, Value* args) {
    return NUMBER_VAL((double)clock() / CLOCKS_PER_SEC);
}

// Built-in: len()
static Value lenNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL; // Should raise error in full implementation
    
    if (IS_STRING(args[0])) {
        return NUMBER_VAL((double)AS_STRING(args[0])->length);
    }
    
    if (IS_SET(args[0])) {
        return NUMBER_VAL((double)AS_SET(args[0])->table.count);
    }

    if (IS_LIST(args[0])) {
        return NUMBER_VAL((double)AS_LIST(args[0])->items.count);
    }

    if (IS_DICT(args[0])) {
        return NUMBER_VAL((double)AS_DICT(args[0])->table.count);
    }

    if (IS_TUPLE(args[0])) {
        return NUMBER_VAL((double)AS_TUPLE(args[0])->items.count);
    }

    if (IS_BYTES(args[0])) {
        return NUMBER_VAL((double)AS_BYTES(args[0])->length);
    }
    
    return NUMBER_VAL(0);
}

// Built-in: set()
static Value setNative(int argCount, Value* args) {
    ObjSet* set = newSet();
    for (int i = 0; i < argCount; i++) {
        tableSet(&set->table, args[i], NIL_VAL);
    }
    return OBJ_VAL(set);
}

// Built-in: list()
static Value listNative(int argCount, Value* args) {
    ObjList* list = newList();
    for (int i = 0; i < argCount; i++) {
        writeValueArray(&list->items, args[i]);
    }
    return OBJ_VAL(list);
}

// Built-in: dict() - interleaved key, value
static Value dictNative(int argCount, Value* args) {
    ObjDict* dict = newDict();
    for (int i = 0; i < argCount; i += 2) {
        if (i + 1 < argCount) {
            tableSet(&dict->table, args[i], args[i + 1]);
        }
    }
    return OBJ_VAL(dict);
}

// Built-in: tuple()
static Value tupleNative(int argCount, Value* args) {
    ObjTuple* tuple = newTuple();
    for (int i = 0; i < argCount; i++) {
        writeValueArray(&tuple->items, args[i]);
    }
    return OBJ_VAL(tuple);
}

// Built-in: abs()
static Value absNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_NUMBER(args[0])) return NIL_VAL;
    double val = AS_NUMBER(args[0]);
    return NUMBER_VAL(val < 0 ? -val : val);
}

// Built-in: min()
static Value minNative(int argCount, Value* args) {
    if (argCount == 0) return NIL_VAL;
    Value minVal = args[0];
    for (int i = 1; i < argCount; i++) {
        if (IS_NUMBER(args[i]) && IS_NUMBER(minVal)) {
            if (AS_NUMBER(args[i]) < AS_NUMBER(minVal)) minVal = args[i];
        }
    }
    return minVal;
}

// Built-in: max()
static Value maxNative(int argCount, Value* args) {
    if (argCount == 0) return NIL_VAL;
    Value maxVal = args[0];
    for (int i = 1; i < argCount; i++) {
        if (IS_NUMBER(args[i]) && IS_NUMBER(maxVal)) {
            if (AS_NUMBER(args[i]) > AS_NUMBER(maxVal)) maxVal = args[i];
        }
    }
    return maxVal;
}

// Built-in: sum()
static Value sumNative(int argCount, Value* args) {
    double total = 0;
    for (int i = 0; i < argCount; i++) {
        if (!IS_NUMBER(args[i])) return NIL_VAL;
        total += AS_NUMBER(args[i]);
    }
    return NUMBER_VAL(total);
}

// Built-in: all()
static Value allNative(int argCount, Value* args) {
    for (int i = 0; i < argCount; i++) {
        if (IS_BOOL(args[i]) && !AS_BOOL(args[i])) return BOOL_VAL(false);
        if (IS_NIL(args[i])) return BOOL_VAL(false);
        if (IS_NUMBER(args[i]) && AS_NUMBER(args[i]) == 0) return BOOL_VAL(false);
    }
    return BOOL_VAL(true);
}

// Built-in: any()
static Value anyNative(int argCount, Value* args) {
    for (int i = 0; i < argCount; i++) {
        if (IS_BOOL(args[i]) && AS_BOOL(args[i])) return BOOL_VAL(true);
        if (IS_NUMBER(args[i]) && AS_NUMBER(args[i]) != 0) return BOOL_VAL(true);
        if (IS_OBJ(args[i])) return BOOL_VAL(true); // Simplified
    }
    return BOOL_VAL(false);
}

// Built-in: str()
static Value strNative(int argCount, Value* args) {
    if (argCount != 1) return OBJ_VAL(copyString("", 0));
    
    if (IS_STRING(args[0])) return args[0];
    
    if (IS_NUMBER(args[0])) {
        char buf[32];
        int len = sprintf(buf, "%g", AS_NUMBER(args[0]));
        return OBJ_VAL(copyString(buf, len));
    }
    
    if (IS_BOOL(args[0])) {
        return AS_BOOL(args[0]) ? OBJ_VAL(copyString("True", 4)) : OBJ_VAL(copyString("False", 5));
    }
    
    if (IS_NIL(args[0])) {
        return OBJ_VAL(copyString("None", 4));
    }
    
    return OBJ_VAL(copyString("<object>", 8));
}

// Built-in: type()
static Value typeNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    
    switch (args[0].type) {
        case VAL_BOOL:   return OBJ_VAL(copyString("bool", 4));
        case VAL_NIL:    return OBJ_VAL(copyString("NoneType", 8));
        case VAL_NUMBER: return OBJ_VAL(copyString("float", 5));
        case VAL_OBJ: {
            switch (OBJ_TYPE(args[0])) {
                case OBJ_STRING:   return OBJ_VAL(copyString("str", 3));
                case OBJ_NATIVE:   return OBJ_VAL(copyString("builtin_function_or_method", 26));
                case OBJ_SET:      return OBJ_VAL(copyString("set", 3));
                case OBJ_LIST:     return OBJ_VAL(copyString("list", 4));
                case OBJ_DICT:     return OBJ_VAL(copyString("dict", 4));
                case OBJ_TUPLE:    return OBJ_VAL(copyString("tuple", 5));
                case OBJ_BYTES:    return OBJ_VAL(copyString("bytes", 5));
                case OBJ_CLASS:    return OBJ_VAL(copyString("type", 4));
                case OBJ_INSTANCE: return OBJ_VAL(AS_INSTANCE(args[0])->klass->name);
                default:           return OBJ_VAL(copyString("object", 6));
            }
        }
    }
    return NIL_VAL;
}

int compareValues(const void* a, const void* b) {
    Value va = *(const Value*)a;
    Value vb = *(const Value*)b;
    
    // Sort logic, simplifid for numbers and strings
    if (IS_NUMBER(va) && IS_NUMBER(vb)) {
        double da = AS_NUMBER(va);
        double db = AS_NUMBER(vb);
        return (da > db) - (da < db);
    } else if (IS_STRING(va) && IS_STRING(vb)) {
        ObjString* sa = AS_STRING(va);
        ObjString* sb = AS_STRING(vb);
        return strcmp(sa->chars, sb->chars);
    }
    return 0; // Uncomparable
}

// Built-in: sorted()
static Value sortedNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    if (!IS_LIST(args[0])) return NIL_VAL; // Expand to other iterables later
    ObjList* list = AS_LIST(args[0]);
    
    ObjList* result = newList();
    for (int i = 0; i < list->items.count; i++) {
        writeValueArray(&result->items, list->items.values[i]);
    }
    
    qsort(result->items.values, result->items.count, sizeof(Value), compareValues);
    return OBJ_VAL(result);
}

// Built-in: enumerate()
static Value enumerateNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    
    if (IS_LIST(args[0])) {
        ObjList* list = AS_LIST(args[0]);
        ObjList* result = newList();
        for (int i = 0; i < list->items.count; i++) {
            ObjTuple* tuple = newTuple();
            writeValueArray(&tuple->items, NUMBER_VAL((double)i));
            writeValueArray(&tuple->items, list->items.values[i]);
            writeValueArray(&result->items, OBJ_VAL(tuple));
        }
        return OBJ_VAL(result);
    }
    
    return NIL_VAL;
}

// Built-in: range()
static Value rangeNative(int argCount, Value* args) {
    if (argCount < 1 || argCount > 3) return NIL_VAL;
    
    double start = 0;
    double stop = 0;
    double step = 1;
    
    if (argCount == 1) {
        if (!IS_NUMBER(args[0])) return NIL_VAL;
        stop = AS_NUMBER(args[0]);
    } else if (argCount == 2) {
        if (!IS_NUMBER(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
        start = AS_NUMBER(args[0]);
        stop = AS_NUMBER(args[1]);
    } else if (argCount == 3) {
        if (!IS_NUMBER(args[0]) || !IS_NUMBER(args[1]) || !IS_NUMBER(args[2])) return NIL_VAL;
        start = AS_NUMBER(args[0]);
        stop = AS_NUMBER(args[1]);
        step = AS_NUMBER(args[2]);
    }
    
    if (step == 0) return NIL_VAL;
    
    ObjList* result = newList();
    for (double i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
        writeValueArray(&result->items, NUMBER_VAL(i));
    }
    
    return OBJ_VAL(result);
}

static void defineNative(const char* name, NativeFn function) {
    Value key = OBJ_VAL(copyString(name, (int)strlen(name)));
    Value native = OBJ_VAL(newNative(function));
    tableSet(&vm.globals, key, native);
}

void registerBuiltins() {
    defineNative("print", printNative);
    defineNative("len", lenNative);
    defineNative("type", typeNative);
    defineNative("abs", absNative);
    defineNative("min", minNative);
    defineNative("max", maxNative);
    defineNative("sum", sumNative);
    defineNative("all", allNative);
    defineNative("any", anyNative);
    defineNative("set", setNative);
    defineNative("list", listNative);
    defineNative("dict", dictNative);
    defineNative("tuple", tupleNative);
    defineNative("clock", clockNative);
    defineNative("sorted", sortedNative);
    defineNative("enumerate", enumerateNative);
    defineNative("range", rangeNative);
    defineNative("str", strNative);
}
