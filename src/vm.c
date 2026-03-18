#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>

#include "tensorpy/common.h"
#include "tensorpy/builtins.h"
#include "tensorpy/compiler.h"
#include "tensorpy/debug.h"
#include "tensorpy/object.h"
#include "tensorpy/platform.h"
#include "tensorpy/scanner.h"
#include "tensorpy/vm.h"

VM vm;

static bool callValue(Value callee, int argCount);
static bool invokeBuiltinMethod(Value receiver, ObjString* name, int argCount);
static InterpretResult run();
static bool raiseException(Value exception);
static void resetStack(void);
static Value importModuleValue(ObjString* moduleName);
static InterpretResult interpretInGlobals(const char* source, const char* filename, ObjEnvironment* env);
static ObjClosure* bindFunctionForFrame(ObjFunction* function, CallFrame* frame);
static void syncFrameLocalToEnv(CallFrame* frame, int slot);
static bool environmentGet(ObjEnvironment* env, Value key, Value* value);
static void environmentSetLocal(ObjEnvironment* env, Value key, Value value);
static ObjEnvironment* currentGlobalEnv(CallFrame* frame);
static void markValue(Value value);
static void markObject(Obj* object);
static void markTable(Table* table);
static void markValueArray(ValueArray* array);
static void markRoots(void);
static void clearMarks(void);
static int countMarkedObjects(void);
static void markRootsForCollection(void);
static void updateNextGC(void);
static bool expandIterableArgs(Value iterable, Value* outArgs, int* outCount);
static bool collectExpandedArgs(Value* sourceArgs, int sourceCount, ObjTuple* starIndexes, Value* outArgs, int* outCount);
static bool collectExpandedKeywords(Value* kwSourceArgs, int kwSourceCount, ObjTuple* kwStarIndexes,
                                    Value* explicitKwNames, int explicitKwCount,
                                    Value* outKwNames, Value* outKwValues, int* outKwCount);

static char* readFileToBuffer(const char* path) {
    return platformReadTextFile(path);
}

static char* moduleNameToPath(const char* moduleName) {
    size_t length = strlen(moduleName);
    char* path = (char*)malloc(length + 1);
    for (size_t i = 0; i < length; i++) {
        path[i] = moduleName[i] == '.' ? '/' : moduleName[i];
    }
    path[length] = '\0';
    return path;
}

static char* readModuleFile(const char* moduleName) {
    char* normalized = moduleNameToPath(moduleName);
    const char* roots[] = {"modules", "lib", "."};
    for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]); i++) {
        size_t pathLen = strlen(roots[i]) + 1 + strlen(normalized) + 3;
        char* path = (char*)malloc(pathLen + 1);
        snprintf(path, pathLen + 1, "%s/%s.py", roots[i], normalized);

        char* source = readFileToBuffer(path);
        free(path);
        if (source != NULL) {
            free(normalized);
            return source;
        }

        pathLen = strlen(roots[i]) + 1 + strlen(normalized) + 12;
        path = (char*)malloc(pathLen + 1);
        snprintf(path, pathLen + 1, "%s/%s/__init__.py", roots[i], normalized);
        source = readFileToBuffer(path);
        free(path);
        if (source != NULL) {
            free(normalized);
            return source;
        }
    }

    size_t fallbackLen = strlen(normalized) + 3;
    char* fallback = (char*)malloc(fallbackLen + 1);
    snprintf(fallback, fallbackLen + 1, "%s.py", normalized);
    char* source = readFileToBuffer(fallback);
    free(fallback);
    free(normalized);
    return source;
}

static ObjString* parentModuleName(ObjString* moduleName) {
    const char* dot = strrchr(moduleName->chars, '.');
    if (dot == NULL) return NULL;
    return copyString(moduleName->chars, (int)(dot - moduleName->chars));
}

static ObjString* childModuleName(ObjString* moduleName) {
    const char* dot = strrchr(moduleName->chars, '.');
    if (dot == NULL) return NULL;
    return copyString(dot + 1, (int)strlen(dot + 1));
}

static bool tryImportModuleProperty(ObjInstance* instance, ObjString* name, Value* value) {
    if (instance->klass != vm.moduleClass) {
        return false;
    }

    Value moduleNameValue;
    Value moduleNameKey = OBJ_VAL(copyString("__name__", 8));
    if (!tableGet(&instance->fields, moduleNameKey, &moduleNameValue) || !IS_STRING(moduleNameValue)) {
        return false;
    }

    ObjString* moduleName = AS_STRING(moduleNameValue);
    int fullLength = moduleName->length + 1 + name->length;
    char* fullName = (char*)malloc((size_t)fullLength + 1);
    snprintf(fullName, (size_t)fullLength + 1, "%s.%s", moduleName->chars, name->chars);
    ObjString* fullModuleName = takeString(fullName, fullLength);
    *value = importModuleValue(fullModuleName);
    if (IS_NIL(*value)) {
        return false;
    }

    tableSet(&instance->fields, OBJ_VAL(name), *value);
    return true;
}

static bool isClassOrSubclass(ObjClass* klass, ObjClass* expected) {
    while (klass != NULL) {
        if (klass == expected) {
            return true;
        }
        klass = klass->superClass;
    }
    return false;
}

static Value makeExceptionArgsTuple(const char* message) {
    ObjTuple* args = newTuple();
    if (message != NULL) {
        writeValueArray(&args->items, OBJ_VAL(copyString(message, (int)strlen(message))));
    }
    return OBJ_VAL(args);
}

static ObjInstance* newExceptionInstance(ObjClass* klass, const char* message) {
    ObjInstance* instance = newInstance(klass);
    Value messageKey = OBJ_VAL(copyString("message", 7));
    const char* storedMessage = message != NULL ? message : "";
    tableSet(&instance->fields, messageKey,
             OBJ_VAL(copyString(storedMessage, (int)strlen(storedMessage))));
    Value argsKey = OBJ_VAL(copyString("args", 4));
    tableSet(&instance->fields, argsKey, makeExceptionArgsTuple(message));
    return instance;
}

static void defineGlobalClass(const char* name, const char* superName) {
    ObjString* key = copyString(name, (int)strlen(name));
    ObjClass* klass = newClass(key);
    if (superName != NULL) {
        Value superValue;
        Value superKey = OBJ_VAL(copyString(superName, (int)strlen(superName)));
        if (tableGet(vm.globalEnv->table, superKey, &superValue) && IS_CLASS(superValue)) {
            klass->superClass = AS_CLASS(superValue);
        }
    }
    tableSet(vm.globalEnv->table, OBJ_VAL(key), OBJ_VAL(klass));
}

static bool valueToExceptionMessage(Value value, char* buffer, size_t size) {
    if (IS_STRING(value)) {
        snprintf(buffer, size, "%s", AS_STRING(value)->chars);
        return true;
    }
    if (IS_NUMBER(value)) {
        snprintf(buffer, size, "%g", AS_NUMBER(value));
        return true;
    }
    if (IS_BOOL(value)) {
        snprintf(buffer, size, "%s", AS_BOOL(value) ? "True" : "False");
        return true;
    }
    if (IS_NIL(value)) {
        snprintf(buffer, size, "None");
        return true;
    }
    return false;
}

static Value getExceptionTypeByName(const char* name) {
    Value classValue;
    Value key = OBJ_VAL(copyString(name, (int)strlen(name)));
    if (tableGet(vm.globalEnv->table, key, &classValue)) {
        return classValue;
    }
    return NIL_VAL;
}

static bool isExceptionClass(ObjClass* klass) {
    Value exceptionType = getExceptionTypeByName("Exception");
    return IS_CLASS(exceptionType) && isClassOrSubclass(klass, AS_CLASS(exceptionType));
}

static const char* exceptionMessageArg(Value value, char* buffer, size_t size) {
    if (IS_STRING(value)) {
        return AS_STRING(value)->chars;
    }
    if (valueToExceptionMessage(value, buffer, size)) {
        return buffer;
    }
    return NULL;
}

static Value createExceptionValue(const char* className, const char* message) {
    Value classValue = getExceptionTypeByName(className);
    if (!IS_CLASS(classValue)) {
        return OBJ_VAL(copyString(message, (int)strlen(message)));
    }

    return OBJ_VAL(newExceptionInstance(AS_CLASS(classValue), message));
}

static bool environmentGet(ObjEnvironment* env, Value key, Value* value) {
    while (env != NULL) {
        if (tableGet(env->table, key, value)) {
            return true;
        }
        env = env->parent;
    }
    return false;
}

static void environmentSetLocal(ObjEnvironment* env, Value key, Value value) {
    tableSet(env->table, key, value);
}

static ObjEnvironment* currentGlobalEnv(CallFrame* frame) {
    ObjEnvironment* env = frame->env;
    while (env != NULL && env->parent != NULL) {
        env = env->parent;
    }
    return env != NULL ? env : vm.globalEnv;
}

static ObjClosure* bindFunctionForFrame(ObjFunction* function, CallFrame* frame) {
    return newClosure(function, frame->env);
}

static void syncFrameLocalToEnv(CallFrame* frame, int slot) {
    if (slot < 0 || slot >= frame->function->localNames.count) {
        return;
    }

    Value localName = frame->function->localNames.values[slot];
    if (!IS_STRING(localName)) {
        return;
    }

    environmentSetLocal(frame->env, localName, frame->slots[slot]);
}

static void formatException(Value exception, char* buffer, size_t size) {
    if (IS_INSTANCE(exception)) {
        ObjInstance* instance = AS_INSTANCE(exception);
        Value message;
        Value messageKey = OBJ_VAL(copyString("message", 7));
        if (tableGet(&instance->fields, messageKey, &message) && IS_STRING(message)) {
            if (AS_STRING(message)->length > 0) {
                snprintf(buffer, size, "%s: %s", instance->klass->name->chars, AS_STRING(message)->chars);
                return;
            }
        }
        snprintf(buffer, size, "%s", instance->klass->name->chars);
        return;
    }
    if (IS_CLASS(exception)) {
        snprintf(buffer, size, "%s", AS_CLASS(exception)->name->chars);
        return;
    }
    if (IS_STRING(exception)) {
        snprintf(buffer, size, "%s", AS_STRING(exception)->chars);
        return;
    }
    if (!valueToExceptionMessage(exception, buffer, size)) {
        snprintf(buffer, size, "Exception raised.");
    }
}

static bool exceptionMatches(Value exception, Value expectedType) {
    if (IS_NIL(expectedType)) return true;
    if (IS_CLASS(expectedType)) {
        if (IS_INSTANCE(exception)) {
            return isClassOrSubclass(AS_INSTANCE(exception)->klass, AS_CLASS(expectedType));
        }
        if (IS_CLASS(exception)) {
            return isClassOrSubclass(AS_CLASS(exception), AS_CLASS(expectedType));
        }
        return false;
    }
    return valuesEqual(exception, expectedType);
}

static void markValue(Value value) {
    if (!IS_OBJ(value)) {
        return;
    }
    markObject(AS_OBJ(value));
}

static void markValueArray(ValueArray* array) {
    for (int i = 0; i < array->count; i++) {
        markValue(array->values[i]);
    }
}

static void markTable(Table* table) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (IS_NIL(entry->key)) {
            continue;
        }
        markValue(entry->key);
        markValue(entry->value);
    }
}

static void markObject(Obj* object) {
    if (object == NULL || object->isMarked) {
        return;
    }

    object->isMarked = true;

    switch (object->type) {
        case OBJ_STRING:
        case OBJ_NATIVE:
        case OBJ_BYTES:
            return;
        case OBJ_FUNCTION: {
            ObjFunction* function = (ObjFunction*)object;
            markValueArray(&function->defaults);
            markValueArray(&function->paramNames);
            markValueArray(&function->localNames);
            markValueArray(&function->chunk.constants);
            if (function->name != NULL) {
                markObject((Obj*)function->name);
            }
            return;
        }
        case OBJ_CLOSURE: {
            ObjClosure* closure = (ObjClosure*)object;
            markObject((Obj*)closure->function);
            markObject((Obj*)closure->env);
            return;
        }
        case OBJ_ENVIRONMENT: {
            ObjEnvironment* env = (ObjEnvironment*)object;
            markObject((Obj*)env->parent);
            if (env->table != NULL) {
                markTable(env->table);
            }
            return;
        }
        case OBJ_SET:
            markTable(&((ObjSet*)object)->table);
            return;
        case OBJ_LIST:
            markValueArray(&((ObjList*)object)->items);
            return;
        case OBJ_DICT:
            markTable(&((ObjDict*)object)->table);
            return;
        case OBJ_TUPLE:
            markValueArray(&((ObjTuple*)object)->items);
            return;
        case OBJ_CLASS: {
            ObjClass* klass = (ObjClass*)object;
            markObject((Obj*)klass->name);
            markObject((Obj*)klass->superClass);
            markTable(&klass->methods);
            return;
        }
        case OBJ_INSTANCE: {
            ObjInstance* instance = (ObjInstance*)object;
            markObject((Obj*)instance->klass);
            markTable(&instance->fields);
            return;
        }
        case OBJ_BOUND_METHOD: {
            ObjBoundMethod* bound = (ObjBoundMethod*)object;
            markValue(bound->receiver);
            markValue(bound->method);
            return;
        }
        case OBJ_SLICE: {
            ObjSlice* slice = (ObjSlice*)object;
            markValue(slice->start);
            markValue(slice->stop);
            markValue(slice->step);
            return;
        }
        case OBJ_ITERATOR:
            markValue(((ObjIterator*)object)->iterable);
            return;
    }
}

static void markRoots(void) {
    for (Value* slot = vm.stack; slot < vm.stackTop; slot++) {
        markValue(*slot);
    }

    for (int i = 0; i < vm.frameCount; i++) {
        CallFrame* frame = &vm.frames[i];
        markObject((Obj*)frame->closure);
        markObject((Obj*)frame->function);
        markObject((Obj*)frame->env);
    }

    for (int i = 0; i < vm.handlerCount; i++) {
        markValue(vm.handlers[i].expectedType);
    }

    markObject((Obj*)vm.globalEnv);
    markObject((Obj*)vm.initString);
    markObject((Obj*)vm.moduleClass);
    markTable(&vm.modules);
    markTable(&vm.strings);
}

static void markRootsForCollection(void) {
    for (Value* slot = vm.stack; slot < vm.stackTop; slot++) {
        markValue(*slot);
    }

    for (int i = 0; i < vm.frameCount; i++) {
        CallFrame* frame = &vm.frames[i];
        markObject((Obj*)frame->closure);
        markObject((Obj*)frame->function);
        markObject((Obj*)frame->env);
    }

    for (int i = 0; i < vm.handlerCount; i++) {
        markValue(vm.handlers[i].expectedType);
    }

    markObject((Obj*)vm.globalEnv);
    markObject((Obj*)vm.initString);
    markObject((Obj*)vm.moduleClass);
    markTable(&vm.modules);
}

static void clearMarks(void) {
    for (Obj* object = vm.objects; object != NULL; object = object->next) {
        object->isMarked = false;
    }
}

static int countMarkedObjects(void) {
    int count = 0;
    for (Obj* object = vm.objects; object != NULL; object = object->next) {
        if (object->isMarked) {
            count++;
        }
    }
    return count;
}

static void updateNextGC(void) {
    vm.nextGC = vm.objectCount < 64 ? 64 : vm.objectCount * 2;
}

int gcMarkRootsAndCount(void) {
    clearMarks();
    markRoots();
    int count = countMarkedObjects();
    clearMarks();
    return count;
}

int gcCountReachableFromValue(Value value) {
    clearMarks();
    markValue(value);
    int count = countMarkedObjects();
    clearMarks();
    return count;
}

int gcCollect(void) {
    clearMarks();
    markRootsForCollection();
    tableRemoveWhite(&vm.strings);
    int freed = sweepUnmarkedObjects();
    updateNextGC();
    return freed;
}

int gcObjectCount(void) {
    return countObjects();
}

static bool raiseException(Value exception) {
    while (vm.handlerCount > 0) {
        Handler handler = vm.handlers[--vm.handlerCount];
        if (!exceptionMatches(exception, handler.expectedType)) {
            continue;
        }

        vm.frameCount = handler.frameCount;
        vm.stackTop = handler.stackTop;
        push(exception);

        vm.frames[vm.frameCount - 1].ip = handler.handlerIP;
        return true;
    }

    char buffer[1024];
    formatException(exception, buffer, sizeof(buffer));
    fprintf(stderr, "%s\n", buffer);

    for (int i = vm.frameCount - 1; i >= 0; i--) {
        CallFrame* frame = &vm.frames[i];
        ObjFunction* function = frame->function;
        size_t instruction = frame->ip - function->chunk.code - 1;
        int line = function->chunk.lines[instruction];
        fprintf(stderr, "[%s: line %d] in ", scanner.filename, line);
        if (function->name == NULL) {
            fprintf(stderr, "script\n");
        } else {
            fprintf(stderr, "%s()\n", function->name->chars);
        }
    }

    resetStack();
    return false;
}

static void resetStack() {
    vm.stackTop = vm.stack;
    vm.frameCount = 0;
    vm.handlerCount = 0;
}

static bool runtimeError(const char* format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    return raiseException(createExceptionValue("RuntimeError", buffer));
}

static void registerBuiltinHelpers(void) {
    const char* source =
        "def reversed(seq):\n"
        "    out = []\n"
        "    i = len(seq) - 1\n"
        "    while i >= 0:\n"
        "        out.append(seq[i])\n"
        "        i = i - 1\n"
        "    return out\n"
        "\n"
        "def zip(a, b):\n"
        "    out = []\n"
        "    limit = len(a)\n"
        "    if len(b) < limit:\n"
        "        limit = len(b)\n"
        "    i = 0\n"
        "    while i < limit:\n"
        "        out.append((a[i], b[i]))\n"
        "        i = i + 1\n"
        "    return out\n"
        "\n"
        "def map(func, seq):\n"
        "    out = []\n"
        "    for item in seq:\n"
        "        out.append(func(item))\n"
        "    return out\n"
        "\n"
        "def filter(func, seq):\n"
        "    out = []\n"
        "    for item in seq:\n"
        "        if func(item):\n"
        "            out.append(item)\n"
        "    return out\n";

    interpret(source, "<builtins>");
}

void initVM() {
    resetStack();
    vm.objects = NULL;
    vm.objectCount = 0;
    vm.nextGC = 64;
    vm.gcPauseDepth = 1;
    vm.gcEnabled = false;
    initTable(&vm.modules);
    initTable(&vm.strings);
    vm.globalEnv = newEnvironment(NULL);
    
    vm.initString = NULL; // Prevent GC from seeing garbage
    vm.initString = copyString("__init__", 8);
    vm.moduleClass = newClass(copyString("module", 6));

    registerBuiltins();
    defineGlobalClass("Exception", NULL);
    defineGlobalClass("RuntimeError", "Exception");
    defineGlobalClass("TypeError", "Exception");
    defineGlobalClass("ValueError", "Exception");
    defineGlobalClass("KeyError", "Exception");
    defineGlobalClass("IndexError", "Exception");
    defineGlobalClass("AttributeError", "Exception");
    defineGlobalClass("ZeroDivisionError", "Exception");
    registerBuiltinHelpers();
    vm.gcEnabled = true;
    vm.gcPauseDepth = 0;
    updateNextGC();
}

void freeVM() {
    freeObjects();
    freeTable(&vm.modules);
    freeTable(&vm.strings);
    vm.globalEnv = NULL;
    vm.initString = NULL;
    vm.moduleClass = NULL;
}

void push(Value value) {
    *vm.stackTop = value;
    vm.stackTop++;
}

Value pop() {
    vm.stackTop--;
    return *vm.stackTop;
}

static Value peek(int distance) {
    return vm.stackTop[-1 - distance];
}

static bool isFalsey(Value value) {
    if (IS_NIL(value)) return true;
    if (IS_BOOL(value)) return !AS_BOOL(value);
    if (IS_NUMBER(value)) return AS_NUMBER(value) == 0;
    if (IS_STRING(value)) return AS_STRING(value)->length == 0;
    if (IS_LIST(value)) return AS_LIST(value)->items.count == 0;
    if (IS_DICT(value)) return AS_DICT(value)->table.count == 0;
    if (IS_SET(value)) return AS_SET(value)->table.count == 0;
    if (IS_TUPLE(value)) return AS_TUPLE(value)->items.count == 0;
    return false;
}

static bool call(ObjClosure* closure, int argCount) {
    ObjFunction* function = closure->function;
    int normalizedArgCount = argCount;
    if (function->hasVarargs) {
        int fixedArity = function->arity;
        int extraCount = argCount > fixedArity ? argCount - fixedArity : 0;
        ObjTuple* varargs = newTuple();
        Value* argBase = vm.stackTop - argCount;
        for (int i = 0; i < extraCount; i++) {
            writeValueArray(&varargs->items, argBase[fixedArity + i]);
        }
        vm.stackTop = argBase + fixedArity;
        push(OBJ_VAL(varargs));
        normalizedArgCount = fixedArity + 1;
        argCount = fixedArity;
    }

    if (argCount < function->arity) {
        int defaultsRequired = function->arity - argCount;
        if (defaultsRequired <= function->defaultsCount) {
            // Push defaults
            for (int i = function->defaultsCount - defaultsRequired; i < function->defaultsCount; i++) {
                push(function->defaults.values[i]);
            }
            argCount = function->arity;
            normalizedArgCount = function->arity + (function->hasVarargs ? 1 : 0);
        } else {
            if (argCount < function->paramNames.count) {
                runtimeError("Missing required argument '%s'.",
                             AS_CSTRING(function->paramNames.values[argCount]));
            } else {
                runtimeError("Expected %d arguments but got %d.", function->arity, argCount);
            }
            return false;
        }
    } else if (argCount > function->arity) {
        runtimeError("Expected %d arguments but got %d.", function->arity, argCount);
        return false;
    }

    if (vm.frameCount == FRAMES_MAX) {
        runtimeError("Stack overflow.");
        return false;
    }

    CallFrame* frame = &vm.frames[vm.frameCount++];
    frame->closure = closure;
    frame->function = function;
    frame->ip = function->chunk.code;
    frame->slots = vm.stackTop - normalizedArgCount - 1;
    if (function->name == NULL) {
        frame->env = closure->env != NULL ? closure->env : vm.globalEnv;
    } else {
        frame->env = newEnvironment(closure->env != NULL ? closure->env : vm.globalEnv);
    }

    // Reserve space for locals
    for (int i = normalizedArgCount + 1; i < function->maxSlots; i++) {
        push(NIL_VAL);
    }

    for (int i = 0; i < function->localNames.count && i < function->maxSlots; i++) {
        syncFrameLocalToEnv(frame, i);
    }
    return true;
}

static bool callValueKeyword(Value callee, int posCount, int kwCount);

static bool expandIterableArgs(Value iterable, Value* outArgs, int* outCount) {
    if (IS_LIST(iterable)) {
        ObjList* list = AS_LIST(iterable);
        for (int i = 0; i < list->items.count; i++) {
            outArgs[(*outCount)++] = list->items.values[i];
        }
        return true;
    }
    if (IS_TUPLE(iterable)) {
        ObjTuple* tuple = AS_TUPLE(iterable);
        for (int i = 0; i < tuple->items.count; i++) {
            outArgs[(*outCount)++] = tuple->items.values[i];
        }
        return true;
    }
    if (IS_STRING(iterable)) {
        ObjString* string = AS_STRING(iterable);
        for (int i = 0; i < string->length; i++) {
            outArgs[(*outCount)++] = OBJ_VAL(copyString(string->chars + i, 1));
        }
        return true;
    }
    if (IS_SET(iterable)) {
        ObjSet* set = AS_SET(iterable);
        for (int i = 0; i < set->table.capacity; i++) {
            Entry* entry = &set->table.entries[i];
            if (!IS_NIL(entry->key)) {
                outArgs[(*outCount)++] = entry->key;
            }
        }
        return true;
    }
    if (IS_DICT(iterable)) {
        ObjDict* dict = AS_DICT(iterable);
        for (int i = 0; i < dict->table.capacity; i++) {
            Entry* entry = &dict->table.entries[i];
            if (!IS_NIL(entry->key)) {
                outArgs[(*outCount)++] = entry->key;
            }
        }
        return true;
    }
    runtimeError("Object is not iterable.");
    return false;
}

static bool collectExpandedArgs(Value* sourceArgs, int sourceCount, ObjTuple* starIndexes, Value* outArgs, int* outCount) {
    int starCursor = 0;
    for (int i = 0; i < sourceCount; i++) {
        bool isStar = false;
        if (starCursor < starIndexes->items.count) {
            Value marker = starIndexes->items.values[starCursor];
            if (IS_NUMBER(marker) && (int)AS_NUMBER(marker) == i) {
                isStar = true;
                starCursor++;
            }
        }

        if (isStar) {
            if (!expandIterableArgs(sourceArgs[i], outArgs, outCount)) {
                return false;
            }
        } else {
            outArgs[(*outCount)++] = sourceArgs[i];
        }
    }
    return true;
}

static bool callValue(Value callee, int argCount) {
    if (IS_OBJ(callee)) {
        switch (OBJ_TYPE(callee)) {
            case OBJ_BOUND_METHOD: {
                ObjBoundMethod* bound = AS_BOUND_METHOD(callee);
                // Shift arguments up by 1 to make room for receiver
                for (int i = 0; i < argCount; i++) {
                    vm.stackTop[-i] = vm.stackTop[-i - 1];
                }
                vm.stackTop[-argCount] = bound->receiver;
                vm.stackTop++;
                return callValue(bound->method, argCount + 1);
            }
            case OBJ_CLASS: {
                ObjClass* klass = AS_CLASS(callee);
                ObjInstance* instance = newInstance(klass);
                vm.stackTop[-argCount - 1] = OBJ_VAL(instance);
                Value initializer;
                if (tableGet(&klass->methods, OBJ_VAL(vm.initString), &initializer)) {
                    for (int i = 0; i <= argCount; i++) {
                        vm.stackTop[-i] = vm.stackTop[-i - 1];
                    }
                    vm.stackTop[-argCount - 1] = initializer;
                    vm.stackTop++;
                    return callValue(initializer, argCount + 1);
                } else if (isExceptionClass(klass)) {
                    if (argCount > 1) {
                        runtimeError("Expected 0 or 1 arguments but got %d.", argCount);
                        return false;
                    }

                    char buffer[256];
                    const char* message = NULL;
                    if (argCount == 1) {
                        message = exceptionMessageArg(peek(0), buffer, sizeof(buffer));
                        if (message == NULL) {
                            runtimeError("Exception message must be a string or simple value.");
                            return false;
                        }
                    }

                    vm.stackTop[-argCount - 1] = OBJ_VAL(newExceptionInstance(klass, message));
                    vm.stackTop -= argCount;
                    return true;
                } else if (argCount != 0) {
                    runtimeError("Expected 0 arguments but got %d.", argCount);
                    return false;
                }
                return true;
            }
            case OBJ_NATIVE: {
                NativeFn native = AS_NATIVE(callee);
                Value result = native(argCount, vm.stackTop - argCount);
                vm.stackTop -= argCount + 1;
                push(result);
                return true;
            }
            case OBJ_CLOSURE:
                return call(AS_CLOSURE(callee), argCount);
            case OBJ_FUNCTION:
                return call(newClosure(AS_FUNCTION(callee), vm.globalEnv), argCount);
            default:
                break; // Non-callable object type.
        }
    }
    runtimeError("Can only call functions and classes.");
    return false;
}

static bool callValueKeyword(Value callee, int posCount, int kwCount) {
    if (!IS_OBJ(callee)) {
        runtimeError("Can only call functions and classes.");
        return false;
    }

    ObjClosure* closure = NULL;
    ObjFunction* fn = NULL;
    Value finalCallee = callee;

    if (OBJ_TYPE(callee) == OBJ_BOUND_METHOD) {
        ObjBoundMethod* bound = AS_BOUND_METHOD(callee);
        for (int i = 0; i < posCount + 2 * kwCount; i++) {
            vm.stackTop[-i] = vm.stackTop[-i - 1];
        }
        vm.stackTop[-posCount - 2 * kwCount] = bound->receiver;
        vm.stackTop++;
        finalCallee = bound->method;
        if (IS_CLOSURE(bound->method)) {
            closure = AS_CLOSURE(bound->method);
            fn = closure->function;
        } else if (IS_FUNCTION(bound->method)) {
            fn = AS_FUNCTION(bound->method);
        }
        posCount++;
    } else if (OBJ_TYPE(callee) == OBJ_CLOSURE) {
        closure = AS_CLOSURE(callee);
        fn = closure->function;
    } else if (OBJ_TYPE(callee) == OBJ_FUNCTION) {
        fn = AS_FUNCTION(callee);
    } else if (OBJ_TYPE(callee) == OBJ_CLASS) {
        ObjClass* klass = AS_CLASS(callee);
        ObjInstance* instance = newInstance(klass);
        vm.stackTop[-posCount - 2 * kwCount - 1] = OBJ_VAL(instance);
        Value initializer;
        if (tableGet(&klass->methods, OBJ_VAL(vm.initString), &initializer)) {
            for (int i = 0; i < posCount + 2 * kwCount + 1; i++) {
                vm.stackTop[-i] = vm.stackTop[-i - 1];
            }
            vm.stackTop[-posCount - 2 * kwCount - 1] = initializer;
            vm.stackTop++;
            finalCallee = initializer;
            if (IS_CLOSURE(initializer)) {
                closure = AS_CLOSURE(initializer);
                fn = closure->function;
            } else if (IS_FUNCTION(initializer)) {
                fn = AS_FUNCTION(initializer);
            }
            posCount++;
        } else {
            if (isExceptionClass(klass)) {
                if (kwCount > 0) {
                    runtimeError("Builtin exception constructors do not accept keyword arguments.");
                    return false;
                }
                if (posCount > 1) {
                    runtimeError("Expected 0 or 1 arguments but got %d.", posCount);
                    return false;
                }

                char buffer[256];
                const char* message = NULL;
                if (posCount == 1) {
                    message = exceptionMessageArg(peek(0), buffer, sizeof(buffer));
                    if (message == NULL) {
                        runtimeError("Exception message must be a string or simple value.");
                        return false;
                    }
                }

                vm.stackTop[-posCount - 1] = OBJ_VAL(newExceptionInstance(klass, message));
                vm.stackTop -= posCount;
                return true;
            }
            if (posCount > 0 || kwCount > 0) {
                runtimeError("Expected 0 arguments but got some.");
                return false;
            }
            return true;
        }
    }

    if (fn == NULL) {
        runtimeError("Can only call functions and classes.");
        return false;
    }

    Value kwNames[255];
    Value kwValues[255];
    for (int i = 0; i < kwCount; i++) kwNames[kwCount - 1 - i] = pop();
    for (int i = 0; i < kwCount; i++) kwValues[kwCount - 1 - i] = pop();

    Value posArgs[255];
    for (int i = 0; i < posCount; i++) posArgs[posCount - 1 - i] = pop();

    pop(); // callee

    Value finalArgs[256];
    bool provided[256] = {false};
    int extraPosCount = 0;

    for (int i = 0; i < posCount; i++) {
        if (i >= fn->arity) {
            if (fn->hasVarargs) {
                extraPosCount = posCount - fn->arity;
                break;
            }
            runtimeError("Too many positional arguments.");
            return false;
        }
        finalArgs[i] = posArgs[i];
        provided[i] = true;
    }

    for (int i = 0; i < kwCount; i++) {
        if (!IS_STRING(kwNames[i])) {
            runtimeError("Keyword must be a string.");
            return false;
        }
        ObjString* kwName = AS_STRING(kwNames[i]);
        int index = -1;
        for (int j = 0; j < fn->paramNames.count; j++) {
            if (AS_STRING(fn->paramNames.values[j]) == kwName) {
                index = j;
                break;
            }
        }
        if (index == -1) {
            runtimeError("Unexpected keyword argument '%s'.", kwName->chars);
            return false;
        }
        if (provided[index]) {
            runtimeError("Multiple values for argument '%s'.", kwName->chars);
            return false;
        }
        finalArgs[index] = kwValues[i];
        provided[index] = true;
    }

    for (int i = 0; i < fn->arity; i++) {
        if (!provided[i]) {
            int defaultIdx = i - (fn->arity - fn->defaultsCount);
            if (defaultIdx >= 0) {
                finalArgs[i] = fn->defaults.values[defaultIdx];
                provided[i] = true;
            } else {
                if (i < fn->paramNames.count) {
                    runtimeError("Missing required argument '%s'.", AS_CSTRING(fn->paramNames.values[i]));
                } else {
                    runtimeError("Missing required argument at index %d.", i);
                }
                return false;
            }
        }
    }

    push(finalCallee);
    for (int i = 0; i < fn->arity; i++) push(finalArgs[i]);
    if (fn->hasVarargs && extraPosCount > 0) {
        for (int i = 0; i < extraPosCount; i++) {
            push(posArgs[fn->arity + i]);
        }
        if (closure != NULL) {
            return call(closure, fn->arity + extraPosCount);
        }
        return call(newClosure(fn, vm.globalEnv), fn->arity + extraPosCount);
    }
    if (closure != NULL) {
        return call(closure, fn->arity);
    }
    return call(newClosure(fn, vm.globalEnv), fn->arity);
}

static bool invokeFromClassKeyword(ObjClass* klass, ObjString* name, int posCount, int kwCount) {
    Value method;
    if (!tableGet(&klass->methods, OBJ_VAL(name), &method)) {
        runtimeError("Undefined property '%s'.", name->chars);
        return false;
    }
    for (int i = 0; i < posCount + 2 * kwCount + 1; i++) {
        vm.stackTop[-i] = vm.stackTop[-i - 1];
    }
    vm.stackTop[-posCount - 2 * kwCount - 1] = method;
    vm.stackTop++;
    return callValueKeyword(method, posCount + 1, kwCount);
}

static bool invokeKeyword(ObjString* name, int posCount, int kwCount) {
    Value receiver = peek(posCount + 2 * kwCount);
    if (!IS_INSTANCE(receiver)) {
        // Built-ins don't fully support all keyword args yet, but we'll try
        // passing total effective argCount (pos + 2*kw)
        return invokeBuiltinMethod(receiver, name, posCount + 2 * kwCount);
    }
    ObjInstance* instance = AS_INSTANCE(receiver);

    Value method;
    if (tableGet(&instance->fields, OBJ_VAL(name), &method)) {
        vm.stackTop[-posCount - 2 * kwCount - 1] = method;
        return callValueKeyword(method, posCount, kwCount);
    }

    return invokeFromClassKeyword(instance->klass, name, posCount, kwCount);
}

static bool invokeFromClass(ObjClass* klass, ObjString* name, int argCount) {
    Value method;
    if (!tableGet(&klass->methods, OBJ_VAL(name), &method)) {
        raiseException(createExceptionValue("AttributeError", "Undefined property."));
        return false;
    }
    // Stack is [receiver, arg1, arg2...]
    // Shift to make room for method -> [method, receiver, arg1, arg2...]
    for (int i = 0; i <= argCount; i++) {
        vm.stackTop[-i] = vm.stackTop[-i - 1];
    }
    vm.stackTop[-argCount - 1] = method;
    vm.stackTop++;
    return callValue(method, argCount + 1);
}

static ObjList* tableKeysList(Table* table) {
    ObjList* result = newList();
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (!IS_NIL(entry->key)) {
            writeValueArray(&result->items, entry->key);
        }
    }
    return result;
}

static ObjList* tableValuesList(Table* table) {
    ObjList* result = newList();
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (!IS_NIL(entry->key)) {
            writeValueArray(&result->items, entry->value);
        }
    }
    return result;
}

static ObjList* tableItemsList(Table* table) {
    ObjList* result = newList();
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (!IS_NIL(entry->key)) {
            ObjTuple* item = newTuple();
            writeValueArray(&item->items, entry->key);
            writeValueArray(&item->items, entry->value);
            writeValueArray(&result->items, OBJ_VAL(item));
        }
    }
    return result;
}

static Value tablePopFirst(Table* table, bool* found) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (!IS_NIL(entry->key)) {
            Value key = entry->key;
            Value value = entry->value;
            tableDelete(table, key);
            *found = true;
            ObjTuple* pair = newTuple();
            writeValueArray(&pair->items, key);
            writeValueArray(&pair->items, value);
            return OBJ_VAL(pair);
        }
    }
    *found = false;
    return NIL_VAL;
}

static bool listAssignSlice(ObjList* list, ObjSlice* slice, Value value) {
    if (!IS_LIST(value)) {
        runtimeError("Can only assign a list to a list slice.");
        return false;
    }

    int step = IS_NIL(slice->step) ? 1 : (int)AS_NUMBER(slice->step);
    if (step != 1) {
        runtimeError("Slice assignment only supports step=1.");
        return false;
    }

    int start = IS_NIL(slice->start) ? 0 : (int)AS_NUMBER(slice->start);
    int stop = IS_NIL(slice->stop) ? list->items.count : (int)AS_NUMBER(slice->stop);
    if (start < 0) start += list->items.count;
    if (stop < 0) stop += list->items.count;
    if (start < 0) start = 0;
    if (start > list->items.count) start = list->items.count;
    if (stop < start) stop = start;
    if (stop > list->items.count) stop = list->items.count;

    ObjList* src = AS_LIST(value);
    int removed = stop - start;
    int added = src->items.count;
    int newCount = list->items.count - removed + added;

    while (list->items.capacity < newCount) {
        writeValueArray(&list->items, NIL_VAL);
        list->items.count--;
    }

    if (added > removed) {
        for (int i = list->items.count - 1; i >= stop; i--) {
            list->items.values[i + (added - removed)] = list->items.values[i];
        }
    } else if (added < removed) {
        for (int i = stop; i < list->items.count; i++) {
            list->items.values[i - (removed - added)] = list->items.values[i];
        }
    }

    for (int i = 0; i < added; i++) {
        list->items.values[start + i] = src->items.values[i];
    }
    list->items.count = newCount;
    return true;
}

static void reverseListItems(ObjList* list) {
    for (int i = 0, j = list->items.count - 1; i < j; i++, j--) {
        Value tmp = list->items.values[i];
        list->items.values[i] = list->items.values[j];
        list->items.values[j] = tmp;
    }
}

static ObjString* makeStringTransform(ObjString* input, int (*transform)(int)) {
    char* chars = (char*)malloc((size_t)input->length + 1);
    for (int i = 0; i < input->length; i++) {
        chars[i] = (char)transform((unsigned char)input->chars[i]);
    }
    chars[input->length] = '\0';
    return takeString(chars, input->length);
}

static bool isTruthyValue(Value value) {
    return !isFalsey(value);
}

static bool isAllChars(ObjString* string, int (*predicate)(int), bool requireNonEmpty) {
    if (string->length == 0) return !requireNonEmpty;
    for (int i = 0; i < string->length; i++) {
        if (!predicate((unsigned char)string->chars[i])) return false;
    }
    return true;
}

static bool invokeBuiltinMethod(Value receiver, ObjString* name, int argCount) {
    if (IS_LIST(receiver)) {
        ObjList* list = AS_LIST(receiver);
        if (strcmp(name->chars, "append") == 0) {
            if (argCount != 1) { runtimeError("append() takes 1 argument"); return false; }
            writeValueArray(&list->items, peek(0));
            pop(); // arg
            pop(); // receiver
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "pop") == 0) {
            if (argCount > 1) { runtimeError("pop() takes at most 1 argument"); return false; }
            if (list->items.count == 0) {
                raiseException(createExceptionValue("IndexError", "pop from empty list"));
                return false;
            }
            int index = list->items.count - 1;
            if (argCount == 1) {
                if (!IS_NUMBER(peek(0))) { runtimeError("pop() index must be a number"); return false; }
                index = (int)AS_NUMBER(peek(0));
                if (index < 0) index += list->items.count;
                if (index < 0 || index >= list->items.count) {
                    raiseException(createExceptionValue("IndexError", "pop index out of range"));
                    return false;
                }
                pop(); // arg
            }
            Value val = list->items.values[index];
            for (int i = index; i < list->items.count - 1; i++) {
                list->items.values[i] = list->items.values[i + 1];
            }
            list->items.count--;
            pop(); // receiver
            push(val);
            return true;
        } else if (strcmp(name->chars, "sort") == 0) {
            Value key = NIL_VAL;
            bool reverse = false;
            bool sawKey = false;
            bool sawReverse = false;
            if (argCount == 1) {
                key = peek(0);
                if (IS_STRING(peek(0)) && strcmp(AS_STRING(peek(0))->chars, "reverse") == 0) {
                    runtimeError("sort() missing keyword value");
                    return false;
                }
            } else if (argCount == 2) {
                if (IS_STRING(peek(0))) {
                    ObjString* kw = AS_STRING(peek(0));
                    if (strcmp(kw->chars, "key") == 0) {
                        key = peek(1);
                        sawKey = true;
                    } else if (strcmp(kw->chars, "reverse") == 0) {
                        reverse = isTruthyValue(peek(1));
                        sawReverse = true;
                    } else {
                        runtimeError("sort() got unexpected keyword argument");
                        return false;
                    }
                } else {
                    key = peek(1);
                    reverse = isTruthyValue(peek(0));
                    sawKey = true;
                    sawReverse = true;
                }
            } else if (argCount == 4) {
                Value kwName1 = peek(1);
                Value kwValue1 = peek(3);
                Value kwName2 = peek(0);
                Value kwValue2 = peek(2);
                if (!IS_STRING(kwName1) || !IS_STRING(kwName2)) {
                    runtimeError("sort() keyword names must be strings");
                    return false;
                }
                ObjString* name1 = AS_STRING(kwName1);
                ObjString* name2 = AS_STRING(kwName2);
                if (strcmp(name1->chars, "key") == 0) {
                    key = kwValue1;
                    sawKey = true;
                } else if (strcmp(name1->chars, "reverse") == 0) {
                    reverse = isTruthyValue(kwValue1);
                    sawReverse = true;
                }
                else {
                    runtimeError("sort() got unexpected keyword argument");
                    return false;
                }
                if (strcmp(name2->chars, "key") == 0) {
                    if (sawKey) {
                        runtimeError("sort() got multiple values for keyword argument 'key'");
                        return false;
                    }
                    key = kwValue2;
                    sawKey = true;
                } else if (strcmp(name2->chars, "reverse") == 0) {
                    if (sawReverse) {
                        runtimeError("sort() got multiple values for keyword argument 'reverse'");
                        return false;
                    }
                    reverse = isTruthyValue(kwValue2);
                    sawReverse = true;
                }
                else {
                    runtimeError("sort() got unexpected keyword argument");
                    return false;
                }
            } else if (argCount > 2) {
                runtimeError("sort() takes at most 2 arguments");
                return false;
            }
            
            if (IS_NIL(key)) {
                qsort(list->items.values, list->items.count, sizeof(Value), compareValues);
            } else {
                // Sort with key function
                // This is tricky with qsort. For now, we'll do a simple bubble sort or similar
                // to avoid the qsort context issue, OR we use a global.
                // Let's use a simple selection sort for now to guarantee correctness with the VM.
                for (int i = 0; i < list->items.count - 1; i++) {
                    for (int j = i + 1; j < list->items.count; j++) {
                        // val1 = key(list[i])
                        push(key);
                        push(list->items.values[i]);
                        if (!callValue(key, 1)) return false;
                        run();
                        Value val1 = pop();
                        
                        // val2 = key(list[j])
                        push(key);
                        push(list->items.values[j]);
                        if (!callValue(key, 1)) return false;
                        run();
                        Value val2 = pop();
                        
                        if (compareValues(&val1, &val2) > 0) {
                            Value temp = list->items.values[i];
                            list->items.values[i] = list->items.values[j];
                            list->items.values[j] = temp;
                        }
                    }
                }
            }
            if (reverse) reverseListItems(list);
            for (int i = 0; i < argCount; i++) pop();
            pop(); // receiver
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "extend") == 0) {
            if (argCount != 1) { runtimeError("extend() takes 1 argument"); return false; }
            if (!IS_LIST(peek(0))) { runtimeError("extend() argument must be a list"); return false; }
            ObjList* other = AS_LIST(pop());
            for (int i = 0; i < other->items.count; i++) {
                writeValueArray(&list->items, other->items.values[i]);
            }
            pop(); // receiver
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "remove") == 0) {
            if (argCount != 1) { runtimeError("remove() takes 1 argument"); return false; }
            Value target = peek(0);
            int index = -1;
            for (int i = 0; i < list->items.count; i++) {
                if (valuesEqual(list->items.values[i], target)) {
                    index = i;
                    break;
                }
            }
            if (index == -1) {
                raiseException(createExceptionValue("ValueError", "list.remove(x): x not in list"));
                return false;
            }
            for (int i = index; i < list->items.count - 1; i++) {
                list->items.values[i] = list->items.values[i + 1];
            }
            list->items.count--;
            pop(); // arg
            pop(); // receiver
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "clear") == 0) {
            if (argCount != 0) { runtimeError("clear() takes 0 arguments"); return false; }
            list->items.count = 0;
            pop();
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "copy") == 0) {
            if (argCount != 0) { runtimeError("copy() takes 0 arguments"); return false; }
            ObjList* result = newList();
            for (int i = 0; i < list->items.count; i++) {
                writeValueArray(&result->items, list->items.values[i]);
            }
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "count") == 0) {
            if (argCount != 1) { runtimeError("count() takes 1 argument"); return false; }
            Value target = pop();
            int count = 0;
            for (int i = 0; i < list->items.count; i++) {
                if (valuesEqual(list->items.values[i], target)) count++;
            }
            pop();
            push(NUMBER_VAL((double)count));
            return true;
        } else if (strcmp(name->chars, "index") == 0) {
            if (argCount != 1) { runtimeError("index() takes 1 argument"); return false; }
            Value target = pop();
            int index = -1;
            for (int i = 0; i < list->items.count; i++) {
                if (valuesEqual(list->items.values[i], target)) {
                    index = i;
                    break;
                }
            }
            pop();
            if (index == -1) {
                raiseException(createExceptionValue("ValueError", "list.index(x): x not in list"));
                return false;
            }
            push(NUMBER_VAL((double)index));
            return true;
        } else if (strcmp(name->chars, "insert") == 0) {
            if (argCount != 2) { runtimeError("insert() takes 2 arguments"); return false; }
            Value item = pop();
            Value indexValue = pop();
            if (!IS_NUMBER(indexValue)) { runtimeError("insert() index must be a number"); return false; }
            int index = (int)AS_NUMBER(indexValue);
            if (index < 0) index = 0;
            if (index > list->items.count) index = list->items.count;
            writeValueArray(&list->items, NIL_VAL);
            for (int i = list->items.count - 1; i > index; i--) {
                list->items.values[i] = list->items.values[i - 1];
            }
            list->items.values[index] = item;
            pop();
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "reverse") == 0) {
            if (argCount != 0) { runtimeError("reverse() takes 0 arguments"); return false; }
            reverseListItems(list);
            pop();
            push(NIL_VAL);
            return true;
        }
    } else if (IS_STRING(receiver)) {
        ObjString* string = AS_STRING(receiver);
        if (strcmp(name->chars, "lower") == 0) {
            if (argCount != 0) { runtimeError("lower() takes 0 arguments"); return false; }
            pop();
            push(OBJ_VAL(makeStringTransform(string, tolower)));
            return true;
        } else if (strcmp(name->chars, "upper") == 0) {
            if (argCount != 0) { runtimeError("upper() takes 0 arguments"); return false; }
            pop();
            push(OBJ_VAL(makeStringTransform(string, toupper)));
            return true;
        } else if (strcmp(name->chars, "encode") == 0) {
            if (argCount != 0) { runtimeError("encode() takes 0 arguments"); return false; }
            ObjBytes* result = newBytes(string->length, (const uint8_t*)string->chars);
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "capitalize") == 0) {
            if (argCount != 0) { runtimeError("capitalize() takes 0 arguments"); return false; }
            char* chars = (char*)malloc((size_t)string->length + 1);
            for (int i = 0; i < string->length; i++) {
                unsigned char c = (unsigned char)string->chars[i];
                chars[i] = (char)(i == 0 ? toupper(c) : tolower(c));
            }
            chars[string->length] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, string->length)));
            return true;
        } else if (strcmp(name->chars, "swapcase") == 0) {
            if (argCount != 0) { runtimeError("swapcase() takes 0 arguments"); return false; }
            char* chars = (char*)malloc((size_t)string->length + 1);
            for (int i = 0; i < string->length; i++) {
                unsigned char c = (unsigned char)string->chars[i];
                if (islower(c)) chars[i] = (char)toupper(c);
                else if (isupper(c)) chars[i] = (char)tolower(c);
                else chars[i] = (char)c;
            }
            chars[string->length] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, string->length)));
            return true;
        } else if (strcmp(name->chars, "title") == 0) {
            if (argCount != 0) { runtimeError("title() takes 0 arguments"); return false; }
            char* chars = (char*)malloc((size_t)string->length + 1);
            bool newWord = true;
            for (int i = 0; i < string->length; i++) {
                unsigned char c = (unsigned char)string->chars[i];
                if (isalpha(c)) {
                    chars[i] = (char)(newWord ? toupper(c) : tolower(c));
                    newWord = false;
                } else {
                    chars[i] = (char)c;
                    newWord = !isdigit(c);
                }
            }
            chars[string->length] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, string->length)));
            return true;
        } else if (strcmp(name->chars, "islower") == 0) {
            if (argCount != 0) { runtimeError("islower() takes 0 arguments"); return false; }
            bool sawAlpha = false;
            bool ok = true;
            for (int i = 0; i < string->length; i++) {
                unsigned char c = (unsigned char)string->chars[i];
                if (isalpha(c)) {
                    sawAlpha = true;
                    if (!islower(c)) {
                        ok = false;
                        break;
                    }
                }
            }
            pop();
            push(BOOL_VAL(sawAlpha && ok));
            return true;
        } else if (strcmp(name->chars, "isupper") == 0) {
            if (argCount != 0) { runtimeError("isupper() takes 0 arguments"); return false; }
            bool sawAlpha = false;
            bool ok = true;
            for (int i = 0; i < string->length; i++) {
                unsigned char c = (unsigned char)string->chars[i];
                if (isalpha(c)) {
                    sawAlpha = true;
                    if (!isupper(c)) {
                        ok = false;
                        break;
                    }
                }
            }
            pop();
            push(BOOL_VAL(sawAlpha && ok));
            return true;
        } else if (strcmp(name->chars, "isdigit") == 0) {
            if (argCount != 0) { runtimeError("isdigit() takes 0 arguments"); return false; }
            pop();
            push(BOOL_VAL(isAllChars(string, isdigit, true)));
            return true;
        } else if (strcmp(name->chars, "isalpha") == 0) {
            if (argCount != 0) { runtimeError("isalpha() takes 0 arguments"); return false; }
            pop();
            push(BOOL_VAL(isAllChars(string, isalpha, true)));
            return true;
        } else if (strcmp(name->chars, "strip") == 0) {
            if (argCount != 0) { runtimeError("strip() takes 0 arguments"); return false; }
            int start = 0;
            int end = string->length;
            while (start < end && isspace((unsigned char)string->chars[start])) start++;
            while (end > start && isspace((unsigned char)string->chars[end - 1])) end--;
            pop();
            push(OBJ_VAL(copyString(string->chars + start, end - start)));
            return true;
        } else if (strcmp(name->chars, "lstrip") == 0) {
            if (argCount != 0) { runtimeError("lstrip() takes 0 arguments"); return false; }
            int start = 0;
            while (start < string->length && isspace((unsigned char)string->chars[start])) start++;
            pop();
            push(OBJ_VAL(copyString(string->chars + start, string->length - start)));
            return true;
        } else if (strcmp(name->chars, "rstrip") == 0) {
            if (argCount != 0) { runtimeError("rstrip() takes 0 arguments"); return false; }
            int end = string->length;
            while (end > 0 && isspace((unsigned char)string->chars[end - 1])) end--;
            pop();
            push(OBJ_VAL(copyString(string->chars, end)));
            return true;
        } else if (strcmp(name->chars, "center") == 0 ||
                   strcmp(name->chars, "ljust") == 0 ||
                   strcmp(name->chars, "rjust") == 0) {
            if (argCount != 1 || !IS_NUMBER(peek(0))) {
                runtimeError("%s() takes 1 numeric argument", name->chars);
                return false;
            }
            int width = (int)AS_NUMBER(pop());
            if (width <= string->length) {
                pop();
                push(OBJ_VAL(copyString(string->chars, string->length)));
                return true;
            }
            int pad = width - string->length;
            int left = 0;
            int right = 0;
            if (strcmp(name->chars, "center") == 0) {
                left = pad / 2;
                right = pad - left;
            } else if (strcmp(name->chars, "ljust") == 0) {
                right = pad;
            } else {
                left = pad;
            }
            char* chars = (char*)malloc((size_t)width + 1);
            memset(chars, ' ', (size_t)left);
            memcpy(chars + left, string->chars, string->length);
            memset(chars + left + string->length, ' ', (size_t)right);
            chars[width] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, width)));
            return true;
        } else if (strcmp(name->chars, "startswith") == 0 || strcmp(name->chars, "endswith") == 0) {
            if (argCount != 1 || !IS_STRING(peek(0))) { runtimeError("%s() takes 1 string argument", name->chars); return false; }
            ObjString* other = AS_STRING(pop());
            bool match = false;
            if (strcmp(name->chars, "startswith") == 0) {
                match = string->length >= other->length &&
                        memcmp(string->chars, other->chars, other->length) == 0;
            } else {
                match = string->length >= other->length &&
                        memcmp(string->chars + string->length - other->length, other->chars, other->length) == 0;
            }
            pop();
            push(BOOL_VAL(match));
            return true;
        } else if (strcmp(name->chars, "find") == 0) {
            if (argCount != 1 || !IS_STRING(peek(0))) { runtimeError("find() takes 1 string argument"); return false; }
            ObjString* needle = AS_STRING(pop());
            int found = -1;
            if (needle->length == 0) {
                found = 0;
            } else {
                for (int i = 0; i <= string->length - needle->length; i++) {
                    if (memcmp(string->chars + i, needle->chars, needle->length) == 0) {
                        found = i;
                        break;
                    }
                }
            }
            pop();
            push(NUMBER_VAL((double)found));
            return true;
        } else if (strcmp(name->chars, "count") == 0) {
            if (argCount != 1 || !IS_STRING(peek(0))) { runtimeError("count() takes 1 string argument"); return false; }
            ObjString* needle = AS_STRING(pop());
            int count = 0;
            if (needle->length == 0) {
                count = string->length + 1;
            } else {
                for (int i = 0; i <= string->length - needle->length;) {
                    if (memcmp(string->chars + i, needle->chars, needle->length) == 0) {
                        count++;
                        i += needle->length;
                    } else {
                        i++;
                    }
                }
            }
            pop();
            push(NUMBER_VAL((double)count));
            return true;
        } else if (strcmp(name->chars, "index") == 0) {
            if (argCount != 1 || !IS_STRING(peek(0))) { runtimeError("index() takes 1 string argument"); return false; }
            ObjString* needle = AS_STRING(pop());
            int found = -1;
            if (needle->length == 0) {
                found = 0;
            } else {
                for (int i = 0; i <= string->length - needle->length; i++) {
                    if (memcmp(string->chars + i, needle->chars, needle->length) == 0) {
                        found = i;
                        break;
                    }
                }
            }
            pop();
            if (found == -1) {
                raiseException(createExceptionValue("ValueError", "substring not found"));
                return false;
            }
            push(NUMBER_VAL((double)found));
            return true;
        } else if (strcmp(name->chars, "replace") == 0) {
            if (argCount != 2 || !IS_STRING(peek(1)) || !IS_STRING(peek(0))) { runtimeError("replace() takes 2 string arguments"); return false; }
            ObjString* replacement = AS_STRING(pop());
            ObjString* old = AS_STRING(pop());
            if (old->length == 0) {
                raiseException(createExceptionValue("ValueError", "replace() old value cannot be empty"));
                return false;
            }
            int count = 0;
            for (int i = 0; i <= string->length - old->length;) {
                if (memcmp(string->chars + i, old->chars, old->length) == 0) {
                    count++;
                    i += old->length;
                } else {
                    i++;
                }
            }
            int newLength = string->length + count * (replacement->length - old->length);
            char* chars = (char*)malloc((size_t)newLength + 1);
            int out = 0;
            for (int i = 0; i < string->length;) {
                if (i <= string->length - old->length &&
                    memcmp(string->chars + i, old->chars, old->length) == 0) {
                    memcpy(chars + out, replacement->chars, replacement->length);
                    out += replacement->length;
                    i += old->length;
                } else {
                    chars[out++] = string->chars[i++];
                }
            }
            chars[out] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, out)));
            return true;
        } else if (strcmp(name->chars, "split") == 0) {
            ObjString* sep = NULL;
            if (argCount > 1) { runtimeError("split() takes at most 1 argument"); return false; }
            if (argCount == 1) {
                if (!IS_STRING(peek(0))) { runtimeError("split() separator must be a string"); return false; }
                sep = AS_STRING(pop());
            }
            ObjList* result = newList();
            if (sep == NULL) {
                int i = 0;
                while (i < string->length) {
                    while (i < string->length && isspace((unsigned char)string->chars[i])) i++;
                    int start = i;
                    while (i < string->length && !isspace((unsigned char)string->chars[i])) i++;
                    if (i > start) writeValueArray(&result->items, OBJ_VAL(copyString(string->chars + start, i - start)));
                }
            } else if (sep->length == 0) {
                raiseException(createExceptionValue("ValueError", "empty separator"));
                return false;
            } else {
                int start = 0;
                for (int i = 0; i <= string->length - sep->length;) {
                    if (memcmp(string->chars + i, sep->chars, sep->length) == 0) {
                        writeValueArray(&result->items, OBJ_VAL(copyString(string->chars + start, i - start)));
                        i += sep->length;
                        start = i;
                    } else {
                        i++;
                    }
                }
                writeValueArray(&result->items, OBJ_VAL(copyString(string->chars + start, string->length - start)));
            }
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "join") == 0) {
            if (argCount != 1 || !IS_LIST(peek(0))) { runtimeError("join() takes 1 list argument"); return false; }
            ObjList* parts = AS_LIST(pop());
            int totalLength = 0;
            for (int i = 0; i < parts->items.count; i++) {
                if (!IS_STRING(parts->items.values[i])) {
                    runtimeError("join() expects a list of strings");
                    return false;
                }
                totalLength += AS_STRING(parts->items.values[i])->length;
                if (i > 0) totalLength += string->length;
            }
            char* chars = (char*)malloc((size_t)totalLength + 1);
            int offset = 0;
            for (int i = 0; i < parts->items.count; i++) {
                if (i > 0 && string->length > 0) {
                    memcpy(chars + offset, string->chars, string->length);
                    offset += string->length;
                }
                ObjString* part = AS_STRING(parts->items.values[i]);
                memcpy(chars + offset, part->chars, part->length);
                offset += part->length;
            }
            chars[offset] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, offset)));
            return true;
        }
    } else if (IS_BYTES(receiver)) {
        ObjBytes* bytes = AS_BYTES(receiver);
        if (strcmp(name->chars, "decode") == 0) {
            if (argCount != 0) { runtimeError("decode() takes 0 arguments"); return false; }
            char* chars = (char*)malloc((size_t)bytes->length + 1);
            memcpy(chars, bytes->bytes, bytes->length);
            chars[bytes->length] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, bytes->length)));
            return true;
        } else if (strcmp(name->chars, "hex") == 0) {
            if (argCount != 0) { runtimeError("hex() takes 0 arguments"); return false; }
            char* chars = (char*)malloc((size_t)bytes->length * 2 + 1);
            for (int i = 0; i < bytes->length; i++) {
                sprintf(chars + i * 2, "%02x", bytes->bytes[i]);
            }
            chars[bytes->length * 2] = '\0';
            pop();
            push(OBJ_VAL(takeString(chars, bytes->length * 2)));
            return true;
        } else if (strcmp(name->chars, "startswith") == 0 || strcmp(name->chars, "endswith") == 0) {
            if (argCount != 1 || !IS_BYTES(peek(0))) {
                runtimeError("%s() takes 1 bytes argument", name->chars);
                return false;
            }
            ObjBytes* other = AS_BYTES(pop());
            bool match = false;
            if (strcmp(name->chars, "startswith") == 0) {
                match = bytes->length >= other->length &&
                        memcmp(bytes->bytes, other->bytes, (size_t)other->length) == 0;
            } else {
                match = bytes->length >= other->length &&
                        memcmp(bytes->bytes + bytes->length - other->length,
                               other->bytes,
                               (size_t)other->length) == 0;
            }
            pop();
            push(BOOL_VAL(match));
            return true;
        } else if (strcmp(name->chars, "count") == 0) {
            if (argCount != 1) { runtimeError("count() takes 1 argument"); return false; }
            int needle = -1;
            if (IS_NUMBER(peek(0))) {
                needle = (int)AS_NUMBER(peek(0));
            } else if (IS_BYTES(peek(0)) && AS_BYTES(peek(0))->length == 1) {
                needle = AS_BYTES(peek(0))->bytes[0];
            } else {
                runtimeError("count() expects a byte value");
                return false;
            }
            pop();
            int count = 0;
            for (int i = 0; i < bytes->length; i++) {
                if (bytes->bytes[i] == needle) count++;
            }
            pop();
            push(NUMBER_VAL((double)count));
            return true;
        } else if (strcmp(name->chars, "split") == 0) {
            if (argCount > 1) { runtimeError("split() takes at most 1 argument"); return false; }
            ObjList* result = newList();
            if (argCount == 0) {
                int i = 0;
                while (i < bytes->length) {
                    while (i < bytes->length && isspace(bytes->bytes[i])) i++;
                    int start = i;
                    while (i < bytes->length && !isspace(bytes->bytes[i])) i++;
                    if (i > start) {
                        writeValueArray(&result->items, OBJ_VAL(newBytes(i - start, bytes->bytes + start)));
                    }
                }
                pop();
                push(OBJ_VAL(result));
                return true;
            }

            if (!IS_BYTES(peek(0))) {
                runtimeError("split() separator must be bytes");
                return false;
            }
            ObjBytes* sep = AS_BYTES(pop());
            if (sep->length == 0) {
                raiseException(createExceptionValue("ValueError", "empty separator"));
                return false;
            }
            int start = 0;
            for (int i = 0; i <= bytes->length - sep->length;) {
                if (memcmp(bytes->bytes + i, sep->bytes, (size_t)sep->length) == 0) {
                    writeValueArray(&result->items, OBJ_VAL(newBytes(i - start, bytes->bytes + start)));
                    i += sep->length;
                    start = i;
                } else {
                    i++;
                }
            }
            writeValueArray(&result->items, OBJ_VAL(newBytes(bytes->length - start, bytes->bytes + start)));
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "index") == 0) {
            if (argCount != 1) { runtimeError("index() takes 1 argument"); return false; }
            int needle = -1;
            if (IS_NUMBER(peek(0))) {
                needle = (int)AS_NUMBER(peek(0));
            } else if (IS_BYTES(peek(0)) && AS_BYTES(peek(0))->length == 1) {
                needle = AS_BYTES(peek(0))->bytes[0];
            } else {
                runtimeError("index() expects a byte value");
                return false;
            }
            pop();
            int index = -1;
            for (int i = 0; i < bytes->length; i++) {
                if (bytes->bytes[i] == needle) {
                    index = i;
                    break;
                }
            }
            pop();
            if (index == -1) {
                raiseException(createExceptionValue("ValueError", "byte not found"));
                return false;
            }
            push(NUMBER_VAL((double)index));
            return true;
        } else if (strcmp(name->chars, "find") == 0) {
            if (argCount != 1) { runtimeError("find() takes 1 argument"); return false; }
            int needle = -1;
            if (IS_NUMBER(peek(0))) {
                needle = (int)AS_NUMBER(peek(0));
            } else if (IS_BYTES(peek(0)) && AS_BYTES(peek(0))->length == 1) {
                needle = AS_BYTES(peek(0))->bytes[0];
            } else {
                runtimeError("find() expects a byte value");
                return false;
            }
            pop();
            int index = -1;
            for (int i = 0; i < bytes->length; i++) {
                if (bytes->bytes[i] == needle) {
                    index = i;
                    break;
                }
            }
            pop();
            push(NUMBER_VAL((double)index));
            return true;
        } else if (strcmp(name->chars, "replace") == 0) {
            if (argCount != 2 || !IS_BYTES(peek(1)) || !IS_BYTES(peek(0))) {
                runtimeError("replace() takes 2 bytes arguments");
                return false;
            }
            ObjBytes* replacement = AS_BYTES(pop());
            ObjBytes* old = AS_BYTES(pop());
            if (old->length == 0) {
                raiseException(createExceptionValue("ValueError", "replace() old value cannot be empty"));
                return false;
            }
            int count = 0;
            for (int i = 0; i <= bytes->length - old->length;) {
                if (memcmp(bytes->bytes + i, old->bytes, (size_t)old->length) == 0) {
                    count++;
                    i += old->length;
                } else {
                    i++;
                }
            }
            int newLength = bytes->length + count * (replacement->length - old->length);
            uint8_t* newBytesData = (uint8_t*)malloc((size_t)newLength);
            int out = 0;
            for (int i = 0; i < bytes->length;) {
                if (i <= bytes->length - old->length &&
                    memcmp(bytes->bytes + i, old->bytes, (size_t)old->length) == 0) {
                    memcpy(newBytesData + out, replacement->bytes, (size_t)replacement->length);
                    out += replacement->length;
                    i += old->length;
                } else {
                    newBytesData[out++] = bytes->bytes[i++];
                }
            }
            ObjBytes* result = newBytes(out, newBytesData);
            free(newBytesData);
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "join") == 0) {
            if (argCount != 1 || !IS_LIST(peek(0))) {
                runtimeError("join() takes 1 list argument");
                return false;
            }
            ObjList* parts = AS_LIST(pop());
            int totalLength = 0;
            for (int i = 0; i < parts->items.count; i++) {
                if (!IS_BYTES(parts->items.values[i])) {
                    runtimeError("join() expects a list of bytes");
                    return false;
                }
                totalLength += AS_BYTES(parts->items.values[i])->length;
                if (i > 0) totalLength += bytes->length;
            }
            uint8_t* joined = (uint8_t*)malloc((size_t)totalLength);
            int offset = 0;
            for (int i = 0; i < parts->items.count; i++) {
                if (i > 0 && bytes->length > 0) {
                    memcpy(joined + offset, bytes->bytes, (size_t)bytes->length);
                    offset += bytes->length;
                }
                ObjBytes* part = AS_BYTES(parts->items.values[i]);
                memcpy(joined + offset, part->bytes, (size_t)part->length);
                offset += part->length;
            }
            ObjBytes* result = newBytes(offset, joined);
            free(joined);
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "strip") == 0 ||
                   strcmp(name->chars, "lstrip") == 0 ||
                   strcmp(name->chars, "rstrip") == 0) {
            if (argCount != 0) { runtimeError("%s() takes 0 arguments", name->chars); return false; }
            int start = 0;
            int end = bytes->length;
            if (strcmp(name->chars, "strip") == 0 || strcmp(name->chars, "lstrip") == 0) {
                while (start < end && isspace(bytes->bytes[start])) start++;
            }
            if (strcmp(name->chars, "strip") == 0 || strcmp(name->chars, "rstrip") == 0) {
                while (end > start && isspace(bytes->bytes[end - 1])) end--;
            }
            ObjBytes* result = newBytes(end - start, bytes->bytes + start);
            pop();
            push(OBJ_VAL(result));
            return true;
        }
    } else if (IS_DICT(receiver)) {
        ObjDict* dict = AS_DICT(receiver);
        if (strcmp(name->chars, "get") == 0) {
            if (argCount < 1 || argCount > 2) { runtimeError("get() takes 1 or 2 arguments"); return false; }
            Value defaultValue = (argCount == 2) ? pop() : NIL_VAL;
            Value key = pop();
            Value value;
            if (tableGet(&dict->table, key, &value)) {
                pop(); // receiver
                push(value);
            } else {
                pop(); // receiver
                push(defaultValue);
            }
            return true;
        } else if (strcmp(name->chars, "keys") == 0) {
            if (argCount != 0) { runtimeError("keys() takes 0 arguments"); return false; }
            pop();
            push(OBJ_VAL(tableKeysList(&dict->table)));
            return true;
        } else if (strcmp(name->chars, "values") == 0) {
            if (argCount != 0) { runtimeError("values() takes 0 arguments"); return false; }
            pop();
            push(OBJ_VAL(tableValuesList(&dict->table)));
            return true;
        } else if (strcmp(name->chars, "items") == 0) {
            if (argCount != 0) { runtimeError("items() takes 0 arguments"); return false; }
            pop();
            push(OBJ_VAL(tableItemsList(&dict->table)));
            return true;
        } else if (strcmp(name->chars, "clear") == 0) {
            if (argCount != 0) { runtimeError("clear() takes 0 arguments"); return false; }
            freeTable(&dict->table);
            initTable(&dict->table);
            pop();
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "copy") == 0) {
            if (argCount != 0) { runtimeError("copy() takes 0 arguments"); return false; }
            ObjDict* result = newDict();
            tableAddAll(&dict->table, &result->table);
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "pop") == 0) {
            if (argCount < 1 || argCount > 2) { runtimeError("pop() takes 1 or 2 arguments"); return false; }
            Value defaultValue = (argCount == 2) ? pop() : NIL_VAL;
            Value key = pop();
            Value value;
            bool found = tableGet(&dict->table, key, &value);
            if (found) tableDelete(&dict->table, key);
            pop();
            if (found) {
                push(value);
            } else if (argCount == 2) {
                push(defaultValue);
            } else {
                raiseException(createExceptionValue("KeyError", "Key not found"));
                return false;
            }
            return true;
        } else if (strcmp(name->chars, "popitem") == 0) {
            if (argCount != 0) { runtimeError("popitem() takes 0 arguments"); return false; }
            bool found = false;
            Value pair = tablePopFirst(&dict->table, &found);
            pop();
            if (!found) {
                raiseException(createExceptionValue("KeyError", "popitem(): dictionary is empty"));
                return false;
            }
            push(pair);
            return true;
        } else if (strcmp(name->chars, "setdefault") == 0) {
            if (argCount < 1 || argCount > 2) { runtimeError("setdefault() takes 1 or 2 arguments"); return false; }
            Value defaultValue = (argCount == 2) ? pop() : NIL_VAL;
            Value key = pop();
            Value value;
            bool found = tableGet(&dict->table, key, &value);
            if (!found) {
                value = defaultValue;
                tableSet(&dict->table, key, value);
            }
            pop();
            push(value);
            return true;
        } else if (strcmp(name->chars, "fromkeys") == 0) {
            if (argCount < 1 || argCount > 2) { runtimeError("fromkeys() takes 1 or 2 arguments"); return false; }
            Value defaultValue = (argCount == 2) ? pop() : NIL_VAL;
            Value keysValue = pop();
            ObjDict* result = newDict();
            if (IS_LIST(keysValue)) {
                ObjList* keys = AS_LIST(keysValue);
                for (int i = 0; i < keys->items.count; i++) {
                    tableSet(&result->table, keys->items.values[i], defaultValue);
                }
            } else if (IS_TUPLE(keysValue)) {
                ObjTuple* keys = AS_TUPLE(keysValue);
                for (int i = 0; i < keys->items.count; i++) {
                    tableSet(&result->table, keys->items.values[i], defaultValue);
                }
            } else if (IS_SET(keysValue)) {
                Table* table = &AS_SET(keysValue)->table;
                for (int i = 0; i < table->capacity; i++) {
                    Entry* entry = &table->entries[i];
                    if (!IS_NIL(entry->key)) {
                        tableSet(&result->table, entry->key, defaultValue);
                    }
                }
            } else if (IS_DICT(keysValue)) {
                Table* table = &AS_DICT(keysValue)->table;
                for (int i = 0; i < table->capacity; i++) {
                    Entry* entry = &table->entries[i];
                    if (!IS_NIL(entry->key)) {
                        tableSet(&result->table, entry->key, defaultValue);
                    }
                }
            } else {
                runtimeError("fromkeys() expects an iterable of keys");
                return false;
            }
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "update") == 0) {
            if (argCount != 1 || !IS_DICT(peek(0))) { runtimeError("update() takes 1 dict argument"); return false; }
            ObjDict* other = AS_DICT(pop());
            tableAddAll(&other->table, &dict->table);
            pop();
            push(NIL_VAL);
            return true;
        }
    } else if (IS_SET(receiver)) {
        ObjSet* set = AS_SET(receiver);
        if (strcmp(name->chars, "add") == 0) {
            if (argCount != 1) { runtimeError("add() takes 1 argument"); return false; }
            tableSet(&set->table, peek(0), NIL_VAL);
            pop(); pop(); push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "remove") == 0) {
            if (argCount != 1) { runtimeError("remove() takes 1 argument"); return false; }
            if (!tableDelete(&set->table, peek(0))) {
                raiseException(createExceptionValue("KeyError", "Item not found in set"));
                return false;
            }
            pop(); pop(); push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "discard") == 0) {
            if (argCount != 1) { runtimeError("discard() takes 1 argument"); return false; }
            tableDelete(&set->table, peek(0));
            pop(); pop(); push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "clear") == 0) {
            if (argCount != 0) { runtimeError("clear() takes 0 arguments"); return false; }
            freeTable(&set->table);
            initTable(&set->table);
            pop(); push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "copy") == 0) {
            if (argCount != 0) { runtimeError("copy() takes 0 arguments"); return false; }
            ObjSet* result = newSet();
            tableAddAll(&set->table, &result->table);
            pop(); push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "union") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("union() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            ObjSet* result = newSet();
            tableAddAll(&set->table, &result->table);
            tableAddAll(&other->table, &result->table);
            pop(); push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "update") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("update() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            tableAddAll(&other->table, &set->table);
            pop();
            push(NIL_VAL);
            return true;
        } else if (strcmp(name->chars, "intersection") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("intersection() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            ObjSet* result = newSet();
            for (int i = 0; i < set->table.capacity; i++) {
                Entry* entry = &set->table.entries[i];
                if (!IS_NIL(entry->key)) {
                    Value value;
                    if (tableGet(&other->table, entry->key, &value)) {
                        tableSet(&result->table, entry->key, NIL_VAL);
                    }
                }
            }
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "difference") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("difference() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            ObjSet* result = newSet();
            for (int i = 0; i < set->table.capacity; i++) {
                Entry* entry = &set->table.entries[i];
                if (!IS_NIL(entry->key)) {
                    Value value;
                    if (!tableGet(&other->table, entry->key, &value)) {
                        tableSet(&result->table, entry->key, NIL_VAL);
                    }
                }
            }
            pop();
            push(OBJ_VAL(result));
            return true;
        } else if (strcmp(name->chars, "issubset") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("issubset() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            bool ok = true;
            for (int i = 0; i < set->table.capacity; i++) {
                Entry* entry = &set->table.entries[i];
                if (!IS_NIL(entry->key)) {
                    Value value;
                    if (!tableGet(&other->table, entry->key, &value)) {
                        ok = false;
                        break;
                    }
                }
            }
            pop();
            push(BOOL_VAL(ok));
            return true;
        } else if (strcmp(name->chars, "issuperset") == 0) {
            if (argCount != 1 || !IS_SET(peek(0))) { runtimeError("issuperset() takes 1 set argument"); return false; }
            ObjSet* other = AS_SET(pop());
            bool ok = true;
            for (int i = 0; i < other->table.capacity; i++) {
                Entry* entry = &other->table.entries[i];
                if (!IS_NIL(entry->key)) {
                    Value value;
                    if (!tableGet(&set->table, entry->key, &value)) {
                        ok = false;
                        break;
                    }
                }
            }
            pop();
            push(BOOL_VAL(ok));
            return true;
        }
    } else if (IS_TUPLE(receiver)) {
        ObjTuple* tuple = AS_TUPLE(receiver);
        if (strcmp(name->chars, "count") == 0) {
            if (argCount != 1) { runtimeError("count() takes 1 argument"); return false; }
            Value target = pop();
            int count = 0;
            for (int i = 0; i < tuple->items.count; i++) {
                if (valuesEqual(tuple->items.values[i], target)) count++;
            }
            pop();
            push(NUMBER_VAL((double)count));
            return true;
        } else if (strcmp(name->chars, "index") == 0) {
            if (argCount != 1) { runtimeError("index() takes 1 argument"); return false; }
            Value target = pop();
            int index = -1;
            for (int i = 0; i < tuple->items.count; i++) {
                if (valuesEqual(tuple->items.values[i], target)) {
                    index = i;
                    break;
                }
            }
            pop();
            if (index == -1) {
                raiseException(createExceptionValue("ValueError", "tuple.index(x): x not in tuple"));
                return false;
            }
            push(NUMBER_VAL((double)index));
            return true;
        }
    }
    
    runtimeError("Only instances have methods.");
    return false;
}

static bool invoke(ObjString* name, int argCount) {
    Value receiver = peek(argCount);

    if (!IS_INSTANCE(receiver)) {
        return invokeBuiltinMethod(receiver, name, argCount);
    }

    ObjInstance* instance = AS_INSTANCE(receiver);

    Value value;
    if (tableGet(&instance->fields, OBJ_VAL(name), &value)) {
        vm.stackTop[-argCount - 1] = value;
        return callValue(value, argCount);
    }

    return invokeFromClass(instance->klass, name, argCount);
}

static bool callValueExpanded(Value callee, int argCount, ObjTuple* starIndexes) {
    Value sourceArgs[255];
    for (int i = 0; i < argCount; i++) {
        sourceArgs[argCount - 1 - i] = pop();
    }
    pop();

    Value expandedArgs[255];
    int expandedCount = 0;
    if (!collectExpandedArgs(sourceArgs, argCount, starIndexes, expandedArgs, &expandedCount)) {
        return false;
    }

    push(callee);
    for (int i = 0; i < expandedCount; i++) {
        push(expandedArgs[i]);
    }
    return callValue(callee, expandedCount);
}

static bool invokeExpanded(ObjString* name, int argCount, ObjTuple* starIndexes) {
    Value sourceArgs[255];
    for (int i = 0; i < argCount; i++) {
        sourceArgs[argCount - 1 - i] = pop();
    }
    Value receiver = pop();

    Value expandedArgs[255];
    int expandedCount = 0;
    if (!collectExpandedArgs(sourceArgs, argCount, starIndexes, expandedArgs, &expandedCount)) {
        return false;
    }

    push(receiver);
    for (int i = 0; i < expandedCount; i++) {
        push(expandedArgs[i]);
    }
    return invoke(name, expandedCount);
}

static bool collectExpandedKeywords(Value* kwSourceArgs, int kwSourceCount, ObjTuple* kwStarIndexes,
                                    Value* explicitKwNames, int explicitKwCount,
                                    Value* outKwNames, Value* outKwValues, int* outKwCount) {
    int explicitCursor = 0;
    int kwStarCursor = 0;

    for (int i = 0; i < kwSourceCount; i++) {
        bool isKwStar = false;
        if (kwStarCursor < kwStarIndexes->items.count) {
            Value marker = kwStarIndexes->items.values[kwStarCursor];
            if (IS_NUMBER(marker) && (int)AS_NUMBER(marker) == i) {
                isKwStar = true;
                kwStarCursor++;
            }
        }

        if (isKwStar) {
            if (!IS_DICT(kwSourceArgs[i])) {
                runtimeError("** argument must be a dict.");
                return false;
            }
            ObjDict* dict = AS_DICT(kwSourceArgs[i]);
            for (int j = 0; j < dict->table.capacity; j++) {
                Entry* entry = &dict->table.entries[j];
                if (IS_NIL(entry->key)) {
                    continue;
                }
                if (!IS_STRING(entry->key)) {
                    runtimeError("Keyword must be a string.");
                    return false;
                }
                outKwNames[*outKwCount] = entry->key;
                outKwValues[*outKwCount] = entry->value;
                (*outKwCount)++;
            }
        } else {
            if (explicitCursor >= explicitKwCount) {
                runtimeError("Invalid expanded keyword argument state.");
                return false;
            }
            outKwNames[*outKwCount] = explicitKwNames[explicitCursor++];
            outKwValues[*outKwCount] = kwSourceArgs[i];
            (*outKwCount)++;
        }
    }

    return true;
}

static bool callValueExpandedKeyword(Value callee, int posCount, int kwSourceCount, int explicitKwCount,
                                     ObjTuple* posStarIndexes, ObjTuple* kwStarIndexes) {
    Value explicitKwNames[255];
    for (int i = 0; i < explicitKwCount; i++) {
        explicitKwNames[explicitKwCount - 1 - i] = pop();
    }

    Value kwSourceArgs[255];
    for (int i = 0; i < kwSourceCount; i++) {
        kwSourceArgs[kwSourceCount - 1 - i] = pop();
    }

    Value posSourceArgs[255];
    for (int i = 0; i < posCount; i++) {
        posSourceArgs[posCount - 1 - i] = pop();
    }

    pop();

    Value expandedPosArgs[255];
    int expandedPosCount = 0;
    if (!collectExpandedArgs(posSourceArgs, posCount, posStarIndexes, expandedPosArgs, &expandedPosCount)) {
        return false;
    }

    Value expandedKwNames[255];
    Value expandedKwValues[255];
    int expandedKwCount = 0;
    if (!collectExpandedKeywords(kwSourceArgs, kwSourceCount, kwStarIndexes,
                                 explicitKwNames, explicitKwCount,
                                 expandedKwNames, expandedKwValues, &expandedKwCount)) {
        return false;
    }

    push(callee);
    for (int i = 0; i < expandedPosCount; i++) {
        push(expandedPosArgs[i]);
    }
    for (int i = 0; i < expandedKwCount; i++) {
        push(expandedKwValues[i]);
    }
    for (int i = 0; i < expandedKwCount; i++) {
        push(expandedKwNames[i]);
    }
    return callValueKeyword(callee, expandedPosCount, expandedKwCount);
}

static bool invokeExpandedKeyword(ObjString* name, int posCount, int kwSourceCount, int explicitKwCount,
                                  ObjTuple* posStarIndexes, ObjTuple* kwStarIndexes) {
    Value explicitKwNames[255];
    for (int i = 0; i < explicitKwCount; i++) {
        explicitKwNames[explicitKwCount - 1 - i] = pop();
    }

    Value kwSourceArgs[255];
    for (int i = 0; i < kwSourceCount; i++) {
        kwSourceArgs[kwSourceCount - 1 - i] = pop();
    }

    Value posSourceArgs[255];
    for (int i = 0; i < posCount; i++) {
        posSourceArgs[posCount - 1 - i] = pop();
    }

    Value receiver = pop();

    Value expandedPosArgs[255];
    int expandedPosCount = 0;
    if (!collectExpandedArgs(posSourceArgs, posCount, posStarIndexes, expandedPosArgs, &expandedPosCount)) {
        return false;
    }

    Value expandedKwNames[255];
    Value expandedKwValues[255];
    int expandedKwCount = 0;
    if (!collectExpandedKeywords(kwSourceArgs, kwSourceCount, kwStarIndexes,
                                 explicitKwNames, explicitKwCount,
                                 expandedKwNames, expandedKwValues, &expandedKwCount)) {
        return false;
    }

    push(receiver);
    for (int i = 0; i < expandedPosCount; i++) {
        push(expandedPosArgs[i]);
    }
    for (int i = 0; i < expandedKwCount; i++) {
        push(expandedKwValues[i]);
    }
    for (int i = 0; i < expandedKwCount; i++) {
        push(expandedKwNames[i]);
    }
    return invokeKeyword(name, expandedPosCount, expandedKwCount);
}

static bool bindMethod(ObjClass* klass, ObjString* name) {
    Value method;
    if (!tableGet(&klass->methods, OBJ_VAL(name), &method)) {
        raiseException(createExceptionValue("AttributeError", "Undefined property."));
        return false;
    }

    ObjBoundMethod* bound = newBoundMethod(peek(0), method);
    pop();
    push(OBJ_VAL(bound));
    return true;
}

static Value importModuleValue(ObjString* moduleName) {
    Value cached;
    if (tableGet(&vm.modules, OBJ_VAL(moduleName), &cached)) {
        return cached;
    }

    ObjString* parentName = parentModuleName(moduleName);
    Value parentModule = NIL_VAL;
    ObjString* childName = NULL;
    if (parentName != NULL) {
        parentModule = importModuleValue(parentName);
        if (IS_NIL(parentModule)) {
            return NIL_VAL;
        }
        childName = childModuleName(moduleName);
    }

    char* source = readModuleFile(moduleName->chars);
    if (source == NULL) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "Module not found: %s", moduleName->chars);
        raiseException(createExceptionValue("RuntimeError", buffer));
        return NIL_VAL;
    }

    ObjInstance* module = newInstance(vm.moduleClass);
    Value moduleValue = OBJ_VAL(module);
    tableSet(&module->fields, OBJ_VAL(copyString("__name__", 8)), OBJ_VAL(moduleName));
    tableSet(&vm.modules, OBJ_VAL(moduleName), moduleValue);

    ObjEnvironment* moduleEnv = newEnvironmentWithTable(vm.globalEnv, &module->fields);
    InterpretResult result = interpretInGlobals(source, moduleName->chars, moduleEnv);
    free(source);
    if (result != INTERPRET_OK) {
        tableDelete(&vm.modules, OBJ_VAL(moduleName));
        return NIL_VAL;
    }

    if (parentName != NULL && IS_INSTANCE(parentModule) && childName != NULL) {
        tableSet(&AS_INSTANCE(parentModule)->fields, OBJ_VAL(childName), moduleValue);
    }
    return moduleValue;
}

static void defineMethod(ObjString* name) {
    Value method = peek(0);
    ObjClass* klass = AS_CLASS(peek(1));
    tableSet(&klass->methods, OBJ_VAL(name), method);
    pop();
}

static InterpretResult run() {
    CallFrame* frame = &vm.frames[vm.frameCount - 1];

#define READ_BYTE() (*frame->ip++)
#define READ_CONSTANT() (frame->function->chunk.constants.values[READ_BYTE()])
#define READ_SHORT() \
    (frame->ip += 2, (uint16_t)((frame->ip[-2] << 8) | frame->ip[-1]))

#define RUNTIME_ERROR(...) \
    do { \
        if (runtimeError(__VA_ARGS__)) { \
            frame = &vm.frames[vm.frameCount - 1]; \
            goto decode; \
        } \
        return INTERPRET_RUNTIME_ERROR; \
    } while (false)

#define BINARY_OP(valueType, op) \
    do { \
      if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) { \
        RUNTIME_ERROR("Operands must be numbers."); \
      } \
      double b = AS_NUMBER(pop()); \
      double a = AS_NUMBER(pop()); \
      push(valueType(a op b)); \
    } while (false)

#define BITWISE_OP(op) \
    do { \
      if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) { \
        RUNTIME_ERROR("Operands must be numbers."); \
      } \
      int64_t b = (int64_t)AS_NUMBER(pop()); \
      int64_t a = (int64_t)AS_NUMBER(pop()); \
      push(NUMBER_VAL((double)(a op b))); \
    } while (false)

#define HANDLE_RE() \
    do { \
        if (vm.frameCount > 0) { \
            frame = &vm.frames[vm.frameCount - 1]; \
            goto decode; \
        } \
        return INTERPRET_RUNTIME_ERROR; \
    } while (false)

    int initialFrameCount = vm.frameCount;
    for (;;) {
    decode:
#ifdef DEBUG_TRACE_EXECUTION
        printf("          ");
        for (Value* slot = vm.stack; slot < vm.stackTop; slot++) {
            printf("[ ");
            printValue(*slot);
            printf(" ]");
        }
        printf("\n");
        disassembleInstruction(&frame->function->chunk, (int)(frame->ip - frame->function->chunk.code));
#endif

        uint8_t instruction;
        switch (instruction = READ_BYTE()) {
            case OP_CONSTANT: {
                Value constant = READ_CONSTANT();
                if (IS_FUNCTION(constant)) {
                    constant = OBJ_VAL(bindFunctionForFrame(AS_FUNCTION(constant), frame));
                }
                push(constant);
                break;
            }
            case OP_NIL: push(NIL_VAL); break;
            case OP_TRUE: push(BOOL_VAL(true)); break;
            case OP_FALSE: push(BOOL_VAL(false)); break;
            case OP_POP: pop(); break;
            case OP_GET_LOCAL: {
                uint8_t slot = READ_BYTE();
                push(frame->slots[slot]);
                break;
            }
            case OP_SET_LOCAL: {
                uint8_t slot = READ_BYTE();
                frame->slots[slot] = peek(0);
                syncFrameLocalToEnv(frame, slot);
                break;
            }
            case OP_GET_GLOBAL: {
                Value constant = READ_CONSTANT();
                ObjString* name = AS_STRING(constant);
                Value value;
                if (!environmentGet(frame->env, OBJ_VAL(name), &value)) {
                    RUNTIME_ERROR("Undefined variable '%s'.", name->chars);
                }
                push(value);
                break;
            }
            case OP_DEFINE_GLOBAL: {
                ObjString* name = AS_STRING(READ_CONSTANT());
                environmentSetLocal(frame->env, OBJ_VAL(name), peek(0));
                pop();
                break;
            }
            case OP_SET_GLOBAL: {
                ObjString* name = AS_STRING(READ_CONSTANT());
                environmentSetLocal(frame->env, OBJ_VAL(name), peek(0));
                break;
            }
            case OP_EQUAL: {
                Value b = pop();
                Value a = pop();
                push(BOOL_VAL(valuesEqual(a, b)));
                break;
            }
            case OP_IS: {
                Value b = pop();
                Value a = pop();
                if (a.type != b.type) {
                    push(BOOL_VAL(false));
                } else if (IS_OBJ(a)) {
                    push(BOOL_VAL(AS_OBJ(a) == AS_OBJ(b)));
                } else {
                    push(BOOL_VAL(valuesEqual(a, b)));
                }
                break;
            }
            case OP_CONTAINS: {
                Value container = pop();
                Value item = pop();
                bool found = false;
                if (IS_LIST(container)) {
                    ObjList* list = AS_LIST(container);
                    for (int i = 0; i < list->items.count; i++) {
                        if (valuesEqual(item, list->items.values[i])) {
                            found = true;
                            break;
                        }
                    }
                } else if (IS_STRING(container)) {
                    if (IS_STRING(item)) {
                        ObjString* haystack = AS_STRING(container);
                        ObjString* needle = AS_STRING(item);
                        found = strstr(haystack->chars, needle->chars) != NULL;
                    }
                } else if (IS_BYTES(container)) {
                    ObjBytes* haystack = AS_BYTES(container);
                    if (IS_NUMBER(item)) {
                        int needle = (int)AS_NUMBER(item);
                        for (int i = 0; i < haystack->length; i++) {
                            if (haystack->bytes[i] == needle) {
                                found = true;
                                break;
                            }
                        }
                    } else if (IS_BYTES(item)) {
                        ObjBytes* needle = AS_BYTES(item);
                        if (needle->length == 0) {
                            found = true;
                        } else if (needle->length <= haystack->length) {
                            for (int i = 0; i <= haystack->length - needle->length; i++) {
                                if (memcmp(haystack->bytes + i, needle->bytes, needle->length) == 0) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                } else if (IS_DICT(container)) {
                    Value val;
                    found = tableGet(&AS_DICT(container)->table, item, &val);
                } else if (IS_SET(container)) {
                    Value val;
                    found = tableGet(&AS_SET(container)->table, item, &val);
                }
                push(BOOL_VAL(found));
                break;
            }
            case OP_LIST_EXTEND: {
                ObjList* other = AS_LIST(pop());
                ObjList* list = AS_LIST(pop());
                for (int i = 0; i < other->items.count; i++) {
                    writeValueArray(&list->items, other->items.values[i]);
                }
                push(NIL_VAL);
                break;
            }
            case OP_GREATER:  BINARY_OP(BOOL_VAL, >); break;
            case OP_LESS:     BINARY_OP(BOOL_VAL, <); break;
            case OP_ADD: {
                if (IS_STRING(peek(0)) || IS_STRING(peek(1))) {
                    Value other = IS_STRING(peek(0)) ? peek(1) : peek(0);
                    
                    // Convert 'other' to string
                    Value converted;
                    if (IS_STRING(other)) {
                        converted = other;
                    } else if (IS_NUMBER(other)) {
                        char buf[32];
                        int len = sprintf(buf, "%g", AS_NUMBER(other));
                        converted = OBJ_VAL(copyString(buf, len));
                    } else if (IS_BOOL(other)) {
                        converted = AS_BOOL(other) ? OBJ_VAL(copyString("True", 4)) : OBJ_VAL(copyString("False", 5));
                    } else if (IS_NIL(other)) {
                        converted = OBJ_VAL(copyString("None", 4));
                    } else {
                        converted = OBJ_VAL(copyString("<object>", 8));
                    }
                    
                    ObjString* a = IS_STRING(peek(1)) ? AS_STRING(peek(1)) : AS_STRING(converted);
                    ObjString* b = IS_STRING(peek(0)) ? AS_STRING(peek(0)) : AS_STRING(converted);
                    
                    int length = a->length + b->length;
                    char* chars = malloc(length + 1);
                    memcpy(chars, a->chars, a->length);
                    memcpy(chars + a->length, b->chars, b->length);
                    chars[length] = '\0';
                    
                    pop(); pop();
                    push(OBJ_VAL(takeString(chars, length)));
                } else if (IS_LIST(peek(0)) && IS_LIST(peek(1))) {
                    ObjList* b = AS_LIST(pop());
                    ObjList* a = AS_LIST(pop());
                    ObjList* result = newList();
                    for (int i = 0; i < a->items.count; i++) {
                        writeValueArray(&result->items, a->items.values[i]);
                    }
                    for (int i = 0; i < b->items.count; i++) {
                        writeValueArray(&result->items, b->items.values[i]);
                    }
                    push(OBJ_VAL(result));
                } else {
                    BINARY_OP(NUMBER_VAL, +);
                }
                break;
            }
            case OP_SUBTRACT: BINARY_OP(NUMBER_VAL, -); break;
            case OP_MULTIPLY: {
                if (IS_NUMBER(peek(0))) {
                    double count = AS_NUMBER(pop());
                    if (IS_LIST(peek(0))) {
                        ObjList* list = AS_LIST(pop());
                        ObjList* result = newList();
                        for (int i = 0; i < (int)count; i++) {
                            for (int j = 0; j < list->items.count; j++) {
                                writeValueArray(&result->items, list->items.values[j]);
                            }
                        }
                        push(OBJ_VAL(result));
                        break;
                    } else if (IS_STRING(peek(0))) {
                        ObjString* str = AS_STRING(pop());
                        int length = str->length * (int)count;
                        char* chars = malloc(length + 1);
                        for (int i = 0; i < (int)count; i++) {
                            memcpy(chars + i * str->length, str->chars, str->length);
                        }
                        chars[length] = '\0';
                        push(OBJ_VAL(takeString(chars, length)));
                        break;
                    } else {
                        push(NUMBER_VAL(AS_NUMBER(pop()) * count));
                        break;
                    }
                }
                BINARY_OP(NUMBER_VAL, *);
                break;
            }
            case OP_DIVIDE: {
                if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) {
                    RUNTIME_ERROR("Operands must be numbers.");
                }
                double b = AS_NUMBER(pop());
                if (b == 0) {
                    RUNTIME_ERROR("Division by zero.");
                }
                double a = AS_NUMBER(pop());
                push(NUMBER_VAL(a / b));
                break;
            }
            case OP_MODULO: {
                if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) {
                    runtimeError("Operands must be numbers.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                double b = AS_NUMBER(pop());
                if (b == 0) {
                    if (raiseException(createExceptionValue("ZeroDivisionError", "Division by zero."))) {
                        HANDLE_RE();
                    }
                    return INTERPRET_RUNTIME_ERROR;
                }
                double a = AS_NUMBER(pop());
                // Python's modulo: r = a - b * floor(a / b)
                push(NUMBER_VAL(a - b * floor(a / b)));
                break;
            }
            case OP_FLOOR_DIVIDE: {
                if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) {
                    runtimeError("Operands must be numbers.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                double b = AS_NUMBER(pop());
                if (b == 0) {
                    if (raiseException(createExceptionValue("ZeroDivisionError", "Division by zero."))) {
                        HANDLE_RE();
                    }
                    return INTERPRET_RUNTIME_ERROR;
                }
                double a = AS_NUMBER(pop());
                push(NUMBER_VAL(floor(a / b)));
                break;
            }
            case OP_POWER: {
                if (!IS_NUMBER(peek(0)) || !IS_NUMBER(peek(1))) {
                    runtimeError("Operands must be numbers.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                double b = AS_NUMBER(pop());
                double a = AS_NUMBER(pop());
                push(NUMBER_VAL(pow(a, b)));
                break;
            }
            case OP_BIT_AND: BITWISE_OP(&); break;
            case OP_BIT_OR:  BITWISE_OP(|); break;
            case OP_BIT_XOR: BITWISE_OP(^); break;
            case OP_BIT_NOT: {
                if (!IS_NUMBER(peek(0))) {
                    runtimeError("Operand must be a number.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                int64_t a = (int64_t)AS_NUMBER(pop());
                push(NUMBER_VAL((double)(~a)));
                break;
            }
            case OP_BUILD_LIST: {
                uint8_t count = READ_BYTE();
                ObjList* list = newList();
                for (int i = 0; i < count; i++) {
                    writeValueArray(&list->items, NIL_VAL);
                }
                for (int i = count - 1; i >= 0; i--) {
                    list->items.values[i] = pop();
                }
                push(OBJ_VAL(list));
                break;
            }
            case OP_BUILD_TUPLE: {
                uint8_t count = READ_BYTE();
                ObjTuple* tuple = newTuple();
                for (int i = 0; i < count; i++) {
                    writeValueArray(&tuple->items, NIL_VAL);
                }
                for (int i = count - 1; i >= 0; i--) {
                    tuple->items.values[i] = pop();
                }
                push(OBJ_VAL(tuple));
                break;
            }
            case OP_BUILD_SET: {
                uint8_t count = READ_BYTE();
                ObjSet* set = newSet();
                for (int i = 0; i < count; i++) {
                    Value item = pop();
                    tableSet(&set->table, item, NIL_VAL);
                }
                push(OBJ_VAL(set));
                break;
            }
            case OP_BUILD_DICT: {
                uint8_t count = READ_BYTE(); // number of entries (key-value pairs)
                ObjDict* dict = newDict();
                for (int i = 0; i < count; i++) {
                    Value value = pop();
                    Value key = pop();
                    tableSet(&dict->table, key, value);
                }
                push(OBJ_VAL(dict));
                break;
            }
            case OP_BUILD_SLICE: {
                Value step = pop();
                Value stop = pop();
                Value start = pop();
                push(OBJ_VAL(newSlice(start, stop, step)));
                break;
            }
            case OP_GET_ITER: {
                Value obj = peek(0);
                if (!IS_LIST(obj) && !IS_TUPLE(obj) && !IS_DICT(obj) && !IS_SET(obj) && !IS_STRING(obj)) {
                    runtimeError("Object is not iterable.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vm.stackTop[-1] = OBJ_VAL(newIterator(obj));
                break;
            }
            case OP_FOR_ITER: {
                uint16_t offset = READ_SHORT();
                Value iterVal = peek(0);
                if (!IS_ITERATOR(iterVal)) {
                    printf("CRITICAL: OP_FOR_ITER expected iterator, got type %d\n", iterVal.type);
                    if (IS_OBJ(iterVal)) printf("Obj type: %d\n", AS_OBJ(iterVal)->type);
                }
                ObjIterator* iterator = AS_ITERATOR(iterVal);
                Value iterable = iterator->iterable;
                
                bool hasNext = false;
                Value nextVal = NIL_VAL;
                
                if (IS_LIST(iterable)) {
                    ObjList* list = AS_LIST(iterable);
                    if (iterator->index < list->items.count) {
                        nextVal = list->items.values[iterator->index++];
                        hasNext = true;
                    }
                } else if (IS_TUPLE(iterable)) {
                    ObjTuple* tuple = AS_TUPLE(iterable);
                    if (iterator->index < tuple->items.count) {
                        nextVal = tuple->items.values[iterator->index++];
                        hasNext = true;
                    }
                } else if (IS_SET(iterable)) {
                    ObjSet* set = AS_SET(iterable);
                    while (iterator->index < set->table.capacity) {
                        Entry* entry = &set->table.entries[iterator->index++];
                        if (!IS_NIL(entry->key)) {
                            nextVal = entry->key;
                            hasNext = true;
                            break;
                        }
                    }
                } else if (IS_DICT(iterable)) {
                    ObjDict* dict = AS_DICT(iterable);
                    while (iterator->index < dict->table.capacity) {
                        Entry* entry = &dict->table.entries[iterator->index++];
                        if (!IS_NIL(entry->key)) {
                            nextVal = entry->key;
                            hasNext = true;
                            break;
                        }
                    }
                } else if (IS_STRING(iterable)) {
                    ObjString* string = AS_STRING(iterable);
                    if (iterator->index < string->length) {
                        char c = string->chars[iterator->index++];
                        nextVal = OBJ_VAL(copyString(&c, 1));
                        hasNext = true;
                    }
                }
                
                if (hasNext) {
                    push(nextVal);
                } else {
                    frame->ip += offset; 
                }
                break;
            }
            case OP_SUBSCRIPT: {
                Value index = pop();
                Value obj = pop();
                if (IS_LIST(obj)) {
                    ObjList* list = AS_LIST(obj);
                    if (IS_SLICE(index)) {
                        ObjSlice* slice = AS_SLICE(index);
                        int step = IS_NIL(slice->step) ? 1 : (int)AS_NUMBER(slice->step);
                        if (step == 0) {
                            runtimeError("slice step cannot be zero");
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        int start = IS_NIL(slice->start) ? (step > 0 ? 0 : list->items.count - 1) : (int)AS_NUMBER(slice->start);
                        int stop = IS_NIL(slice->stop) ? (step > 0 ? list->items.count : -1) : (int)AS_NUMBER(slice->stop);
                        if (start < 0) start += list->items.count;
                        if (stop < 0 && !IS_NIL(slice->stop)) stop += list->items.count;
                        ObjList* result = newList();
                        for (int i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
                            if (i >= 0 && i < list->items.count) {
                                writeValueArray(&result->items, list->items.values[i]);
                            }
                        }
                        push(OBJ_VAL(result));
                        break;
                    } else if (IS_NUMBER(index)) {
                        int idx = (int)AS_NUMBER(index);
                        if (idx < 0) idx += list->items.count;
                        if (idx < 0 || idx >= list->items.count) {
                            if (raiseException(createExceptionValue("IndexError", "List index out of range."))) {
                                HANDLE_RE();
                            }
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        push(list->items.values[idx]);
                    } else {
                        runtimeError("List index must be a number or slice.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                } else if (IS_TUPLE(obj)) {
                    ObjTuple* tuple = AS_TUPLE(obj);
                    if (!IS_NUMBER(index)) {
                        runtimeError("Tuple index must be a number.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    int idx = (int)AS_NUMBER(index);
                    if (idx < 0) idx += tuple->items.count;
                    if (idx < 0 || idx >= tuple->items.count) {
                        runtimeError("Tuple index out of range.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    push(tuple->items.values[idx]);
                } else if (IS_DICT(obj)) {
                    ObjDict* dict = AS_DICT(obj);
                    Value value;
                    if (!tableGet(&dict->table, index, &value)) {
                        if (raiseException(createExceptionValue("KeyError", "Key error."))) {
                            HANDLE_RE();
                        }
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    push(value);
                } else if (IS_STRING(obj)) {
                    ObjString* string = AS_STRING(obj);
                    if (!IS_NUMBER(index)) {
                        runtimeError("String index must be a number.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    int idx = (int)AS_NUMBER(index);
                    if (idx < 0) idx += string->length;
                    if (idx < 0 || idx >= string->length) {
                        if (raiseException(createExceptionValue("IndexError", "String index out of range."))) {
                            HANDLE_RE();
                        }
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    char c[2] = {string->chars[idx], '\0'};
                    push(OBJ_VAL(copyString(c, 1)));
                } else if (IS_BYTES(obj)) {
                    ObjBytes* bytes = AS_BYTES(obj);
                    if (IS_SLICE(index)) {
                        ObjSlice* slice = AS_SLICE(index);
                        int step = IS_NIL(slice->step) ? 1 : (int)AS_NUMBER(slice->step);
                        if (step == 0) {
                            runtimeError("slice step cannot be zero");
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        int start = IS_NIL(slice->start) ? (step > 0 ? 0 : bytes->length - 1) : (int)AS_NUMBER(slice->start);
                        int stop = IS_NIL(slice->stop) ? (step > 0 ? bytes->length : -1) : (int)AS_NUMBER(slice->stop);
                        if (start < 0) start += bytes->length;
                        if (stop < 0 && !IS_NIL(slice->stop)) stop += bytes->length;
                        uint8_t* tmp = (uint8_t*)malloc((size_t)(bytes->length + 1));
                        int count = 0;
                        for (int i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
                            if (i >= 0 && i < bytes->length) {
                                tmp[count++] = bytes->bytes[i];
                            }
                        }
                        ObjBytes* result = newBytes(count, tmp);
                        free(tmp);
                        push(OBJ_VAL(result));
                    } else if (IS_NUMBER(index)) {
                        int idx = (int)AS_NUMBER(index);
                        if (idx < 0) idx += bytes->length;
                        if (idx < 0 || idx >= bytes->length) {
                            if (raiseException(createExceptionValue("IndexError", "Bytes index out of range."))) {
                                HANDLE_RE();
                            }
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        push(NUMBER_VAL((double)bytes->bytes[idx]));
                    } else {
                        runtimeError("Bytes index must be a number or slice.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                } else {
                    runtimeError("Object is not subscriptable.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_STORE_SUBSCRIPT: {
                Value value = pop();
                Value index = pop();
                Value obj = pop();
                if (IS_LIST(obj)) {
                    ObjList* list = AS_LIST(obj);
                    if (IS_SLICE(index)) {
                        if (!listAssignSlice(list, AS_SLICE(index), value)) {
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        push(value);
                        break;
                    }
                    if (!IS_NUMBER(index)) {
                        runtimeError("List index must be a number.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    int idx = (int)AS_NUMBER(index);
                    if (idx < 0) idx += list->items.count;
                    if (idx < 0 || idx >= list->items.count) {
                        if (raiseException(createExceptionValue("IndexError", "List index out of range."))) {
                            HANDLE_RE();
                        }
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    list->items.values[idx] = value;
                    push(value); // assignment returns the value
                } else if (IS_DICT(obj)) {
                    ObjDict* dict = AS_DICT(obj);
                    tableSet(&dict->table, index, value);
                    push(value);
                } else {
                    runtimeError("Object does not support item assignment.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_DELETE_SUBSCRIPT: {
                Value index = pop();
                Value obj = pop();
                if (IS_LIST(obj)) {
                    ObjList* list = AS_LIST(obj);
                    if (!IS_NUMBER(index)) {
                        runtimeError("List index must be a number.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    int idx = (int)AS_NUMBER(index);
                    if (idx < 0) idx += list->items.count;
                    if (idx < 0 || idx >= list->items.count) {
                        if (raiseException(createExceptionValue("IndexError", "List index out of range."))) {
                            HANDLE_RE();
                        }
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    for (int i = idx; i < list->items.count - 1; i++) {
                        list->items.values[i] = list->items.values[i + 1];
                    }
                    list->items.count--;
                } else if (IS_DICT(obj)) {
                    ObjDict* dict = AS_DICT(obj);
                    tableDelete(&dict->table, index);
                } else {
                    runtimeError("Object does not support item deletion.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_SHIFT_LEFT:  BITWISE_OP(<<); break;
            case OP_SHIFT_RIGHT: BITWISE_OP(>>); break;
            case OP_NOT:
                push(BOOL_VAL(isFalsey(pop())));
                break;
            case OP_NEGATE:
                if (!IS_NUMBER(peek(0))) {
                    runtimeError("Operand must be a number.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                push(NUMBER_VAL(-AS_NUMBER(pop())));
                break;
            case OP_PRINT: {
                printValue(pop());
                printf("\n");
                break;
            }
            case OP_JUMP: {
                uint16_t offset = READ_SHORT();
                frame->ip += offset;
                break;
            }
            case OP_JUMP_IF_FALSE: {
                uint16_t offset = READ_SHORT();
                if (isFalsey(peek(0))) frame->ip += offset;
                break;
            }
            case OP_LOOP: {
                uint16_t offset = READ_SHORT();
                frame->ip -= offset;
                break;
            }
            case OP_CLASS:
                push(OBJ_VAL(newClass(AS_STRING(READ_CONSTANT()))));
                break;
            case OP_METHOD:
                defineMethod(AS_STRING(READ_CONSTANT()));
                break;
            case OP_CALL: {
                int argCount = READ_BYTE();
                if (!callValue(peek(argCount), argCount)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_CALL_STAR: {
                int argCount = READ_BYTE();
                ObjTuple* starIndexes = AS_TUPLE(READ_CONSTANT());
                if (!callValueExpanded(peek(argCount), argCount, starIndexes)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_CALL_KW: {
                int posCount = READ_BYTE();
                int kwCount = READ_BYTE();
                if (!callValueKeyword(peek(posCount + 2 * kwCount), posCount, kwCount)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_CALL_EX: {
                int posCount = READ_BYTE();
                int kwSourceCount = READ_BYTE();
                int explicitKwCount = READ_BYTE();
                ObjTuple* posStarIndexes = AS_TUPLE(READ_CONSTANT());
                ObjTuple* kwStarIndexes = AS_TUPLE(READ_CONSTANT());
                if (!callValueExpandedKeyword(peek(posCount + kwSourceCount + explicitKwCount),
                                              posCount, kwSourceCount, explicitKwCount,
                                              posStarIndexes, kwStarIndexes)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_INVOKE: {
                ObjString* method = AS_STRING(READ_CONSTANT());
                int argCount = READ_BYTE();
                if (!invoke(method, argCount)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_INVOKE_STAR: {
                ObjString* method = AS_STRING(READ_CONSTANT());
                int argCount = READ_BYTE();
                ObjTuple* starIndexes = AS_TUPLE(READ_CONSTANT());
                if (!invokeExpanded(method, argCount, starIndexes)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_INVOKE_KW: {
                ObjString* method = AS_STRING(READ_CONSTANT());
                int posCount = READ_BYTE();
                int kwCount = READ_BYTE();
                if (!invokeKeyword(method, posCount, kwCount)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_INVOKE_EX: {
                ObjString* method = AS_STRING(READ_CONSTANT());
                int posCount = READ_BYTE();
                int kwSourceCount = READ_BYTE();
                int explicitKwCount = READ_BYTE();
                ObjTuple* posStarIndexes = AS_TUPLE(READ_CONSTANT());
                ObjTuple* kwStarIndexes = AS_TUPLE(READ_CONSTANT());
                if (!invokeExpandedKeyword(method, posCount, kwSourceCount, explicitKwCount,
                                           posStarIndexes, kwStarIndexes)) {
                    HANDLE_RE();
                }
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
            case OP_GET_PROPERTY: {
                if (!IS_INSTANCE(peek(0))) {
                    if (raiseException(createExceptionValue("AttributeError", "Only instances have properties."))) {
                        HANDLE_RE();
                    }
                    return INTERPRET_RUNTIME_ERROR;
                }
                ObjInstance* instance = AS_INSTANCE(peek(0));
                ObjString* name = AS_STRING(READ_CONSTANT());
                Value value;
                if (tableGet(&instance->fields, OBJ_VAL(name), &value)) {
                    pop(); // Instance
                    push(value);
                    break;
                }
                if (tryImportModuleProperty(instance, name, &value)) {
                    pop();
                    push(value);
                    break;
                }
                if (!bindMethod(instance->klass, name)) {
                    HANDLE_RE();
                }
                break;
            }
            case OP_SET_PROPERTY: {
                if (!IS_INSTANCE(peek(1))) {
                    runtimeError("Only instances have fields.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                ObjInstance* instance = AS_INSTANCE(peek(1));
                tableSet(&instance->fields, OBJ_VAL(AS_STRING(READ_CONSTANT())), peek(0));
                Value value = pop();
                pop(); // Instance
                push(value);
                break;
            }
            case OP_IMPORT_MODULE: {
                ObjString* moduleName = AS_STRING(READ_CONSTANT());
                Value moduleValue = importModuleValue(moduleName);
                if (IS_NIL(moduleValue)) {
                    HANDLE_RE();
                }
                push(moduleValue);
                break;
            }
            case OP_LIST_APPEND: {
                uint8_t distance = READ_BYTE();
                Value value = peek(0);
                ObjList* list = AS_LIST(peek(distance));
                writeValueArray(&list->items, value);
                pop();
                break;
            }
            case OP_SET_DEFAULTS: {
                uint8_t count = READ_BYTE();
                ObjFunction* function = IS_CLOSURE(peek(0))
                    ? AS_CLOSURE(peek(0))->function
                    : AS_FUNCTION(peek(0));
                for (int i = 0; i < count; i++) {
                    writeValueArray(&function->defaults, peek(count - i));
                }
                function->defaultsCount = count;
                // Pop defaults + function
                Value funcVal = pop();
                for (int i = 0; i < count; i++) pop();
                push(funcVal); 
                break;
            }
            case OP_SETUP_EXCEPT: {
                uint16_t offset = READ_SHORT();
                uint8_t expectedTypeIndex = READ_BYTE();
                if (vm.handlerCount < HANDLERS_MAX) {
                    Handler* handler = &vm.handlers[vm.handlerCount++];
                    handler->type = 0;
                    handler->handlerIP = frame->ip + offset;
                    handler->stackTop = vm.stackTop;
                    handler->frameCount = vm.frameCount;
                    handler->expectedType = NIL_VAL;
                    if (expectedTypeIndex != UINT8_MAX) {
                        Value typeName = frame->function->chunk.constants.values[expectedTypeIndex];
                        Value resolvedType;
                        if (!tableGet(currentGlobalEnv(frame)->table, typeName, &resolvedType)) {
                            if (raiseException(createExceptionValue("RuntimeError", "Unknown exception type."))) {
                                HANDLE_RE();
                            }
                            return INTERPRET_RUNTIME_ERROR;
                        }
                        handler->expectedType = resolvedType;
                    }
                } else {
                    RUNTIME_ERROR("Too many nested try blocks.");
                }
                break;
            }
            case OP_POP_EXCEPT: {
                if (vm.handlerCount > 0) {
                    vm.handlerCount--;
                }
                break;
            }
            case OP_RAISE: {
                Value exception = pop();
                if (IS_STRING(exception)) {
                    if (raiseException(createExceptionValue("Exception", AS_STRING(exception)->chars))) {
                        HANDLE_RE();
                    }
                } else if (IS_CLASS(exception)) {
                    if (raiseException(OBJ_VAL(newExceptionInstance(AS_CLASS(exception), NULL)))) {
                        HANDLE_RE();
                    }
                } else if (IS_INSTANCE(exception)) {
                    if (raiseException(exception)) {
                        HANDLE_RE();
                    }
                } else {
                    char buffer[256];
                    if (!valueToExceptionMessage(exception, buffer, sizeof(buffer))) {
                        snprintf(buffer, sizeof(buffer), "Exception raised.");
                    }
                    if (raiseException(createExceptionValue("Exception", buffer))) {
                        HANDLE_RE();
                    }
                }
                return INTERPRET_RUNTIME_ERROR;
                break;
            }
            case OP_RETURN: {
                Value result = pop();
                vm.frameCount--;
                if (vm.frameCount < initialFrameCount) {
                    push(result);
                    return INTERPRET_OK;
                }

                if (vm.frameCount == 0) {
                    pop(); // Pop script entry
                    return INTERPRET_OK;
                }

                vm.stackTop = frame->slots;
                push(result);
                frame = &vm.frames[vm.frameCount - 1];
                break;
            }
        }
    }

#undef READ_BYTE
#undef READ_CONSTANT
#undef READ_SHORT
#undef BINARY_OP
}

static InterpretResult interpretInGlobals(const char* source, const char* filename, ObjEnvironment* env) {
    Value* stackStart = vm.stackTop;
    vm.gcPauseDepth++;
    ObjFunction* function = compile(source, filename);
    if (function == NULL) {
        vm.gcPauseDepth--;
        return INTERPRET_COMPILE_ERROR;
    }

    ObjClosure* closure = newClosure(function, env);
    push(OBJ_VAL(closure));
    vm.gcPauseDepth--;
    call(closure, 0);

    InterpretResult result = run();
    if (result == INTERPRET_OK) {
        vm.stackTop = stackStart;
    }
    return result;
}

InterpretResult interpret(const char* source, const char* filename) {
    return interpretInGlobals(source, filename, vm.globalEnv);
}
