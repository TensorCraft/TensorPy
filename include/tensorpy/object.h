#ifndef TENSORPY_OBJECT_H
#define TENSORPY_OBJECT_H

#include "common.h"
#include "metal.h"
#include "value.h"
#include "table.h"
#include "chunk.h"

typedef enum {
    OBJ_STRING,
    OBJ_FUNCTION,
    OBJ_CLOSURE,
    OBJ_ENVIRONMENT,
    OBJ_NATIVE,
    OBJ_SET,
    OBJ_LIST,
    OBJ_DICT,
    OBJ_TUPLE,
    OBJ_BYTES,
    OBJ_INT,
    OBJ_CLASS,
    OBJ_INSTANCE,
    OBJ_BOUND_METHOD,
    OBJ_SLICE,
    OBJ_ITERATOR,
    OBJ_DEVICE,
    OBJ_DTYPE,
    OBJ_TENSOR,
} ObjType;

typedef enum {
    TP_DEVICE_CPU = 0,
    TP_DEVICE_METAL = 1,
} TPDeviceKind;

typedef enum {
    TP_DTYPE_FLOAT32 = 0,
} TPDTypeKind;

typedef enum {
    TP_AUTOGRAD_NONE = 0,
    TP_AUTOGRAD_ADD,
    TP_AUTOGRAD_SUB,
    TP_AUTOGRAD_MUL,
    TP_AUTOGRAD_DIV,
    TP_AUTOGRAD_RESHAPE,
    TP_AUTOGRAD_RELU,
    TP_AUTOGRAD_TANH,
    TP_AUTOGRAD_SIGMOID,
    TP_AUTOGRAD_GELU,
    TP_AUTOGRAD_MATMUL,
    TP_AUTOGRAD_CONV2D,
    TP_AUTOGRAD_MSE_LOSS,
} TPAutogradOp;

struct Obj {
    ObjType type;
    bool isMarked;
    struct Obj* next;
};

struct ObjString {
    Obj obj;
    int length;
    char* chars;
    uint32_t hash;
};

typedef struct {
    Obj obj;
    int arity;
    int defaultsCount;
    int maxSlots;
    bool hasVarargs;
    ValueArray defaults;
    ValueArray paramNames; // For keyword arguments
    ValueArray localNames; // Slot names for closure capture
    Chunk chunk;
    ObjString* name;
} ObjFunction;

typedef struct ObjEnvironment {
    Obj obj;
    Table values;
    Table* table;
    bool ownsTable;
    struct ObjEnvironment* parent;
} ObjEnvironment;

typedef struct {
    Obj obj;
    ObjFunction* function;
    ObjEnvironment* env;
} ObjClosure;

typedef struct {
    Obj obj;
    Table table;
} ObjSet;

typedef struct {
    Obj obj;
    ValueArray items;
} ObjList;

typedef struct {
    Obj obj;
    Table table;
} ObjDict;

typedef struct {
    Obj obj;
    ValueArray items;
} ObjTuple;

typedef struct ObjClass {
    Obj obj;
    ObjString* name;
    struct ObjClass* superClass;
    Table methods;
} ObjClass;

typedef struct {
    Obj obj;
    Value start;
    Value stop;
    Value step;
} ObjSlice;

typedef struct {
    Obj obj;
    Value iterable;
    int index;
} ObjIterator;

typedef struct {
    Obj obj;
    ObjClass* klass;
    Table fields;
} ObjInstance;

typedef struct {
    Obj obj;
    Value receiver;
    Value method;
} ObjBoundMethod;

typedef struct {
    Obj obj;
    int length;
    uint8_t* bytes;
} ObjBytes;

typedef struct {
    Obj obj;
    bool negative;
    int length;
    uint32_t* digits;
} ObjInt;

typedef struct {
    Obj obj;
    TPDeviceKind kind;
    ObjString* name;
} ObjDevice;

typedef struct {
    Obj obj;
    TPDTypeKind kind;
    ObjString* name;
} ObjDType;

typedef struct ObjTensor {
    Obj obj;
    int rank;
    int size;
    int* shape;
    ObjDType* dtype;
    ObjDevice* device;
    bool contiguous;
    float* data;
    TPMetalBuffer* metalBuffer;
    bool ownsData;
    bool ownsMetalBuffer;
    bool cpuDirty;
    bool metalDirty;
    bool requiresGrad;
    struct ObjTensor* grad;
    struct ObjTensor* parentA;
    struct ObjTensor* parentB;
    TPAutogradOp gradOp;
    float gradAux;
} ObjTensor;

// Native function type for built-ins
typedef Value (*NativeFn)(int argCount, Value* args);
typedef void (*NativeFreeFn)(void* userData);

typedef struct {
    Obj obj;
    NativeFn function;
    void* userData;
    NativeFreeFn freeUserData;
} ObjNative;

static inline bool isObjType(Value value, ObjType type) {
    return IS_OBJ(value) && AS_OBJ(value)->type == type;
}

#define OBJ_TYPE(value)     (AS_OBJ(value)->type)

#define IS_STRING(value)    isObjType(value, OBJ_STRING)
#define IS_FUNCTION(value)  isObjType(value, OBJ_FUNCTION)
#define IS_CLOSURE(value)   isObjType(value, OBJ_CLOSURE)
#define IS_ENVIRONMENT(value) isObjType(value, OBJ_ENVIRONMENT)
#define IS_NATIVE(value)    isObjType(value, OBJ_NATIVE)
#define IS_SET(value)       isObjType(value, OBJ_SET)
#define IS_LIST(value)      isObjType(value, OBJ_LIST)
#define IS_DICT(value)      isObjType(value, OBJ_DICT)
#define IS_TUPLE(value)     isObjType(value, OBJ_TUPLE)
#define IS_BYTES(value)     isObjType(value, OBJ_BYTES)
#define IS_INT(value)       isObjType(value, OBJ_INT)
#define IS_CLASS(value)     isObjType(value, OBJ_CLASS)
#define IS_INSTANCE(value)  isObjType(value, OBJ_INSTANCE)
#define IS_BOUND_METHOD(value) isObjType(value, OBJ_BOUND_METHOD)
#define IS_SLICE(value)     isObjType(value, OBJ_SLICE)
#define IS_ITERATOR(value)  isObjType(value, OBJ_ITERATOR)
#define IS_DEVICE(value)    isObjType(value, OBJ_DEVICE)
#define IS_DTYPE(value)     isObjType(value, OBJ_DTYPE)
#define IS_TENSOR(value)    isObjType(value, OBJ_TENSOR)

#define AS_STRING(value)    ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)   (((ObjString*)AS_OBJ(value))->chars)
#define AS_FUNCTION(value)  ((ObjFunction*)AS_OBJ(value))
#define AS_CLOSURE(value)   ((ObjClosure*)AS_OBJ(value))
#define AS_ENVIRONMENT(value) ((ObjEnvironment*)AS_OBJ(value))
#define AS_NATIVE(value)    (((ObjNative*)AS_OBJ(value))->function)
#define AS_SET(value)       ((ObjSet*)AS_OBJ(value))
#define AS_LIST(value)      ((ObjList*)AS_OBJ(value))
#define AS_DICT(value)      ((ObjDict*)AS_OBJ(value))
#define AS_TUPLE(value)     ((ObjTuple*)AS_OBJ(value))
#define AS_BYTES(value)     ((ObjBytes*)AS_OBJ(value))
#define AS_INT(value)       ((ObjInt*)AS_OBJ(value))
#define AS_CLASS(value)     ((ObjClass*)AS_OBJ(value))
#define AS_INSTANCE(value)  ((ObjInstance*)AS_OBJ(value))
#define AS_BOUND_METHOD(value) ((ObjBoundMethod*)AS_OBJ(value))
#define AS_SLICE(value)     ((ObjSlice*)AS_OBJ(value))
#define AS_ITERATOR(value)  ((ObjIterator*)AS_OBJ(value))
#define AS_DEVICE(value)    ((ObjDevice*)AS_OBJ(value))
#define AS_DTYPE(value)     ((ObjDType*)AS_OBJ(value))
#define AS_TENSOR(value)    ((ObjTensor*)AS_OBJ(value))

ObjString* copyString(const char* chars, int length);
ObjString* takeString(char* chars, int length);
ObjBytes* newBytes(int length, const uint8_t* source);
ObjInt* newIntFromInt64(int64_t value);
ObjInt* newIntFromString(const char* chars, int length);
ObjInt* newIntCopy(const ObjInt* source);
uint32_t intHash(const ObjInt* value);
double intToDouble(const ObjInt* value);
bool intIsZero(const ObjInt* value);
bool intToInt64Exact(const ObjInt* value, int64_t* out);
int intCompare(const ObjInt* a, const ObjInt* b);
ObjInt* intAdd(const ObjInt* a, const ObjInt* b);
ObjInt* intSub(const ObjInt* a, const ObjInt* b);
ObjInt* intMul(const ObjInt* a, const ObjInt* b);
bool intDivMod(const ObjInt* a, const ObjInt* b, ObjInt** outQuotient, ObjInt** outRemainder);
void intToString(const ObjInt* value, char** outChars, int* outLength);
ObjClass* newClass(ObjString* name);
ObjInstance* newInstance(ObjClass* klass);
ObjBoundMethod* newBoundMethod(Value receiver, Value method);
ObjSlice* newSlice(Value start, Value stop, Value step);
ObjIterator* newIterator(Value iterable);
ObjList* newList();
ObjFunction* newFunction();
ObjClosure* newClosure(ObjFunction* function, ObjEnvironment* env);
ObjEnvironment* newEnvironment(ObjEnvironment* parent);
ObjEnvironment* newEnvironmentWithTable(ObjEnvironment* parent, Table* table);
ObjNative* newNative(NativeFn function);
ObjNative* newNativeWithUserData(NativeFn function, void* userData);
ObjNative* newNativeWithFinalizer(NativeFn function, void* userData, NativeFreeFn freeUserData);
ObjSet* newSet();
ObjList* newList();
ObjDict* newDict();
ObjTuple* newTuple();
ObjDevice* newDevice(ObjString* name, TPDeviceKind kind);
ObjDType* newDType(ObjString* name, TPDTypeKind kind);
ObjTensor* newTensor(int rank,
                     const int* shape,
                     ObjDType* dtype,
                     ObjDevice* device,
                     const float* source);
ObjTensor* newTensorView(int rank,
                         const int* shape,
                         ObjDType* dtype,
                         ObjDevice* device,
                         float* data,
                         TPMetalBuffer* metalBuffer);
int tensorElementCount(int rank, const int* shape);
bool getNativeObjectAttribute(Value object, ObjString* name, Value* result);
void appendNativeObjectDirEntries(ObjList* out, Value object);
void freeObjects(void);
int sweepUnmarkedObjects(void);
int countObjects(void);

void printObject(Value value);

#endif // TENSORPY_OBJECT_H
