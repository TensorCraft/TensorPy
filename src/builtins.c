#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "tensorpy/common.h"
#include "tensorpy/memory.h"
#include "tensorpy/object.h"
#include "tensorpy/platform.h"
#include "tensorpy/value.h"
#include "tensorpy/vm.h"
#include "tensorpy/builtins.h"

static bool syncTensorToMetal(ObjTensor* tensor);
static bool syncTensorFromMetal(ObjTensor* tensor);
static ObjTensor* newScalarTensor(float value, ObjDevice* device);
static bool setAutogradBinary(ObjTensor* out, ObjTensor* a, ObjTensor* b, TPAutogradOp op);
static bool setAutogradUnary(ObjTensor* out, ObjTensor* input, TPAutogradOp op);
static Value tensorConv2d(Value inputValue, Value weightValue);
static Value tensorToListValue(ObjTensor* tensor);

static bool valueMatchesTypeName(Value value, ObjString* name) {
    const char* typeName = name->chars;

    if (IS_BOOL(value)) return strcmp(typeName, "bool") == 0;
    if (IS_NIL(value)) return strcmp(typeName, "NoneType") == 0;
    if (IS_NUMBER(value)) return strcmp(typeName, "float") == 0;
    if (!IS_OBJ(value)) return false;

    switch (OBJ_TYPE(value)) {
        case OBJ_STRING: return strcmp(typeName, "str") == 0;
        case OBJ_NATIVE: return strcmp(typeName, "builtin_function_or_method") == 0;
        case OBJ_CLOSURE: return strcmp(typeName, "function") == 0;
        case OBJ_SET: return strcmp(typeName, "set") == 0;
        case OBJ_LIST: return strcmp(typeName, "list") == 0;
        case OBJ_DICT: return strcmp(typeName, "dict") == 0;
        case OBJ_TUPLE: return strcmp(typeName, "tuple") == 0;
        case OBJ_BYTES: return strcmp(typeName, "bytes") == 0;
        case OBJ_DEVICE: return strcmp(typeName, "device") == 0;
        case OBJ_DTYPE: return strcmp(typeName, "dtype") == 0;
        case OBJ_TENSOR: return strcmp(typeName, "tensor") == 0;
        case OBJ_CLASS: return strcmp(typeName, "type") == 0;
        case OBJ_INSTANCE: return strcmp(AS_INSTANCE(value)->klass->name->chars, typeName) == 0;
        default: return strcmp(typeName, "object") == 0;
    }
}

static bool classMatchesExpected(ObjClass* klass, ObjClass* expected) {
    while (klass != NULL) {
        if (klass == expected) {
            return true;
        }
        klass = klass->superClass;
    }
    return false;
}

static bool getAttributeValue(Value object, ObjString* name, Value* result) {
    if (getNativeObjectAttribute(object, name, result)) {
        return true;
    }

    if (IS_INSTANCE(object)) {
        ObjInstance* instance = AS_INSTANCE(object);
        if (tableGet(&instance->fields, OBJ_VAL(name), result)) {
            return true;
        }

        Value method;
        if (tableGet(&instance->klass->methods, OBJ_VAL(name), &method)) {
            if (IS_FUNCTION(method) || IS_CLOSURE(method)) {
                *result = OBJ_VAL(newBoundMethod(object, method));
            } else {
                *result = method;
            }
            return true;
        }
        return false;
    }

    if (IS_CLASS(object)) {
        return tableGet(&AS_CLASS(object)->methods, OBJ_VAL(name), result);
    }

    return false;
}

static int sequenceLength(Value value) {
    if (IS_LIST(value)) {
        return AS_LIST(value)->items.count;
    }
    if (IS_TUPLE(value)) {
        return AS_TUPLE(value)->items.count;
    }
    return -1;
}

static Value sequenceValueAt(Value value, int index) {
    if (IS_LIST(value)) {
        return AS_LIST(value)->items.values[index];
    }
    return AS_TUPLE(value)->items.values[index];
}

static ObjDevice* resolveDeviceArg(Value value) {
    if (IS_NIL(value)) {
        return vm.cpuDevice;
    }
    if (IS_DEVICE(value)) {
        return AS_DEVICE(value);
    }
    if (IS_STRING(value)) {
        if (strcmp(AS_CSTRING(value), "cpu") == 0) {
            return vm.cpuDevice;
        }
        if (strcmp(AS_CSTRING(value), "metal") == 0 ||
            strcmp(AS_CSTRING(value), "metal:0") == 0) {
            return vm.metalDevice;
        }
    }
    return NULL;
}

static ObjDType* resolveDTypeArg(Value value) {
    if (IS_NIL(value)) {
        return vm.float32DType;
    }
    if (IS_DTYPE(value)) {
        return AS_DTYPE(value);
    }
    if (IS_STRING(value) && strcmp(AS_CSTRING(value), "float32") == 0) {
        return vm.float32DType;
    }
    return NULL;
}

static bool parseShapeValue(Value value, int** outShape, int* outRank) {
    int count;
    int i;
    int* shape;

    if (IS_NUMBER(value)) {
        shape = (int*)tpMemAlloc(sizeof(int));
        if (shape == NULL) {
            return false;
        }
        shape[0] = (int)AS_NUMBER(value);
        if (shape[0] < 0) {
            tpMemFree(shape);
            return false;
        }
        *outShape = shape;
        *outRank = 1;
        return true;
    }

    count = sequenceLength(value);
    if (count < 0) {
        return false;
    }

    shape = count > 0 ? (int*)tpMemAlloc(sizeof(int) * (size_t)count) : NULL;
    if (count > 0 && shape == NULL) {
        return false;
    }

    for (i = 0; i < count; i++) {
        Value item = sequenceValueAt(value, i);
        if (!IS_NUMBER(item)) {
            tpMemFree(shape);
            return false;
        }
        shape[i] = (int)AS_NUMBER(item);
        if (shape[i] < 0) {
            tpMemFree(shape);
            return false;
        }
    }

    *outShape = shape;
    *outRank = count;
    return true;
}

static bool inferTensorShape(Value value, int** outShape, int* outRank) {
    int count;
    int* childShape = NULL;
    int childRank = 0;
    int* shape;
    int i;

    if (IS_NUMBER(value)) {
        *outShape = NULL;
        *outRank = 0;
        return true;
    }

    count = sequenceLength(value);
    if (count < 0) {
        return false;
    }

    if (count == 0) {
        shape = (int*)tpMemAlloc(sizeof(int));
        if (shape == NULL) {
            return false;
        }
        shape[0] = 0;
        *outShape = shape;
        *outRank = 1;
        return true;
    }

    if (!inferTensorShape(sequenceValueAt(value, 0), &childShape, &childRank)) {
        return false;
    }

    for (i = 1; i < count; i++) {
        int* otherShape = NULL;
        int otherRank = 0;
        if (!inferTensorShape(sequenceValueAt(value, i), &otherShape, &otherRank)) {
            tpMemFree(childShape);
            return false;
        }
        if (otherRank != childRank ||
            (childRank > 0 && memcmp(otherShape, childShape, sizeof(int) * (size_t)childRank) != 0)) {
            tpMemFree(otherShape);
            tpMemFree(childShape);
            return false;
        }
        tpMemFree(otherShape);
    }

    shape = (int*)tpMemAlloc(sizeof(int) * (size_t)(childRank + 1));
    if (shape == NULL) {
        tpMemFree(childShape);
        return false;
    }
    shape[0] = count;
    if (childRank > 0) {
        memcpy(shape + 1, childShape, sizeof(int) * (size_t)childRank);
    }
    tpMemFree(childShape);
    *outShape = shape;
    *outRank = childRank + 1;
    return true;
}

static bool flattenTensorValues(Value value, float* out, int* offset) {
    int count;
    int i;

    if (IS_NUMBER(value)) {
        out[*offset] = (float)AS_NUMBER(value);
        (*offset)++;
        return true;
    }

    count = sequenceLength(value);
    if (count < 0) {
        return false;
    }

    for (i = 0; i < count; i++) {
        if (!flattenTensorValues(sequenceValueAt(value, i), out, offset)) {
            return false;
        }
    }
    return true;
}

static Value tensorListFromOffset(const ObjTensor* tensor, int dim, int* offset) {
    if (dim >= tensor->rank) {
        return NUMBER_VAL((double)tensor->data[(*offset)++]);
    }

    ObjList* list = newList();
    int count = tensor->shape[dim];
    int i;
    for (i = 0; i < count; i++) {
        writeValueArray(&list->items, tensorListFromOffset(tensor, dim + 1, offset));
    }
    return OBJ_VAL(list);
}

static Value tensorToListValue(ObjTensor* tensor) {
    int offset = 0;
    if (tensor->rank == 0) {
        return NUMBER_VAL((double)tensor->data[0]);
    }
    return tensorListFromOffset(tensor, 0, &offset);
}

static ObjTensor* createTensorFromData(Value value, ObjDType* dtype, ObjDevice* device) {
    int* shape = NULL;
    int rank = 0;
    int size;
    float* data = NULL;
    int offset = 0;
    ObjTensor* tensor;

    if (device == NULL || dtype == NULL) {
        return NULL;
    }
    if (device->kind == TP_DEVICE_METAL && !tpMetalBackendIsAvailable(vm.metalBackend)) {
        vmRaiseExceptionMessage("RuntimeError", "Metal backend is unavailable.");
        return NULL;
    }
    if (!inferTensorShape(value, &shape, &rank)) {
        vmRaiseExceptionMessage("TypeError", "tensor() expects a number or a rectangular nested list/tuple of numbers.");
        return NULL;
    }

    size = tensorElementCount(rank, shape);
    if (size < 0) {
        free(shape);
        vmRaiseExceptionMessage("ValueError", "Invalid tensor shape.");
        return NULL;
    }

    if (size > 0) {
        data = (float*)malloc(sizeof(float) * (size_t)size);
        if (data == NULL) {
            free(shape);
            vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
            return NULL;
        }
    }

    if (!flattenTensorValues(value, data, &offset)) {
        free(shape);
        free(data);
        vmRaiseExceptionMessage("TypeError", "tensor() data contains unsupported values.");
        return NULL;
    }

    tensor = newTensor(rank, shape, dtype, device, data);
    free(shape);
    free(data);
    if (tensor == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
    }
    return tensor;
}

static ObjTensor* createTensorFilled(const int* shape,
                                     int rank,
                                     ObjDType* dtype,
                                     ObjDevice* device,
                                     float fillValue) {
    ObjTensor* tensor;
    int i;

    if (device == NULL || dtype == NULL) {
        return NULL;
    }
    if (device->kind == TP_DEVICE_METAL && !tpMetalBackendIsAvailable(vm.metalBackend)) {
        vmRaiseExceptionMessage("RuntimeError", "Metal backend is unavailable.");
        return NULL;
    }

    tensor = newTensor(rank, shape, dtype, device, NULL);
    if (tensor == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NULL;
    }
    for (i = 0; i < tensor->size; i++) {
        tensor->data[i] = fillValue;
    }
    syncTensorToMetal(tensor);
    return tensor;
}

typedef enum {
    TP_TENSOR_BINARY_ADD,
    TP_TENSOR_BINARY_SUB,
    TP_TENSOR_BINARY_MUL,
    TP_TENSOR_BINARY_DIV,
} TPTensorBinaryOp;

typedef enum {
    TP_TENSOR_UNARY_RELU,
    TP_TENSOR_UNARY_TANH,
    TP_TENSOR_UNARY_SIGMOID,
    TP_TENSOR_UNARY_GELU,
} TPTensorUnaryOp;

static bool tensorShapeEquals(const ObjTensor* a, const ObjTensor* b) {
    if (a->rank != b->rank) {
        return false;
    }
    if (a->rank == 0) {
        return true;
    }
    return memcmp(a->shape, b->shape, sizeof(int) * (size_t)a->rank) == 0;
}

static void fillStrides(const int* shape, int rank, int* outStrides) {
    int stride = 1;
    int i;
    for (i = rank - 1; i >= 0; i--) {
        outStrides[i] = stride;
        stride *= shape[i];
    }
}

static bool broadcastShapes(const ObjTensor* a,
                            const ObjTensor* b,
                            int** outShape,
                            int* outRank) {
    int rank = a->rank > b->rank ? a->rank : b->rank;
    int* shape;
    int i;

    shape = rank > 0 ? (int*)malloc(sizeof(int) * (size_t)rank) : NULL;
    if (rank > 0 && shape == NULL) {
        return false;
    }

    for (i = 0; i < rank; i++) {
        int aDimIndex = i - (rank - a->rank);
        int bDimIndex = i - (rank - b->rank);
        int aDim = aDimIndex >= 0 ? a->shape[aDimIndex] : 1;
        int bDim = bDimIndex >= 0 ? b->shape[bDimIndex] : 1;

        if (aDim != bDim && aDim != 1 && bDim != 1) {
            free(shape);
            return false;
        }
        shape[i] = aDim > bDim ? aDim : bDim;
    }

    *outShape = shape;
    *outRank = rank;
    return true;
}

static int broadcastOffsetForIndex(int linearIndex,
                                   const int* outShape,
                                   const int* outStrides,
                                   int outRank,
                                   const ObjTensor* input,
                                   const int* inputStrides) {
    int inputRank = input->rank;
    int inputOffset = 0;
    int rankOffset = outRank - inputRank;
    int dim;

    if (inputRank == 0) {
        return 0;
    }

    for (dim = 0; dim < outRank; dim++) {
        int coord = outShape[dim] == 0 ? 0 : (linearIndex / outStrides[dim]) % outShape[dim];
        int inputDim = dim - rankOffset;
        if (inputDim >= 0) {
            int clamped = input->shape[inputDim] == 1 ? 0 : coord;
            inputOffset += clamped * inputStrides[inputDim];
        }
    }
    return inputOffset;
}

static ObjTensor* createTensorLike(const ObjTensor* source) {
    return newTensor(source->rank, source->shape, source->dtype, source->device, NULL);
}

static ObjTensor* createTensorFromShape(int rank, const int* shape, ObjDevice* device) {
    return newTensor(rank, shape, vm.float32DType, device != NULL ? device : vm.cpuDevice, NULL);
}

static bool syncTensorToMetal(ObjTensor* tensor) {
    if (tensor == NULL || tensor->device->kind != TP_DEVICE_METAL || tensor->metalBuffer == NULL) {
        return true;
    }
    tensor->metalDirty = false;
    tensor->cpuDirty = false;
    return tpMetalBufferWrite(tensor->metalBuffer,
                              tensor->data,
                              sizeof(float) * (size_t)(tensor->size > 0 ? tensor->size : 1));
}

static bool syncTensorFromMetal(ObjTensor* tensor) {
    if (tensor == NULL || tensor->device->kind != TP_DEVICE_METAL || tensor->metalBuffer == NULL) {
        return true;
    }
    if (!tensor->cpuDirty) {
        return true;
    }
    tensor->cpuDirty = false;
    tensor->metalDirty = false;
    return tpMetalBufferRead(tensor->metalBuffer,
                             tensor->data,
                             sizeof(float) * (size_t)(tensor->size > 0 ? tensor->size : 1));
}

static bool canRunMetalElementwise(const ObjTensor* a, const ObjTensor* b) {
    return tpMetalBackendIsAvailable(vm.metalBackend) &&
           a->device->kind == TP_DEVICE_METAL &&
           b->device->kind == TP_DEVICE_METAL &&
           a->metalBuffer != NULL &&
           b->metalBuffer != NULL &&
           tensorShapeEquals(a, b);
}

static bool canRunMetalScalarElementwise(const ObjTensor* tensor) {
    return tpMetalBackendIsAvailable(vm.metalBackend) &&
           tensor->device->kind == TP_DEVICE_METAL &&
           tensor->metalBuffer != NULL;
}

static float applyBinaryScalar(float a, float b, TPTensorBinaryOp op) {
    switch (op) {
        case TP_TENSOR_BINARY_ADD: return a + b;
        case TP_TENSOR_BINARY_SUB: return a - b;
        case TP_TENSOR_BINARY_MUL: return a * b;
        case TP_TENSOR_BINARY_DIV: return a / b;
    }
    return 0.0f;
}

static float applyUnaryScalar(float x, TPTensorUnaryOp op) {
    switch (op) {
        case TP_TENSOR_UNARY_RELU:
            return x > 0.0f ? x : 0.0f;
        case TP_TENSOR_UNARY_TANH:
            return tanhf(x);
        case TP_TENSOR_UNARY_SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        case TP_TENSOR_UNARY_GELU: {
            float cubic = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * cubic);
            return 0.5f * x * (1.0f + tanhf(inner));
        }
    }
    return x;
}

static ObjTensor* tensorBinaryTensorOp(const ObjTensor* a,
                                       const ObjTensor* b,
                                       TPTensorBinaryOp op) {
    ObjTensor* out;
    int* outShape = NULL;
    int outRank = 0;
    int outSize;
    int* outStrides = NULL;
    int* aStrides = NULL;
    int* bStrides = NULL;
    int i;
    TPAutogradOp gradOp =
        op == TP_TENSOR_BINARY_ADD ? TP_AUTOGRAD_ADD :
        op == TP_TENSOR_BINARY_SUB ? TP_AUTOGRAD_SUB :
        op == TP_TENSOR_BINARY_MUL ? TP_AUTOGRAD_MUL :
        TP_AUTOGRAD_DIV;

    if (!broadcastShapes(a, b, &outShape, &outRank)) {
        vmRaiseExceptionMessage("ValueError", "Incompatible tensor shapes for broadcasting.");
        return NULL;
    }

    out = createTensorFromShape(outRank,
                                outShape,
                                (a->device->kind == TP_DEVICE_METAL || b->device->kind == TP_DEVICE_METAL)
                                    ? vm.metalDevice
                                    : vm.cpuDevice);
    if (out == NULL) {
        free(outShape);
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NULL;
    }

    outSize = out->size;
    if (canRunMetalElementwise(a, b) && tensorShapeEquals(a, out)) {
        bool ok = false;
        if (op == TP_TENSOR_BINARY_ADD) {
            ok = tpMetalAddF32(vm.metalBackend, a->metalBuffer, b->metalBuffer, out->metalBuffer, outSize);
        } else if (op == TP_TENSOR_BINARY_MUL) {
            ok = tpMetalMulF32(vm.metalBackend, a->metalBuffer, b->metalBuffer, out->metalBuffer, outSize);
        }
        if (ok) {
            out->cpuDirty = true;
            out->metalDirty = false;
            free(outShape);
            if (!setAutogradBinary(out, (ObjTensor*)a, (ObjTensor*)b, gradOp)) {
                return NULL;
            }
            return out;
        }
    }

    if (tensorShapeEquals(a, b) && tensorShapeEquals(a, out)) {
        if (!syncTensorFromMetal((ObjTensor*)a) || !syncTensorFromMetal((ObjTensor*)b)) {
            free(outShape);
            vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
            return NULL;
        }
        if (op == TP_TENSOR_BINARY_ADD) {
            tpComputeAddF32(&vm.compute, a->data, b->data, out->data, outSize, TP_COMPUTE_MODE_AUTO, NULL);
        } else if (op == TP_TENSOR_BINARY_MUL) {
            tpComputeMulF32(&vm.compute, a->data, b->data, out->data, outSize, TP_COMPUTE_MODE_AUTO, NULL);
        } else {
            for (i = 0; i < outSize; i++) {
                out->data[i] = applyBinaryScalar(a->data[i], b->data[i], op);
            }
        }
        syncTensorToMetal(out);
        free(outShape);
        if (!setAutogradBinary(out, (ObjTensor*)a, (ObjTensor*)b, gradOp)) {
            return NULL;
        }
        return out;
    }

    outStrides = outRank > 0 ? (int*)malloc(sizeof(int) * (size_t)outRank) : NULL;
    aStrides = a->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)a->rank) : NULL;
    bStrides = b->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)b->rank) : NULL;
    if ((outRank > 0 && outStrides == NULL) ||
        (a->rank > 0 && aStrides == NULL) ||
        (b->rank > 0 && bStrides == NULL)) {
        free(outStrides);
        free(aStrides);
        free(bStrides);
        free(outShape);
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while broadcasting tensor.");
        return NULL;
    }

    if (outRank > 0) {
        fillStrides(outShape, outRank, outStrides);
    }
    if (!syncTensorFromMetal((ObjTensor*)a) || !syncTensorFromMetal((ObjTensor*)b)) {
        free(outStrides);
        free(aStrides);
        free(bStrides);
        free(outShape);
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NULL;
    }
    if (a->rank > 0) {
        fillStrides(a->shape, a->rank, aStrides);
    }
    if (b->rank > 0) {
        fillStrides(b->shape, b->rank, bStrides);
    }
    for (i = 0; i < outSize; i++) {
        int aOffset = broadcastOffsetForIndex(i, outShape, outStrides, outRank, a, aStrides);
        int bOffset = broadcastOffsetForIndex(i, outShape, outStrides, outRank, b, bStrides);
        out->data[i] = applyBinaryScalar(a->data[aOffset], b->data[bOffset], op);
    }

    syncTensorToMetal(out);
    free(outStrides);
    free(aStrides);
    free(bStrides);
    free(outShape);
    if (!setAutogradBinary(out, (ObjTensor*)a, (ObjTensor*)b, gradOp)) {
        return NULL;
    }
    return out;
}

static ObjTensor* tensorBinaryScalarOp(const ObjTensor* tensor,
                                       float scalar,
                                       TPTensorBinaryOp op,
                                       bool scalarOnLeft) {
    ObjTensor* out = createTensorLike(tensor);
    ObjTensor* scalarTensor = NULL;
    int i;

    if (out == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NULL;
    }

    if ((op == TP_TENSOR_BINARY_ADD || op == TP_TENSOR_BINARY_MUL) &&
        canRunMetalScalarElementwise(tensor) &&
        !scalarOnLeft) {
        bool ok = op == TP_TENSOR_BINARY_ADD
            ? tpMetalAddScalarF32(vm.metalBackend, tensor->metalBuffer, scalar, out->metalBuffer, tensor->size)
            : tpMetalMulScalarF32(vm.metalBackend, tensor->metalBuffer, scalar, out->metalBuffer, tensor->size);
        if (ok) {
            out->cpuDirty = true;
            out->metalDirty = false;
        } else {
            vmRaiseExceptionMessage("RuntimeError", "Metal scalar kernel failed.");
            return NULL;
        }
    } else {
    if (!syncTensorFromMetal((ObjTensor*)tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NULL;
    }

    if (!scalarOnLeft && op == TP_TENSOR_BINARY_ADD) {
        tpComputeFillF32(&vm.compute, out->data, tensor->size, scalar, TP_COMPUTE_MODE_AUTO, NULL);
        tpComputeAddF32(&vm.compute, tensor->data, out->data, out->data, tensor->size, TP_COMPUTE_MODE_AUTO, NULL);
        syncTensorToMetal(out);
    } else if (!scalarOnLeft && op == TP_TENSOR_BINARY_MUL) {
        tpComputeFillF32(&vm.compute, out->data, tensor->size, scalar, TP_COMPUTE_MODE_AUTO, NULL);
        tpComputeMulF32(&vm.compute, tensor->data, out->data, out->data, tensor->size, TP_COMPUTE_MODE_AUTO, NULL);
        syncTensorToMetal(out);
    } else {
        for (i = 0; i < tensor->size; i++) {
            float lhs = scalarOnLeft ? scalar : tensor->data[i];
            float rhs = scalarOnLeft ? tensor->data[i] : scalar;
            out->data[i] = applyBinaryScalar(lhs, rhs, op);
        }
        syncTensorToMetal(out);
    }
    }
    scalarTensor = newScalarTensor(scalar, tensor->device);
    if (scalarTensor == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NULL;
    }
    if (!setAutogradBinary(out,
                           scalarOnLeft ? scalarTensor : (ObjTensor*)tensor,
                           scalarOnLeft ? (ObjTensor*)tensor : scalarTensor,
                           op == TP_TENSOR_BINARY_ADD ? TP_AUTOGRAD_ADD :
                           op == TP_TENSOR_BINARY_SUB ? TP_AUTOGRAD_SUB :
                           op == TP_TENSOR_BINARY_MUL ? TP_AUTOGRAD_MUL :
                           TP_AUTOGRAD_DIV)) {
        return NULL;
    }
    return out;
}

static Value mlBinaryOp(Value left, Value right, TPTensorBinaryOp op) {
    if (IS_NUMBER(left) && IS_NUMBER(right)) {
        return NUMBER_VAL((double)applyBinaryScalar((float)AS_NUMBER(left), (float)AS_NUMBER(right), op));
    }
    if (IS_TENSOR(left) && IS_TENSOR(right)) {
        ObjTensor* out = tensorBinaryTensorOp(AS_TENSOR(left), AS_TENSOR(right), op);
        return out != NULL ? OBJ_VAL(out) : NIL_VAL;
    }
    if (IS_TENSOR(left) && IS_NUMBER(right)) {
        ObjTensor* out = tensorBinaryScalarOp(AS_TENSOR(left), (float)AS_NUMBER(right), op, false);
        return out != NULL ? OBJ_VAL(out) : NIL_VAL;
    }
    if (IS_NUMBER(left) && IS_TENSOR(right)) {
        ObjTensor* out = tensorBinaryScalarOp(AS_TENSOR(right), (float)AS_NUMBER(left), op, true);
        return out != NULL ? OBJ_VAL(out) : NIL_VAL;
    }
    vmRaiseExceptionMessage("TypeError", "Expected tensor or number operands.");
    return NIL_VAL;
}

static ObjTensor* tensorUnaryOp(const ObjTensor* tensor, TPTensorUnaryOp op) {
    ObjTensor* out = createTensorLike(tensor);
    int i;

    if (out == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NULL;
    }
    if (!syncTensorFromMetal((ObjTensor*)tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NULL;
    }
    for (i = 0; i < tensor->size; i++) {
        out->data[i] = applyUnaryScalar(tensor->data[i], op);
    }
    syncTensorToMetal(out);
    if (!setAutogradUnary(out,
                          (ObjTensor*)tensor,
                          op == TP_TENSOR_UNARY_RELU ? TP_AUTOGRAD_RELU :
                          op == TP_TENSOR_UNARY_TANH ? TP_AUTOGRAD_TANH :
                          op == TP_TENSOR_UNARY_SIGMOID ? TP_AUTOGRAD_SIGMOID :
                          TP_AUTOGRAD_GELU)) {
        return NULL;
    }
    return out;
}

static Value tensorReduceNumber(const ObjTensor* tensor, const char* kind) {
    float result;
    int i;

    if (tensor->size == 0) {
        vmRaiseExceptionMessage("ValueError", "Reduction on empty tensor.");
        return NIL_VAL;
    }
    if (!syncTensorFromMetal((ObjTensor*)tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NIL_VAL;
    }

    if (strcmp(kind, "sum") == 0) {
        tpComputeSumF32(&vm.compute, tensor->data, tensor->size, &result, TP_COMPUTE_MODE_AUTO, NULL);
        return NUMBER_VAL((double)result);
    }
    if (strcmp(kind, "mean") == 0) {
        tpComputeSumF32(&vm.compute, tensor->data, tensor->size, &result, TP_COMPUTE_MODE_AUTO, NULL);
        return NUMBER_VAL((double)(result / (float)tensor->size));
    }

    result = tensor->data[0];
    for (i = 1; i < tensor->size; i++) {
        if (tensor->data[i] > result) {
            result = tensor->data[i];
        }
    }
    return NUMBER_VAL((double)result);
}

static Value tensorMatmul(Value left, Value right) {
    ObjTensor* a;
    ObjTensor* b;
    ObjTensor* out;
    bool useMetal;
    int i;
    int j;
    int k;

    if (!IS_TENSOR(left) || !IS_TENSOR(right)) {
        vmRaiseExceptionMessage("TypeError", "matmul() expects two tensors.");
        return NIL_VAL;
    }

    a = AS_TENSOR(left);
    b = AS_TENSOR(right);

    if (a->rank == 1 && b->rank == 1) {
        float dot;
        if (!syncTensorFromMetal(a) || !syncTensorFromMetal(b)) {
            vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
            return NIL_VAL;
        }
        if (a->shape[0] != b->shape[0]) {
            vmRaiseExceptionMessage("ValueError", "matmul() dimension mismatch.");
            return NIL_VAL;
        }
        tpComputeDotF32(&vm.compute, a->data, b->data, a->shape[0], &dot, TP_COMPUTE_MODE_AUTO, NULL);
        if (a->requiresGrad || b->requiresGrad) {
            out = newScalarTensor(dot, vm.cpuDevice);
            if (out == NULL) {
                vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
                return NIL_VAL;
            }
            if (!setAutogradBinary(out, a, b, TP_AUTOGRAD_MATMUL)) {
                return NIL_VAL;
            }
            return OBJ_VAL(out);
        }
        return NUMBER_VAL((double)dot);
    }

    if (a->rank == 2 && b->rank == 2) {
        int shape[2];
        if (a->shape[1] != b->shape[0]) {
            vmRaiseExceptionMessage("ValueError", "matmul() dimension mismatch.");
            return NIL_VAL;
        }
        useMetal = (a->device->kind == TP_DEVICE_METAL || b->device->kind == TP_DEVICE_METAL) &&
                   vm.metalDevice != NULL &&
                   vm.metalBackend != NULL &&
                   a->metalBuffer != NULL &&
                   b->metalBuffer != NULL;
        shape[0] = a->shape[0];
        shape[1] = b->shape[1];
        out = createTensorFromShape(2, shape,
                                    useMetal
                                        ? vm.metalDevice
                                        : vm.cpuDevice);
        if (out == NULL) {
            vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
            return NIL_VAL;
        }
        if (useMetal &&
            out->metalBuffer != NULL &&
            tpMetalMatmulF32(vm.metalBackend,
                             a->metalBuffer,
                             b->metalBuffer,
                             out->metalBuffer,
                             a->shape[0],
                             a->shape[1],
                             b->shape[1])) {
            out->cpuDirty = true;
            out->metalDirty = false;
        } else {
            if (!syncTensorFromMetal(a) || !syncTensorFromMetal(b)) {
                vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
                return NIL_VAL;
            }
            for (i = 0; i < shape[0]; i++) {
                for (j = 0; j < shape[1]; j++) {
                    float total = 0.0f;
                    for (k = 0; k < a->shape[1]; k++) {
                        total += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
                    }
                    out->data[i * shape[1] + j] = total;
                }
            }
            syncTensorToMetal(out);
        }
        if (!setAutogradBinary(out, a, b, TP_AUTOGRAD_MATMUL)) {
            return NIL_VAL;
        }
        return OBJ_VAL(out);
    }

    if (a->rank == 2 && b->rank == 1) {
        int shape[1];
        if (!syncTensorFromMetal(a) || !syncTensorFromMetal(b)) {
            vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
            return NIL_VAL;
        }
        if (a->shape[1] != b->shape[0]) {
            vmRaiseExceptionMessage("ValueError", "matmul() dimension mismatch.");
            return NIL_VAL;
        }
        shape[0] = a->shape[0];
        out = createTensorFromShape(1, shape,
                                    (a->device->kind == TP_DEVICE_METAL || b->device->kind == TP_DEVICE_METAL)
                                        ? vm.metalDevice
                                        : vm.cpuDevice);
        if (out == NULL) {
            vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
            return NIL_VAL;
        }
        for (i = 0; i < a->shape[0]; i++) {
            float total = 0.0f;
            for (k = 0; k < a->shape[1]; k++) {
                total += a->data[i * a->shape[1] + k] * b->data[k];
            }
            out->data[i] = total;
        }
        syncTensorToMetal(out);
        if (!setAutogradBinary(out, a, b, TP_AUTOGRAD_MATMUL)) {
            return NIL_VAL;
        }
        return OBJ_VAL(out);
    }

    if (a->rank == 1 && b->rank == 2) {
        int shape[1];
        if (!syncTensorFromMetal(a) || !syncTensorFromMetal(b)) {
            vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
            return NIL_VAL;
        }
        if (a->shape[0] != b->shape[0]) {
            vmRaiseExceptionMessage("ValueError", "matmul() dimension mismatch.");
            return NIL_VAL;
        }
        shape[0] = b->shape[1];
        out = createTensorFromShape(1, shape,
                                    (a->device->kind == TP_DEVICE_METAL || b->device->kind == TP_DEVICE_METAL)
                                        ? vm.metalDevice
                                        : vm.cpuDevice);
        if (out == NULL) {
            vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
            return NIL_VAL;
        }
        for (j = 0; j < b->shape[1]; j++) {
            float total = 0.0f;
            for (k = 0; k < a->shape[0]; k++) {
                total += a->data[k] * b->data[k * b->shape[1] + j];
            }
            out->data[j] = total;
        }
        syncTensorToMetal(out);
        if (!setAutogradBinary(out, a, b, TP_AUTOGRAD_MATMUL)) {
            return NIL_VAL;
        }
        return OBJ_VAL(out);
    }

    vmRaiseExceptionMessage("RuntimeError", "matmul() currently supports only 1D/2D tensors.");
    return NIL_VAL;
}

static Value tensorSoftmax(Value value) {
    ObjTensor* tensor;
    ObjTensor* out;
    int lastDim;
    int outer;
    int base;
    int i;

    if (!IS_TENSOR(value)) {
        vmRaiseExceptionMessage("TypeError", "softmax() expects a tensor.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(value);
    if (!syncTensorFromMetal(tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NIL_VAL;
    }
    if (tensor->requiresGrad) {
        vmRaiseExceptionMessage("RuntimeError", "softmax() backward is not implemented yet.");
        return NIL_VAL;
    }
    if (tensor->rank == 0) {
        return OBJ_VAL(newTensor(0, NULL, tensor->dtype, tensor->device, tensor->data));
    }

    out = createTensorLike(tensor);
    if (out == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NIL_VAL;
    }

    lastDim = tensor->shape[tensor->rank - 1];
    outer = lastDim == 0 ? 0 : tensor->size / lastDim;
    for (base = 0; base < outer; base++) {
        int offset = base * lastDim;
        float maxValue = tensor->data[offset];
        float sum = 0.0f;
        for (i = 1; i < lastDim; i++) {
            float v = tensor->data[offset + i];
            if (v > maxValue) {
                maxValue = v;
            }
        }
        for (i = 0; i < lastDim; i++) {
            float e = expf(tensor->data[offset + i] - maxValue);
            out->data[offset + i] = e;
            sum += e;
        }
        for (i = 0; i < lastDim; i++) {
            out->data[offset + i] /= sum;
        }
    }
    syncTensorToMetal(out);
    return OBJ_VAL(out);
}

static Value tensorLayerNorm(Value value, float eps) {
    ObjTensor* tensor;
    ObjTensor* out;
    int lastDim;
    int outer;
    int base;
    int i;

    if (!IS_TENSOR(value)) {
        vmRaiseExceptionMessage("TypeError", "layernorm() expects a tensor.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(value);
    if (!syncTensorFromMetal(tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NIL_VAL;
    }
    if (tensor->requiresGrad) {
        vmRaiseExceptionMessage("RuntimeError", "layernorm() backward is not implemented yet.");
        return NIL_VAL;
    }
    if (tensor->rank == 0) {
        return value;
    }

    out = createTensorLike(tensor);
    if (out == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NIL_VAL;
    }

    lastDim = tensor->shape[tensor->rank - 1];
    outer = lastDim == 0 ? 0 : tensor->size / lastDim;
    for (base = 0; base < outer; base++) {
        int offset = base * lastDim;
        float mean = 0.0f;
        float variance = 0.0f;
        for (i = 0; i < lastDim; i++) {
            mean += tensor->data[offset + i];
        }
        mean /= (float)lastDim;
        for (i = 0; i < lastDim; i++) {
            float delta = tensor->data[offset + i] - mean;
            variance += delta * delta;
        }
        variance /= (float)lastDim;
        variance = sqrtf(variance + eps);
        for (i = 0; i < lastDim; i++) {
            out->data[offset + i] = (tensor->data[offset + i] - mean) / variance;
        }
    }
    syncTensorToMetal(out);
    return OBJ_VAL(out);
}

typedef struct {
    ObjTensor** items;
    int count;
    int capacity;
} TensorNodeArray;

static bool tensorOnCpuForTraining(const ObjTensor* tensor) {
    return tensor != NULL && tensor->device != NULL && tensor->device->kind == TP_DEVICE_CPU;
}

static bool appendTensorNode(TensorNodeArray* array, ObjTensor* tensor) {
    ObjTensor** items;

    if (array->count >= array->capacity) {
        int newCapacity = array->capacity < 8 ? 8 : array->capacity * 2;
        items = (ObjTensor**)realloc(array->items, sizeof(ObjTensor*) * (size_t)newCapacity);
        if (items == NULL) {
            return false;
        }
        array->items = items;
        array->capacity = newCapacity;
    }

    array->items[array->count++] = tensor;
    return true;
}

static bool tensorArrayContains(const TensorNodeArray* array, const ObjTensor* tensor) {
    int i;

    for (i = 0; i < array->count; i++) {
        if (array->items[i] == tensor) {
            return true;
        }
    }
    return false;
}

static void freeTensorNodeArray(TensorNodeArray* array) {
    free(array->items);
    array->items = NULL;
    array->count = 0;
    array->capacity = 0;
}

static ObjTensor* newScalarTensor(float value, ObjDevice* device) {
    return newTensor(0, NULL, vm.float32DType, device != NULL ? device : vm.cpuDevice, &value);
}

static bool ensureAutogradSupported(const ObjTensor* tensor) {
    if (!tensorOnCpuForTraining(tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Autograd is currently supported only for CPU tensors.");
        return false;
    }
    return true;
}

static bool ensureTensorGrad(ObjTensor* tensor) {
    if (tensor->grad != NULL) {
        return true;
    }

    tensor->grad = newTensor(tensor->rank, tensor->shape, vm.float32DType, vm.cpuDevice, NULL);
    if (tensor->grad == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while allocating gradient tensor.");
        return false;
    }
    tensor->grad->requiresGrad = false;
    return true;
}

static void zeroTensorData(ObjTensor* tensor) {
    if (tensor == NULL || tensor->data == NULL || tensor->size <= 0) {
        return;
    }
    memset(tensor->data, 0, sizeof(float) * (size_t)tensor->size);
}

static bool accumulateToParentFromValues(ObjTensor* parent,
                                         const ObjTensor* outGrad,
                                         const int* outShape,
                                         int outRank,
                                         const float* values) {
    int* outStrides = NULL;
    int* parentStrides = NULL;
    int i;

    if (parent == NULL || values == NULL) {
        return false;
    }
    if (!ensureTensorGrad(parent)) {
        return false;
    }

    if (tensorShapeEquals(parent, outGrad)) {
        tpComputeAddF32(&vm.compute,
                        parent->grad->data,
                        values,
                        parent->grad->data,
                        parent->size,
                        TP_COMPUTE_MODE_AUTO,
                        NULL);
        return true;
    }

    outStrides = outRank > 0 ? (int*)malloc(sizeof(int) * (size_t)outRank) : NULL;
    parentStrides = parent->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)parent->rank) : NULL;
    if ((outRank > 0 && outStrides == NULL) || (parent->rank > 0 && parentStrides == NULL)) {
        free(outStrides);
        free(parentStrides);
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while accumulating gradients.");
        return false;
    }

    if (outRank > 0) {
        fillStrides(outShape, outRank, outStrides);
    }
    if (parent->rank > 0) {
        fillStrides(parent->shape, parent->rank, parentStrides);
    }

    for (i = 0; i < outGrad->size; i++) {
        int parentOffset = broadcastOffsetForIndex(i, outShape, outStrides, outRank, parent, parentStrides);
        parent->grad->data[parentOffset] += values[i];
    }

    free(outStrides);
    free(parentStrides);
    return true;
}

static bool accumulateToParentScaled(ObjTensor* parent,
                                     const ObjTensor* outGrad,
                                     const int* outShape,
                                     int outRank,
                                     float scale) {
    float* values;
    int i;
    bool ok;

    if (parent == NULL) {
        return true;
    }

    values = outGrad->size > 0 ? (float*)malloc(sizeof(float) * (size_t)outGrad->size) : NULL;
    if (outGrad->size > 0 && values == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while accumulating gradients.");
        return false;
    }
    for (i = 0; i < outGrad->size; i++) {
        values[i] = outGrad->grad->data[i] * scale;
    }
    ok = accumulateToParentFromValues(parent, outGrad, outShape, outRank, values);
    free(values);
    return ok;
}

static bool setAutogradBinary(ObjTensor* out,
                              ObjTensor* a,
                              ObjTensor* b,
                              TPAutogradOp op) {
    if (out == NULL) {
        return false;
    }

    out->requiresGrad = (a != NULL && a->requiresGrad) || (b != NULL && b->requiresGrad);
    if (!out->requiresGrad) {
        return true;
    }

    if ((a != NULL && !ensureAutogradSupported(a)) || (b != NULL && !ensureAutogradSupported(b))) {
        return false;
    }
    if (!ensureAutogradSupported(out)) {
        return false;
    }

    out->parentA = a;
    out->parentB = b;
    out->gradOp = op;
    return true;
}

static bool setAutogradUnary(ObjTensor* out,
                             ObjTensor* input,
                             TPAutogradOp op) {
    return setAutogradBinary(out, input, NULL, op);
}

static float geluDerivative(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float tanhInner = tanhf(inner);
    float sech2 = 1.0f - tanhInner * tanhInner;
    float innerPrime = 0.7978845608f * (1.0f + 0.134145f * x2);
    return 0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * innerPrime;
}

static bool buildTopo(ObjTensor* tensor, TensorNodeArray* visited, TensorNodeArray* topo) {
    if (tensor == NULL) {
        return true;
    }
    if (tensorArrayContains(visited, tensor)) {
        return true;
    }
    if (!appendTensorNode(visited, tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while building autograd graph.");
        return false;
    }
    if (!buildTopo(tensor->parentA, visited, topo)) {
        return false;
    }
    if (!buildTopo(tensor->parentB, visited, topo)) {
        return false;
    }
    if (!appendTensorNode(topo, tensor)) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while building autograd graph.");
        return false;
    }
    return true;
}

static bool backwardMatmul(ObjTensor* out) {
    ObjTensor* a = out->parentA;
    ObjTensor* b = out->parentB;
    ObjTensor* grad = out->grad;
    int i;
    int j;
    int k;

    if (a == NULL || b == NULL || grad == NULL) {
        return true;
    }

    if (a->rank == 1 && b->rank == 1) {
        if (a->requiresGrad) {
            if (!ensureTensorGrad(a)) return false;
            for (i = 0; i < a->shape[0]; i++) {
                a->grad->data[i] += grad->data[0] * b->data[i];
            }
        }
        if (b->requiresGrad) {
            if (!ensureTensorGrad(b)) return false;
            for (i = 0; i < b->shape[0]; i++) {
                b->grad->data[i] += grad->data[0] * a->data[i];
            }
        }
        return true;
    }

    if (a->rank == 2 && b->rank == 2) {
        int m = a->shape[0];
        int n = a->shape[1];
        int p = b->shape[1];
        if (a->requiresGrad) {
            if (!ensureTensorGrad(a)) return false;
            for (i = 0; i < m; i++) {
                for (k = 0; k < n; k++) {
                    float total = 0.0f;
                    for (j = 0; j < p; j++) {
                        total += grad->data[i * p + j] * b->data[k * p + j];
                    }
                    a->grad->data[i * n + k] += total;
                }
            }
        }
        if (b->requiresGrad) {
            if (!ensureTensorGrad(b)) return false;
            for (k = 0; k < n; k++) {
                for (j = 0; j < p; j++) {
                    float total = 0.0f;
                    for (i = 0; i < m; i++) {
                        total += a->data[i * n + k] * grad->data[i * p + j];
                    }
                    b->grad->data[k * p + j] += total;
                }
            }
        }
        return true;
    }

    if (a->rank == 2 && b->rank == 1) {
        int m = a->shape[0];
        int n = a->shape[1];
        if (a->requiresGrad) {
            if (!ensureTensorGrad(a)) return false;
            for (i = 0; i < m; i++) {
                for (k = 0; k < n; k++) {
                    a->grad->data[i * n + k] += grad->data[i] * b->data[k];
                }
            }
        }
        if (b->requiresGrad) {
            if (!ensureTensorGrad(b)) return false;
            for (k = 0; k < n; k++) {
                float total = 0.0f;
                for (i = 0; i < m; i++) {
                    total += a->data[i * n + k] * grad->data[i];
                }
                b->grad->data[k] += total;
            }
        }
        return true;
    }

    if (a->rank == 1 && b->rank == 2) {
        int n = a->shape[0];
        int p = b->shape[1];
        if (a->requiresGrad) {
            if (!ensureTensorGrad(a)) return false;
            for (k = 0; k < n; k++) {
                float total = 0.0f;
                for (j = 0; j < p; j++) {
                    total += grad->data[j] * b->data[k * p + j];
                }
                a->grad->data[k] += total;
            }
        }
        if (b->requiresGrad) {
            if (!ensureTensorGrad(b)) return false;
            for (k = 0; k < n; k++) {
                for (j = 0; j < p; j++) {
                    b->grad->data[k * p + j] += a->data[k] * grad->data[j];
                }
            }
        }
        return true;
    }

    vmRaiseExceptionMessage("RuntimeError", "matmul() backward is not implemented for this tensor rank.");
    return false;
}

static bool backwardTensorNode(ObjTensor* out) {
    ObjTensor* a = out->parentA;
    ObjTensor* b = out->parentB;
    float* values = NULL;
    int i;
    bool ok = true;

    if (out == NULL || out->grad == NULL || out->gradOp == TP_AUTOGRAD_NONE) {
        return true;
    }

    switch (out->gradOp) {
        case TP_AUTOGRAD_ADD:
            if (a != NULL && a->requiresGrad) {
                ok = accumulateToParentFromValues(a, out, out->shape, out->rank, out->grad->data);
                if (!ok) return false;
            }
            if (b != NULL && b->requiresGrad) {
                ok = accumulateToParentFromValues(b, out, out->shape, out->rank, out->grad->data);
            }
            return ok;
        case TP_AUTOGRAD_SUB:
            if (a != NULL && a->requiresGrad) {
                ok = accumulateToParentFromValues(a, out, out->shape, out->rank, out->grad->data);
                if (!ok) return false;
            }
            if (b != NULL && b->requiresGrad) {
                ok = accumulateToParentScaled(b, out, out->shape, out->rank, -1.0f);
            }
            return ok;
        case TP_AUTOGRAD_MUL:
        case TP_AUTOGRAD_DIV:
            values = out->size > 0 ? (float*)malloc(sizeof(float) * (size_t)out->size) : NULL;
            if (out->size > 0 && values == NULL) {
                vmRaiseExceptionMessage("RuntimeError", "Out of memory while propagating gradients.");
                return false;
            }
            break;
        default:
            break;
    }

    if (out->gradOp == TP_AUTOGRAD_MUL || out->gradOp == TP_AUTOGRAD_DIV) {
        int* outStrides = out->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)out->rank) : NULL;
        int* aStrides = a != NULL && a->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)a->rank) : NULL;
        int* bStrides = b != NULL && b->rank > 0 ? (int*)malloc(sizeof(int) * (size_t)b->rank) : NULL;
        if ((out->rank > 0 && outStrides == NULL) ||
            (a != NULL && a->rank > 0 && aStrides == NULL) ||
            (b != NULL && b->rank > 0 && bStrides == NULL)) {
            free(values);
            free(outStrides);
            free(aStrides);
            free(bStrides);
            vmRaiseExceptionMessage("RuntimeError", "Out of memory while propagating gradients.");
            return false;
        }
        if (out->rank > 0) fillStrides(out->shape, out->rank, outStrides);
        if (a != NULL && a->rank > 0) fillStrides(a->shape, a->rank, aStrides);
        if (b != NULL && b->rank > 0) fillStrides(b->shape, b->rank, bStrides);

        if (a != NULL && a->requiresGrad) {
            for (i = 0; i < out->size; i++) {
                int bOffset = b != NULL ? broadcastOffsetForIndex(i, out->shape, outStrides, out->rank, b, bStrides) : 0;
                float rhs = b != NULL ? b->data[bOffset] : 0.0f;
                values[i] = out->grad->data[i] * (out->gradOp == TP_AUTOGRAD_MUL ? rhs : (1.0f / rhs));
            }
            ok = accumulateToParentFromValues(a, out, out->shape, out->rank, values);
            if (!ok) {
                free(values);
                free(outStrides);
                free(aStrides);
                free(bStrides);
                return false;
            }
        }
        if (b != NULL && b->requiresGrad) {
            for (i = 0; i < out->size; i++) {
                int aOffset = a != NULL ? broadcastOffsetForIndex(i, out->shape, outStrides, out->rank, a, aStrides) : 0;
                int bOffset = broadcastOffsetForIndex(i, out->shape, outStrides, out->rank, b, bStrides);
                float lhs = a != NULL ? a->data[aOffset] : 0.0f;
                float rhs = b->data[bOffset];
                if (out->gradOp == TP_AUTOGRAD_MUL) {
                    values[i] = out->grad->data[i] * lhs;
                } else {
                    values[i] = out->grad->data[i] * (-lhs / (rhs * rhs));
                }
            }
            ok = accumulateToParentFromValues(b, out, out->shape, out->rank, values);
        }
        free(values);
        free(outStrides);
        free(aStrides);
        free(bStrides);
        return ok;
    }

    switch (out->gradOp) {
        case TP_AUTOGRAD_RESHAPE:
            if (a != NULL && a->requiresGrad) {
                if (!ensureTensorGrad(a)) return false;
                for (i = 0; i < a->size; i++) {
                    a->grad->data[i] += out->grad->data[i];
                }
            }
            return true;
        case TP_AUTOGRAD_RELU:
        case TP_AUTOGRAD_TANH:
        case TP_AUTOGRAD_SIGMOID:
        case TP_AUTOGRAD_GELU:
            if (a != NULL && a->requiresGrad) {
                if (!ensureTensorGrad(a)) return false;
                for (i = 0; i < a->size; i++) {
                    float local = 1.0f;
                    if (out->gradOp == TP_AUTOGRAD_RELU) {
                        local = a->data[i] > 0.0f ? 1.0f : 0.0f;
                    } else if (out->gradOp == TP_AUTOGRAD_TANH) {
                        float y = out->data[i];
                        local = 1.0f - y * y;
                    } else if (out->gradOp == TP_AUTOGRAD_SIGMOID) {
                        float y = out->data[i];
                        local = y * (1.0f - y);
                    } else if (out->gradOp == TP_AUTOGRAD_GELU) {
                        local = geluDerivative(a->data[i]);
                    }
                    a->grad->data[i] += out->grad->data[i] * local;
                }
            }
            return true;
        case TP_AUTOGRAD_MATMUL:
            return backwardMatmul(out);
        case TP_AUTOGRAD_CONV2D: {
            ObjTensor* input = a;
            ObjTensor* weight = b;
            ObjTensor* grad = out->grad;
            int batch;
            int outChannels;
            int inChannels;
            int outHeight;
            int outWidth;
            int kernelHeight;
            int kernelWidth;
            int n;
            int oc;
            int ic;
            int oy;
            int ox;
            int ky;
            int kx;

            if (input == NULL || weight == NULL || grad == NULL) {
                return true;
            }

            batch = input->shape[0];
            inChannels = input->shape[1];
            outChannels = weight->shape[0];
            kernelHeight = weight->shape[2];
            kernelWidth = weight->shape[3];
            outHeight = out->shape[2];
            outWidth = out->shape[3];

            if (input->requiresGrad) {
                if (!ensureTensorGrad(input)) return false;
                for (n = 0; n < batch; n++) {
                    for (oc = 0; oc < outChannels; oc++) {
                        for (oy = 0; oy < outHeight; oy++) {
                            for (ox = 0; ox < outWidth; ox++) {
                                float go = grad->data[((n * outChannels + oc) * outHeight + oy) * outWidth + ox];
                                for (ic = 0; ic < inChannels; ic++) {
                                    for (ky = 0; ky < kernelHeight; ky++) {
                                        for (kx = 0; kx < kernelWidth; kx++) {
                                            int inIndex = ((n * inChannels + ic) * input->shape[2] + (oy + ky)) * input->shape[3] + (ox + kx);
                                            int wIndex = ((oc * inChannels + ic) * kernelHeight + ky) * kernelWidth + kx;
                                            input->grad->data[inIndex] += go * weight->data[wIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (weight->requiresGrad) {
                if (!ensureTensorGrad(weight)) return false;
                for (oc = 0; oc < outChannels; oc++) {
                    for (ic = 0; ic < inChannels; ic++) {
                        for (ky = 0; ky < kernelHeight; ky++) {
                            for (kx = 0; kx < kernelWidth; kx++) {
                                float total = 0.0f;
                                for (n = 0; n < batch; n++) {
                                    for (oy = 0; oy < outHeight; oy++) {
                                        for (ox = 0; ox < outWidth; ox++) {
                                            int inIndex = ((n * inChannels + ic) * input->shape[2] + (oy + ky)) * input->shape[3] + (ox + kx);
                                            int outIndex = ((n * outChannels + oc) * outHeight + oy) * outWidth + ox;
                                            total += input->data[inIndex] * grad->data[outIndex];
                                        }
                                    }
                                }
                                weight->grad->data[((oc * inChannels + ic) * kernelHeight + ky) * kernelWidth + kx] += total;
                            }
                        }
                    }
                }
            }
            return true;
        }
        case TP_AUTOGRAD_MSE_LOSS: {
            float scale = out->grad->data[0] * out->gradAux;
            ObjTensor* pred = a;
            ObjTensor* target = b;
            if (pred != NULL && pred->requiresGrad) {
                if (!ensureTensorGrad(pred)) return false;
                for (i = 0; i < pred->size; i++) {
                    pred->grad->data[i] += scale * (pred->data[i] - target->data[i]);
                }
            }
            if (target != NULL && target->requiresGrad) {
                if (!ensureTensorGrad(target)) return false;
                for (i = 0; i < target->size; i++) {
                    target->grad->data[i] -= scale * (pred->data[i] - target->data[i]);
                }
            }
            return true;
        }
        case TP_AUTOGRAD_NONE:
        case TP_AUTOGRAD_ADD:
        case TP_AUTOGRAD_SUB:
        case TP_AUTOGRAD_MUL:
        case TP_AUTOGRAD_DIV:
            return true;
    }

    return true;
}

static Value tensorBackward(Value value) {
    ObjTensor* tensor;
    TensorNodeArray visited = {0};
    TensorNodeArray topo = {0};
    int i;

    if (!IS_TENSOR(value)) {
        vmRaiseExceptionMessage("TypeError", "backward() expects a tensor.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(value);
    if (!tensor->requiresGrad) {
        vmRaiseExceptionMessage("RuntimeError", "backward() requires a tensor that tracks gradients.");
        return NIL_VAL;
    }
    if (tensor->size != 1) {
        vmRaiseExceptionMessage("RuntimeError", "backward() currently requires a scalar tensor.");
        return NIL_VAL;
    }
    if (!ensureAutogradSupported(tensor)) {
        return NIL_VAL;
    }
    if (!buildTopo(tensor, &visited, &topo)) {
        freeTensorNodeArray(&visited);
        freeTensorNodeArray(&topo);
        return NIL_VAL;
    }
    if (!ensureTensorGrad(tensor)) {
        freeTensorNodeArray(&visited);
        freeTensorNodeArray(&topo);
        return NIL_VAL;
    }
    tensor->grad->data[0] += 1.0f;

    for (i = topo.count - 1; i >= 0; i--) {
        if (!backwardTensorNode(topo.items[i])) {
            freeTensorNodeArray(&visited);
            freeTensorNodeArray(&topo);
            return NIL_VAL;
        }
    }

    freeTensorNodeArray(&visited);
    freeTensorNodeArray(&topo);
    return NIL_VAL;
}

static Value tensorZeroGrad(Value value) {
    ObjTensor* tensor;

    if (!IS_TENSOR(value)) {
        vmRaiseExceptionMessage("TypeError", "zero_grad() expects a tensor.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(value);
    if (tensor->grad != NULL) {
        zeroTensorData(tensor->grad);
    }
    return NIL_VAL;
}

static bool valueIsTensorContainer(Value value) {
    return IS_TENSOR(value) || IS_LIST(value) || IS_TUPLE(value);
}

static bool collectTensorValues(Value value, ObjList* out) {
    int count;
    int i;

    if (IS_TENSOR(value)) {
        writeValueArray(&out->items, value);
        return true;
    }
    if (!valueIsTensorContainer(value)) {
        return false;
    }

    count = sequenceLength(value);
    for (i = 0; i < count; i++) {
        if (!collectTensorValues(sequenceValueAt(value, i), out)) {
            return false;
        }
    }
    return true;
}

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
    (void)argCount;
    (void)args;
    return NUMBER_VAL(platformClockSeconds());
}

static Value floatNative(int argCount, Value* args) {
    char* end;
    double value;

    if (argCount != 1) return NIL_VAL;
    if (IS_NUMBER(args[0])) return args[0];
    if (IS_BOOL(args[0])) return NUMBER_VAL(AS_BOOL(args[0]) ? 1.0 : 0.0);
    if (!IS_STRING(args[0])) return NIL_VAL;

    value = strtod(AS_CSTRING(args[0]), &end);
    if (end == AS_CSTRING(args[0]) || *end != '\0') {
        vmRaiseExceptionMessage("ValueError", "could not convert string to float");
        return NIL_VAL;
    }
    return NUMBER_VAL(value);
}

static Value intNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    if (IS_NUMBER(args[0])) return NUMBER_VAL((double)((int)AS_NUMBER(args[0])));
    if (IS_BOOL(args[0])) return NUMBER_VAL(AS_BOOL(args[0]) ? 1.0 : 0.0);
    if (IS_STRING(args[0])) {
        Value parsed = floatNative(argCount, args);
        if (IS_NIL(parsed)) {
            return NIL_VAL;
        }
        return NUMBER_VAL((double)((int)AS_NUMBER(parsed)));
    }
    return NIL_VAL;
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

    if (IS_TENSOR(args[0])) {
        ObjTensor* tensor = AS_TENSOR(args[0]);
        if (tensor->rank == 0) {
            return NUMBER_VAL(0);
        }
        return NUMBER_VAL((double)tensor->shape[0]);
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
                case OBJ_CLOSURE:  return OBJ_VAL(copyString("function", 8));
                case OBJ_SET:      return OBJ_VAL(copyString("set", 3));
                case OBJ_LIST:     return OBJ_VAL(copyString("list", 4));
                case OBJ_DICT:     return OBJ_VAL(copyString("dict", 4));
                case OBJ_TUPLE:    return OBJ_VAL(copyString("tuple", 5));
                case OBJ_BYTES:    return OBJ_VAL(copyString("bytes", 5));
                case OBJ_DEVICE:   return OBJ_VAL(copyString("device", 6));
                case OBJ_DTYPE:    return OBJ_VAL(copyString("dtype", 5));
                case OBJ_TENSOR:   return OBJ_VAL(copyString("tensor", 6));
                case OBJ_CLASS:    return OBJ_VAL(copyString("type", 4));
                case OBJ_INSTANCE: return OBJ_VAL(AS_INSTANCE(args[0])->klass->name);
                default:           return OBJ_VAL(copyString("object", 6));
            }
        }
    }
    return NIL_VAL;
}

// Built-in: isinstance()
static Value isinstanceNative(int argCount, Value* args) {
    if (argCount != 2) return BOOL_VAL(false);

    if (IS_CLASS(args[1])) {
        if (IS_INSTANCE(args[0])) {
            return BOOL_VAL(classMatchesExpected(AS_INSTANCE(args[0])->klass, AS_CLASS(args[1])));
        }
        return BOOL_VAL(false);
    }

    if (IS_STRING(args[1])) {
        return BOOL_VAL(valueMatchesTypeName(args[0], AS_STRING(args[1])));
    }

    return BOOL_VAL(false);
}

// Built-in: getattr()
static Value getattrNative(int argCount, Value* args) {
    if (argCount < 2 || argCount > 3 || !IS_STRING(args[1])) return NIL_VAL;

    Value value;
    if (getAttributeValue(args[0], AS_STRING(args[1]), &value)) {
        return value;
    }

    if (argCount == 3) {
        return args[2];
    }

    return NIL_VAL;
}

// Built-in: hasattr()
static Value hasattrNative(int argCount, Value* args) {
    if (argCount != 2 || !IS_STRING(args[1])) return BOOL_VAL(false);

    Value value;
    return BOOL_VAL(getAttributeValue(args[0], AS_STRING(args[1]), &value));
}

static Value callableNative(int argCount, Value* args) {
    Value value;
    ObjString* callName;

    if (argCount != 1) return BOOL_VAL(false);

    if (IS_NATIVE(args[0]) || IS_CLOSURE(args[0]) || IS_CLASS(args[0]) || IS_BOUND_METHOD(args[0])) {
        return BOOL_VAL(true);
    }

    if (!IS_INSTANCE(args[0])) {
        return BOOL_VAL(false);
    }

    callName = copyString("__call__", 8);
    return BOOL_VAL(getAttributeValue(args[0], callName, &value));
}

// Built-in: setattr()
static Value setattrNative(int argCount, Value* args) {
    if (argCount != 3 || !IS_STRING(args[1])) return NIL_VAL;

    if (!IS_INSTANCE(args[0])) {
        return NIL_VAL;
    }

    tableSet(&AS_INSTANCE(args[0])->fields, args[1], args[2]);
    return NIL_VAL;
}

static void appendDirEntries(ObjList* out, Table* table) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (!IS_NIL(entry->key)) {
            writeValueArray(&out->items, entry->key);
        }
    }
}

// Built-in: round()
static Value roundNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_NUMBER(args[0])) return NIL_VAL;
    double value = AS_NUMBER(args[0]);
    return NUMBER_VAL(floor(value + 0.5));
}

// Built-in: ord()
static Value ordNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;
    ObjString* string = AS_STRING(args[0]);
    if (string->length != 1) return NIL_VAL;
    return NUMBER_VAL((unsigned char)string->chars[0]);
}

// Built-in: chr()
static Value chrNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_NUMBER(args[0])) return NIL_VAL;
    char chars[2];
    chars[0] = (char)AS_NUMBER(args[0]);
    chars[1] = '\0';
    return OBJ_VAL(copyString(chars, 1));
}

// Built-in: dir()
static Value dirNative(int argCount, Value* args) {
    ObjList* out = newList();
    if (argCount == 0) {
        appendDirEntries(out, vm.globalEnv->table);
        return OBJ_VAL(out);
    }

    if (IS_INSTANCE(args[0])) {
        ObjInstance* instance = AS_INSTANCE(args[0]);
        appendDirEntries(out, &instance->fields);
        appendDirEntries(out, &instance->klass->methods);
        return OBJ_VAL(out);
    }

    if (IS_CLASS(args[0])) {
        appendDirEntries(out, &AS_CLASS(args[0])->methods);
        return OBJ_VAL(out);
    }

    appendNativeObjectDirEntries(out, args[0]);

    return OBJ_VAL(out);
}

// Internal platform helpers for pure TensorPy modules.
static Value platformTimeNative(int argCount, Value* args) {
    (void)argCount;
    (void)args;
    return NUMBER_VAL(platformClockSeconds());
}

static Value platformRandomNative(int argCount, Value* args) {
    (void)argCount;
    (void)args;
    return NUMBER_VAL(platformRandomDouble());
}

static Value platformGetcwdNative(int argCount, Value* args) {
    (void)argCount;
    (void)args;
    char* cwd = platformGetCurrentDirectory();
    if (cwd == NULL) return NIL_VAL;
    ObjString* string = copyString(cwd, (int)strlen(cwd));
    free(cwd);
    return OBJ_VAL(string);
}

static Value platformNameNative(int argCount, Value* args) {
    (void)argCount;
    (void)args;
    const char* name = platformName();
    return OBJ_VAL(copyString(name, (int)strlen(name)));
}

static Value platformReadTextNative(int argCount, Value* args) {
    char* text;
    ObjString* result;

    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;

    text = platformReadTextFile(AS_CSTRING(args[0]));
    if (text == NULL) return NIL_VAL;

    result = copyString(text, (int)strlen(text));
    free(text);
    return OBJ_VAL(result);
}

static Value platformReadBytesNative(int argCount, Value* args) {
    int count = 0;
    uint8_t* bytes;
    ObjBytes* result;

    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;

    bytes = platformReadBinaryFile(AS_CSTRING(args[0]), &count);
    if (bytes == NULL && count != 0) return NIL_VAL;

    result = newBytes(count, bytes);
    free(bytes);
    return OBJ_VAL(result);
}

static Value platformWriteTextNative(int argCount, Value* args) {
    if (argCount != 2 || !IS_STRING(args[0]) || !IS_STRING(args[1])) return BOOL_VAL(false);
    return BOOL_VAL(platformWriteTextFile(AS_CSTRING(args[0]), AS_CSTRING(args[1])));
}

static Value platformWriteBytesNative(int argCount, Value* args) {
    ObjBytes* bytes;

    if (argCount != 2 || !IS_STRING(args[0]) || !IS_BYTES(args[1])) return BOOL_VAL(false);

    bytes = AS_BYTES(args[1]);
    return BOOL_VAL(platformWriteBinaryFile(AS_CSTRING(args[0]), bytes->bytes, bytes->length));
}

static Value platformGetenvNative(int argCount, Value* args) {
    char* value;
    ObjString* result;

    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;

    value = platformGetEnvironmentVariable(AS_CSTRING(args[0]));
    if (value == NULL) return NIL_VAL;

    result = copyString(value, (int)strlen(value));
    free(value);
    return OBJ_VAL(result);
}

static Value platformSystemNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return NUMBER_VAL(-1);
    return NUMBER_VAL((double)platformSystemCommand(AS_CSTRING(args[0])));
}

static Value platformListdirNative(int argCount, Value* args) {
    const char* path = ".";
    char** entries;
    int count = 0;
    ObjList* result;
    int i;

    if (argCount > 1) return NIL_VAL;
    if (argCount == 1) {
        if (!IS_STRING(args[0])) return NIL_VAL;
        path = AS_CSTRING(args[0]);
    }

    entries = platformListDirectory(path, &count);
    if (entries == NULL) return NIL_VAL;

    result = newList();
    for (i = 0; i < count; i++) {
        writeValueArray(&result->items, OBJ_VAL(copyString(entries[i], (int)strlen(entries[i]))));
    }

    platformFreeDirectoryList(entries, count);
    return OBJ_VAL(result);
}

static Value platformExistsNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;
    return BOOL_VAL(platformPathExists(AS_CSTRING(args[0])));
}

static Value platformIsdirNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;
    return BOOL_VAL(platformPathIsDirectory(AS_CSTRING(args[0])));
}

static Value platformIsfileNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return NIL_VAL;
    return BOOL_VAL(platformPathIsFile(AS_CSTRING(args[0])));
}

static Value platformMkdirNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return BOOL_VAL(false);
    return BOOL_VAL(platformCreateDirectory(AS_CSTRING(args[0])));
}

static Value platformMakedirsNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return BOOL_VAL(false);
    return BOOL_VAL(platformCreateDirectories(AS_CSTRING(args[0])));
}

static Value platformRemoveNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return BOOL_VAL(false);
    return BOOL_VAL(platformRemoveFile(AS_CSTRING(args[0])));
}

static Value platformRmdirNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_STRING(args[0])) return BOOL_VAL(false);
    return BOOL_VAL(platformRemoveDirectory(AS_CSTRING(args[0])));
}

static Value platformRenameNative(int argCount, Value* args) {
    if (argCount != 2 || !IS_STRING(args[0]) || !IS_STRING(args[1])) return BOOL_VAL(false);
    return BOOL_VAL(platformRenamePath(AS_CSTRING(args[0]), AS_CSTRING(args[1])));
}

static Value gcMarkCountNative(int argCount, Value* args) {
    (void)args;
    if (argCount != 0) return NIL_VAL;
    return NUMBER_VAL((double)gcMarkRootsAndCount());
}

static Value gcReachableCountNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    return NUMBER_VAL((double)gcCountReachableFromValue(args[0]));
}

static Value gcCollectNative(int argCount, Value* args) {
    (void)args;
    if (argCount != 0) return NIL_VAL;
    return NUMBER_VAL((double)gcCollect());
}

static Value gcObjectCountNative(int argCount, Value* args) {
    (void)args;
    if (argCount != 0) return NIL_VAL;
    return NUMBER_VAL((double)gcObjectCount());
}

static Value mathUnaryNative(int argCount, Value* args, double (*fn)(double)) {
    if (argCount != 1 || !IS_NUMBER(args[0])) return NIL_VAL;
    return NUMBER_VAL(fn(AS_NUMBER(args[0])));
}

static Value mathSqrtNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, sqrt); }
static Value mathSinNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, sin); }
static Value mathCosNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, cos); }
static Value mathTanNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, tan); }
static Value mathFloorNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, floor); }
static Value mathCeilNative(int argCount, Value* args) { return mathUnaryNative(argCount, args, ceil); }

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

static Value mlDeviceNative(int argCount, Value* args) {
    ObjDevice* device;

    if (argCount != 1) return NIL_VAL;
    device = resolveDeviceArg(args[0]);
    if (device == NULL) {
        vmRaiseExceptionMessage("ValueError", "Unknown device.");
        return NIL_VAL;
    }
    return OBJ_VAL(device);
}

static Value mlDTypeNative(int argCount, Value* args) {
    ObjDType* dtype;

    if (argCount != 1) return NIL_VAL;
    dtype = resolveDTypeArg(args[0]);
    if (dtype == NULL) {
        vmRaiseExceptionMessage("ValueError", "Unknown dtype.");
        return NIL_VAL;
    }
    return OBJ_VAL(dtype);
}

static Value mlMetalAvailableNative(int argCount, Value* args) {
    (void)args;
    if (argCount != 0) return NIL_VAL;
    return BOOL_VAL(tpMetalBackendIsAvailable(vm.metalBackend));
}

static Value mlTensorNative(int argCount, Value* args) {
    ObjDType* dtype;
    ObjDevice* device;
    ObjTensor* tensor;
    bool requiresGrad = false;

    if (argCount < 1 || argCount > 4) return NIL_VAL;

    dtype = resolveDTypeArg(argCount >= 2 ? args[1] : NIL_VAL);
    device = resolveDeviceArg(argCount >= 3 ? args[2] : NIL_VAL);
    if (argCount >= 4) {
        if (!IS_BOOL(args[3])) return NIL_VAL;
        requiresGrad = AS_BOOL(args[3]);
    }
    if (dtype == NULL) {
        vmRaiseExceptionMessage("ValueError", "Unsupported dtype.");
        return NIL_VAL;
    }
    if (device == NULL) {
        vmRaiseExceptionMessage("ValueError", "Unsupported device.");
        return NIL_VAL;
    }

    tensor = createTensorFromData(args[0], dtype, device);
    if (tensor != NULL && requiresGrad) {
        if (!ensureAutogradSupported(tensor)) {
            return NIL_VAL;
        }
        tensor->requiresGrad = true;
    }
    return tensor != NULL ? OBJ_VAL(tensor) : NIL_VAL;
}

static Value mlZerosNative(int argCount, Value* args) {
    int* shape = NULL;
    int rank = 0;
    ObjDType* dtype;
    ObjDevice* device;
    ObjTensor* tensor;

    if (argCount < 1 || argCount > 3) return NIL_VAL;
    if (!parseShapeValue(args[0], &shape, &rank)) {
        vmRaiseExceptionMessage("TypeError", "zeros() shape must be a number, list, or tuple of numbers.");
        return NIL_VAL;
    }

    dtype = resolveDTypeArg(argCount >= 2 ? args[1] : NIL_VAL);
    device = resolveDeviceArg(argCount >= 3 ? args[2] : NIL_VAL);
    if (dtype == NULL || device == NULL) {
        free(shape);
        vmRaiseExceptionMessage("ValueError", "Unsupported dtype or device.");
        return NIL_VAL;
    }

    tensor = createTensorFilled(shape, rank, dtype, device, 0.0f);
    free(shape);
    return tensor != NULL ? OBJ_VAL(tensor) : NIL_VAL;
}

static Value mlOnesNative(int argCount, Value* args) {
    int* shape = NULL;
    int rank = 0;
    ObjDType* dtype;
    ObjDevice* device;
    ObjTensor* tensor;

    if (argCount < 1 || argCount > 3) return NIL_VAL;
    if (!parseShapeValue(args[0], &shape, &rank)) {
        vmRaiseExceptionMessage("TypeError", "ones() shape must be a number, list, or tuple of numbers.");
        return NIL_VAL;
    }

    dtype = resolveDTypeArg(argCount >= 2 ? args[1] : NIL_VAL);
    device = resolveDeviceArg(argCount >= 3 ? args[2] : NIL_VAL);
    if (dtype == NULL || device == NULL) {
        free(shape);
        vmRaiseExceptionMessage("ValueError", "Unsupported dtype or device.");
        return NIL_VAL;
    }

    tensor = createTensorFilled(shape, rank, dtype, device, 1.0f);
    free(shape);
    return tensor != NULL ? OBJ_VAL(tensor) : NIL_VAL;
}

static Value mlFullNative(int argCount, Value* args) {
    int* shape = NULL;
    int rank = 0;
    ObjDType* dtype;
    ObjDevice* device;
    ObjTensor* tensor;

    if (argCount < 2 || argCount > 4 || !IS_NUMBER(args[1])) return NIL_VAL;
    if (!parseShapeValue(args[0], &shape, &rank)) {
        vmRaiseExceptionMessage("TypeError", "full() shape must be a number, list, or tuple of numbers.");
        return NIL_VAL;
    }

    dtype = resolveDTypeArg(argCount >= 3 ? args[2] : NIL_VAL);
    device = resolveDeviceArg(argCount >= 4 ? args[3] : NIL_VAL);
    if (dtype == NULL || device == NULL) {
        free(shape);
        vmRaiseExceptionMessage("ValueError", "Unsupported dtype or device.");
        return NIL_VAL;
    }

    tensor = createTensorFilled(shape, rank, dtype, device, (float)AS_NUMBER(args[1]));
    free(shape);
    return tensor != NULL ? OBJ_VAL(tensor) : NIL_VAL;
}

static Value mlArangeNative(int argCount, Value* args) {
    double start = 0.0;
    double stop;
    double step = 1.0;
    int shape[1];
    int count = 0;
    int i;
    ObjTensor* tensor;

    if (argCount < 1 || argCount > 3) return NIL_VAL;
    if (!IS_NUMBER(args[0])) return NIL_VAL;

    if (argCount == 1) {
        stop = AS_NUMBER(args[0]);
    } else if (argCount == 2) {
        if (!IS_NUMBER(args[1])) return NIL_VAL;
        start = AS_NUMBER(args[0]);
        stop = AS_NUMBER(args[1]);
    } else {
        if (!IS_NUMBER(args[1]) || !IS_NUMBER(args[2])) return NIL_VAL;
        start = AS_NUMBER(args[0]);
        stop = AS_NUMBER(args[1]);
        step = AS_NUMBER(args[2]);
    }

    if (step == 0.0) {
        vmRaiseExceptionMessage("ValueError", "arange() step cannot be zero.");
        return NIL_VAL;
    }

    for (double value = start; step > 0 ? value < stop : value > stop; value += step) {
        count++;
    }

    shape[0] = count;
    tensor = createTensorFilled(shape, 1, vm.float32DType, vm.cpuDevice, 0.0f);
    if (tensor == NULL) {
        return NIL_VAL;
    }

    for (i = 0; i < count; i++) {
        tensor->data[i] = (float)(start + step * i);
    }
    syncTensorToMetal(tensor);
    return OBJ_VAL(tensor);
}

static Value mlReshapeNative(int argCount, Value* args) {
    int* shape = NULL;
    int rank = 0;
    ObjTensor* tensor;
    ObjTensor* result;

    if (argCount != 2 || !IS_TENSOR(args[0])) return NIL_VAL;
    if (!parseShapeValue(args[1], &shape, &rank)) {
        vmRaiseExceptionMessage("TypeError", "reshape() shape must be a number, list, or tuple of numbers.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(args[0]);
    if (tensorElementCount(rank, shape) != tensor->size) {
        free(shape);
        vmRaiseExceptionMessage("ValueError", "reshape() size mismatch.");
        return NIL_VAL;
    }

    result = newTensor(rank, shape, tensor->dtype, tensor->device, tensor->data);
    free(shape);
    if (result == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while reshaping tensor.");
        return NIL_VAL;
    }
    if (tensor->requiresGrad) {
        if (!ensureAutogradSupported(tensor) || !ensureAutogradSupported(result)) {
            return NIL_VAL;
        }
        result->requiresGrad = true;
        result->parentA = tensor;
        result->gradOp = TP_AUTOGRAD_RESHAPE;
    }
    return OBJ_VAL(result);
}

static Value mlCastNative(int argCount, Value* args) {
    ObjDType* dtype;
    ObjTensor* tensor;
    ObjTensor* result;

    if (argCount != 2 || !IS_TENSOR(args[0])) return NIL_VAL;
    dtype = resolveDTypeArg(args[1]);
    if (dtype == NULL) {
        vmRaiseExceptionMessage("ValueError", "Unsupported dtype.");
        return NIL_VAL;
    }

    tensor = AS_TENSOR(args[0]);
    result = newTensor(tensor->rank, tensor->shape, dtype, tensor->device, tensor->data);
    if (result == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while casting tensor.");
        return NIL_VAL;
    }
    return OBJ_VAL(result);
}

static Value mlAddNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return mlBinaryOp(args[0], args[1], TP_TENSOR_BINARY_ADD);
}

static Value mlSubNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return mlBinaryOp(args[0], args[1], TP_TENSOR_BINARY_SUB);
}

static Value mlMulNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return mlBinaryOp(args[0], args[1], TP_TENSOR_BINARY_MUL);
}

static Value mlDivNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return mlBinaryOp(args[0], args[1], TP_TENSOR_BINARY_DIV);
}

static Value mlSumNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    return tensorReduceNumber(AS_TENSOR(args[0]), "sum");
}

static Value mlMeanNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    return tensorReduceNumber(AS_TENSOR(args[0]), "mean");
}

static Value mlMaxNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    return tensorReduceNumber(AS_TENSOR(args[0]), "max");
}

static Value mlMatmulNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return tensorMatmul(args[0], args[1]);
}

static Value mlConv2dNative(int argCount, Value* args) {
    if (argCount != 2) return NIL_VAL;
    return tensorConv2d(args[0], args[1]);
}

static Value mlToListNative(int argCount, Value* args) {
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    if (!syncTensorFromMetal(AS_TENSOR(args[0]))) {
        vmRaiseExceptionMessage("RuntimeError", "Failed to synchronize Metal tensor.");
        return NIL_VAL;
    }
    return tensorToListValue(AS_TENSOR(args[0]));
}

static Value mlReluNative(int argCount, Value* args) {
    ObjTensor* out;
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    out = tensorUnaryOp(AS_TENSOR(args[0]), TP_TENSOR_UNARY_RELU);
    return out != NULL ? OBJ_VAL(out) : NIL_VAL;
}

static Value mlSigmoidNative(int argCount, Value* args) {
    ObjTensor* out;
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    out = tensorUnaryOp(AS_TENSOR(args[0]), TP_TENSOR_UNARY_SIGMOID);
    return out != NULL ? OBJ_VAL(out) : NIL_VAL;
}

static Value mlTanhNative(int argCount, Value* args) {
    ObjTensor* out;
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    out = tensorUnaryOp(AS_TENSOR(args[0]), TP_TENSOR_UNARY_TANH);
    return out != NULL ? OBJ_VAL(out) : NIL_VAL;
}

static Value mlGeluNative(int argCount, Value* args) {
    ObjTensor* out;
    if (argCount != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    out = tensorUnaryOp(AS_TENSOR(args[0]), TP_TENSOR_UNARY_GELU);
    return out != NULL ? OBJ_VAL(out) : NIL_VAL;
}

static Value mlSoftmaxNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    return tensorSoftmax(args[0]);
}

static Value mlLayerNormNative(int argCount, Value* args) {
    float eps = 1e-5f;
    if (argCount < 1 || argCount > 2) return NIL_VAL;
    if (argCount == 2) {
        if (!IS_NUMBER(args[1])) return NIL_VAL;
        eps = (float)AS_NUMBER(args[1]);
    }
    return tensorLayerNorm(args[0], eps);
}

static Value mlParameterNative(int argCount, Value* args) {
    Value value;

    if (argCount < 1 || argCount > 3) return NIL_VAL;
    if (argCount == 1) {
        value = mlTensorNative(1, args);
    } else if (argCount == 2) {
        value = mlTensorNative(2, args);
    } else {
        value = mlTensorNative(3, args);
    }

    if (IS_TENSOR(value)) {
        ObjTensor* tensor = AS_TENSOR(value);
        if (!ensureAutogradSupported(tensor)) {
            return NIL_VAL;
        }
        tensor->requiresGrad = true;
    }
    return value;
}

static Value mlMseLossNative(int argCount, Value* args) {
    ObjTensor* pred;
    ObjTensor* target;
    ObjTensor* loss;
    float total = 0.0f;
    int i;

    if (argCount != 2 || !IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL;
    pred = AS_TENSOR(args[0]);
    target = AS_TENSOR(args[1]);
    if (!tensorShapeEquals(pred, target)) {
        vmRaiseExceptionMessage("ValueError", "mse_loss() expects tensors with matching shapes.");
        return NIL_VAL;
    }
    if ((pred->requiresGrad || target->requiresGrad) &&
        (!ensureAutogradSupported(pred) || !ensureAutogradSupported(target))) {
        return NIL_VAL;
    }

    loss = newScalarTensor(0.0f, vm.cpuDevice);
    if (loss == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NIL_VAL;
    }
    for (i = 0; i < pred->size; i++) {
        float delta = pred->data[i] - target->data[i];
        total += delta * delta;
    }
    loss->data[0] = pred->size > 0 ? total / (float)pred->size : 0.0f;
    loss->requiresGrad = pred->requiresGrad || target->requiresGrad;
    loss->parentA = pred;
    loss->parentB = target;
    loss->gradOp = TP_AUTOGRAD_MSE_LOSS;
    loss->gradAux = pred->size > 0 ? (2.0f / (float)pred->size) : 0.0f;
    return OBJ_VAL(loss);
}

static Value tensorConv2d(Value inputValue, Value weightValue) {
    ObjTensor* input;
    ObjTensor* weight;
    ObjTensor* out;
    int outShape[4];
    int batch;
    int inChannels;
    int inHeight;
    int inWidth;
    int outChannels;
    int kernelChannels;
    int kernelHeight;
    int kernelWidth;
    int outHeight;
    int outWidth;
    int n;
    int oc;
    int oy;
    int ox;
    int ic;
    int ky;
    int kx;

    if (!IS_TENSOR(inputValue) || !IS_TENSOR(weightValue)) {
        vmRaiseExceptionMessage("TypeError", "conv2d() expects two tensors.");
        return NIL_VAL;
    }

    input = AS_TENSOR(inputValue);
    weight = AS_TENSOR(weightValue);

    if (input->rank != 4 || weight->rank != 4) {
        vmRaiseExceptionMessage("ValueError", "conv2d() currently expects input [N,C,H,W] and weight [O,C,KH,KW].");
        return NIL_VAL;
    }
    if (!tensorOnCpuForTraining(input) || !tensorOnCpuForTraining(weight)) {
        vmRaiseExceptionMessage("RuntimeError", "conv2d() currently supports only CPU tensors.");
        return NIL_VAL;
    }

    batch = input->shape[0];
    inChannels = input->shape[1];
    inHeight = input->shape[2];
    inWidth = input->shape[3];
    outChannels = weight->shape[0];
    kernelChannels = weight->shape[1];
    kernelHeight = weight->shape[2];
    kernelWidth = weight->shape[3];

    if (inChannels != kernelChannels) {
        vmRaiseExceptionMessage("ValueError", "conv2d() channel mismatch.");
        return NIL_VAL;
    }
    if (kernelHeight <= 0 || kernelWidth <= 0 || kernelHeight > inHeight || kernelWidth > inWidth) {
        vmRaiseExceptionMessage("ValueError", "conv2d() invalid kernel shape.");
        return NIL_VAL;
    }

    outHeight = inHeight - kernelHeight + 1;
    outWidth = inWidth - kernelWidth + 1;
    outShape[0] = batch;
    outShape[1] = outChannels;
    outShape[2] = outHeight;
    outShape[3] = outWidth;

    out = createTensorFromShape(4, outShape, vm.cpuDevice);
    if (out == NULL) {
        vmRaiseExceptionMessage("RuntimeError", "Out of memory while creating tensor.");
        return NIL_VAL;
    }

    for (n = 0; n < batch; n++) {
        for (oc = 0; oc < outChannels; oc++) {
            for (oy = 0; oy < outHeight; oy++) {
                for (ox = 0; ox < outWidth; ox++) {
                    float total = 0.0f;
                    for (ic = 0; ic < inChannels; ic++) {
                        for (ky = 0; ky < kernelHeight; ky++) {
                            for (kx = 0; kx < kernelWidth; kx++) {
                                int inIndex = ((n * inChannels + ic) * inHeight + (oy + ky)) * inWidth + (ox + kx);
                                int weightIndex = ((oc * inChannels + ic) * kernelHeight + ky) * kernelWidth + kx;
                                total += input->data[inIndex] * weight->data[weightIndex];
                            }
                        }
                    }
                    out->data[((n * outChannels + oc) * outHeight + oy) * outWidth + ox] = total;
                }
            }
        }
    }

    if (!setAutogradBinary(out, input, weight, TP_AUTOGRAD_CONV2D)) {
        return NIL_VAL;
    }
    return OBJ_VAL(out);
}

static Value mlBackwardNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    return tensorBackward(args[0]);
}

static Value mlZeroGradTensorNative(int argCount, Value* args) {
    if (argCount != 1) return NIL_VAL;
    return tensorZeroGrad(args[0]);
}

static Value mlZeroGradNative(int argCount, Value* args) {
    ObjList* params;
    int i;

    if (argCount != 1) return NIL_VAL;
    params = newList();
    if (!collectTensorValues(args[0], params)) {
        vmRaiseExceptionMessage("TypeError", "zero_grad() expects a tensor or nested list/tuple of tensors.");
        return NIL_VAL;
    }
    for (i = 0; i < params->items.count; i++) {
        tensorZeroGrad(params->items.values[i]);
    }
    return NIL_VAL;
}

static Value mlSgdStepNative(int argCount, Value* args) {
    ObjList* params;
    double lr;
    int i;
    int j;

    if (argCount != 2 || !IS_NUMBER(args[1])) return NIL_VAL;
    params = newList();
    if (!collectTensorValues(args[0], params)) {
        vmRaiseExceptionMessage("TypeError", "sgd_step() expects a tensor or nested list/tuple of tensors.");
        return NIL_VAL;
    }
    lr = AS_NUMBER(args[1]);
    for (i = 0; i < params->items.count; i++) {
        if (!IS_TENSOR(params->items.values[i])) {
            continue;
        }
        ObjTensor* tensor = AS_TENSOR(params->items.values[i]);
        if (!tensorOnCpuForTraining(tensor)) {
            vmRaiseExceptionMessage("RuntimeError", "sgd_step() currently supports only CPU tensors.");
            return NIL_VAL;
        }
        if (tensor->grad == NULL) {
            continue;
        }
        for (j = 0; j < tensor->size; j++) {
            tensor->data[j] -= (float)lr * tensor->grad->data[j];
        }
        syncTensorToMetal(tensor);
    }
    return NIL_VAL;
}

static Value mlAdamStepNative(int argCount, Value* args) {
    ObjList* params;
    ObjList* mList;
    ObjList* vList;
    double lr;
    double beta1;
    double beta2;
    double eps;
    double step;
    double beta1Pow;
    double beta2Pow;
    double stepSize;
    int i;
    int j;

    if (argCount != 8 ||
        !IS_NUMBER(args[3]) ||
        !IS_NUMBER(args[4]) ||
        !IS_NUMBER(args[5]) ||
        !IS_NUMBER(args[6]) ||
        !IS_NUMBER(args[7])) {
        return NIL_VAL;
    }

    params = newList();
    mList = newList();
    vList = newList();
    if (!collectTensorValues(args[0], params) ||
        !collectTensorValues(args[1], mList) ||
        !collectTensorValues(args[2], vList)) {
        vmRaiseExceptionMessage("TypeError", "adam_step() expects tensor collections for params, m, and v.");
        return NIL_VAL;
    }
    if (params->items.count != mList->items.count || params->items.count != vList->items.count) {
        vmRaiseExceptionMessage("ValueError", "adam_step() state size mismatch.");
        return NIL_VAL;
    }

    lr = AS_NUMBER(args[3]);
    beta1 = AS_NUMBER(args[4]);
    beta2 = AS_NUMBER(args[5]);
    eps = AS_NUMBER(args[6]);
    step = AS_NUMBER(args[7]);
    beta1Pow = pow(beta1, step);
    beta2Pow = pow(beta2, step);
    stepSize = lr * sqrt(1.0 - beta2Pow) / (1.0 - beta1Pow);

    for (i = 0; i < params->items.count; i++) {
        ObjTensor* tensor;
        ObjTensor* m;
        ObjTensor* v;

        if (!IS_TENSOR(params->items.values[i]) ||
            !IS_TENSOR(mList->items.values[i]) ||
            !IS_TENSOR(vList->items.values[i])) {
            continue;
        }

        tensor = AS_TENSOR(params->items.values[i]);
        m = AS_TENSOR(mList->items.values[i]);
        v = AS_TENSOR(vList->items.values[i]);

        if (!tensorOnCpuForTraining(tensor) ||
            !tensorOnCpuForTraining(m) ||
            !tensorOnCpuForTraining(v)) {
            vmRaiseExceptionMessage("RuntimeError", "adam_step() currently supports only CPU tensors.");
            return NIL_VAL;
        }
        if (!tensorShapeEquals(tensor, m) || !tensorShapeEquals(tensor, v)) {
            vmRaiseExceptionMessage("ValueError", "adam_step() state tensor shape mismatch.");
            return NIL_VAL;
        }
        if (tensor->grad == NULL) {
            continue;
        }

        for (j = 0; j < tensor->size; j++) {
            float grad = tensor->grad->data[j];
            m->data[j] = (float)(beta1 * m->data[j] + (1.0 - beta1) * grad);
            v->data[j] = (float)(beta2 * v->data[j] + (1.0 - beta2) * grad * grad);
            tensor->data[j] -= (float)(stepSize * m->data[j] / (sqrt(v->data[j]) + eps));
        }
        syncTensorToMetal(tensor);
        syncTensorToMetal(m);
        syncTensorToMetal(v);
    }
    return NIL_VAL;
}

static void defineNative(const char* name, NativeFn function) {
    Value key = OBJ_VAL(copyString(name, (int)strlen(name)));
    Value native = OBJ_VAL(newNative(function));
    tableSet(vm.globalEnv->table, key, native);
}

static void defineValue(const char* name, Value value) {
    Value key = OBJ_VAL(copyString(name, (int)strlen(name)));
    tableSet(vm.globalEnv->table, key, value);
}

static void defineModuleNative(ObjInstance* module, const char* name, NativeFn function) {
    tableSet(&module->fields,
             OBJ_VAL(copyString(name, (int)strlen(name))),
             OBJ_VAL(newNative(function)));
}

static void defineBuiltinMlModule(void) {
    ObjString* moduleName = copyString("ml", 2);
    ObjInstance* module = newInstance(vm.moduleClass);

    tableSet(&module->fields, OBJ_VAL(copyString("__name__", 8)), OBJ_VAL(moduleName));
    tableSet(&module->fields, OBJ_VAL(copyString("cpu", 3)), OBJ_VAL(vm.cpuDevice));
    tableSet(&module->fields, OBJ_VAL(copyString("metal", 5)), OBJ_VAL(vm.metalDevice));
    tableSet(&module->fields, OBJ_VAL(copyString("float32", 7)), OBJ_VAL(vm.float32DType));
    defineModuleNative(module, "device", mlDeviceNative);
    defineModuleNative(module, "dtype", mlDTypeNative);
    defineModuleNative(module, "metal_available", mlMetalAvailableNative);
    defineModuleNative(module, "tensor", mlTensorNative);
    defineModuleNative(module, "Parameter", mlParameterNative);
    defineModuleNative(module, "zeros", mlZerosNative);
    defineModuleNative(module, "ones", mlOnesNative);
    defineModuleNative(module, "full", mlFullNative);
    defineModuleNative(module, "arange", mlArangeNative);
    defineModuleNative(module, "reshape", mlReshapeNative);
    defineModuleNative(module, "cast", mlCastNative);
    defineModuleNative(module, "add", mlAddNative);
    defineModuleNative(module, "sub", mlSubNative);
    defineModuleNative(module, "mul", mlMulNative);
    defineModuleNative(module, "div", mlDivNative);
    defineModuleNative(module, "sum", mlSumNative);
    defineModuleNative(module, "mean", mlMeanNative);
    defineModuleNative(module, "max", mlMaxNative);
    defineModuleNative(module, "matmul", mlMatmulNative);
    defineModuleNative(module, "conv2d", mlConv2dNative);
    defineModuleNative(module, "tolist", mlToListNative);
    defineModuleNative(module, "relu", mlReluNative);
    defineModuleNative(module, "tanh", mlTanhNative);
    defineModuleNative(module, "sigmoid", mlSigmoidNative);
    defineModuleNative(module, "gelu", mlGeluNative);
    defineModuleNative(module, "softmax", mlSoftmaxNative);
    defineModuleNative(module, "layernorm", mlLayerNormNative);
    defineModuleNative(module, "mse_loss", mlMseLossNative);
    defineModuleNative(module, "backward", mlBackwardNative);
    defineModuleNative(module, "zero_grad", mlZeroGradNative);
    defineModuleNative(module, "sgd_step", mlSgdStepNative);
    defineModuleNative(module, "adam_step", mlAdamStepNative);

    tableSet(&vm.modules, OBJ_VAL(moduleName), OBJ_VAL(module));
    tableSet(vm.globalEnv->table, OBJ_VAL(moduleName), OBJ_VAL(module));
}

void registerBuiltins() {
    defineNative("print", printNative);
    defineNative("float", floatNative);
    defineNative("int", intNative);
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
    defineNative("isinstance", isinstanceNative);
    defineNative("getattr", getattrNative);
    defineNative("hasattr", hasattrNative);
    defineNative("setattr", setattrNative);
    defineNative("callable", callableNative);
    defineNative("round", roundNative);
    defineNative("ord", ordNative);
    defineNative("chr", chrNative);
    defineNative("dir", dirNative);
    defineNative("__platform_time", platformTimeNative);
    defineNative("__platform_random", platformRandomNative);
    defineNative("__platform_getcwd", platformGetcwdNative);
    defineNative("__platform_name", platformNameNative);
    defineNative("__platform_read_text", platformReadTextNative);
    defineNative("__platform_read_bytes", platformReadBytesNative);
    defineNative("__platform_write_text", platformWriteTextNative);
    defineNative("__platform_write_bytes", platformWriteBytesNative);
    defineNative("__platform_getenv", platformGetenvNative);
    defineNative("__platform_system", platformSystemNative);
    defineNative("__platform_listdir", platformListdirNative);
    defineNative("__platform_exists", platformExistsNative);
    defineNative("__platform_isdir", platformIsdirNative);
    defineNative("__platform_isfile", platformIsfileNative);
    defineNative("__platform_mkdir", platformMkdirNative);
    defineNative("__platform_makedirs", platformMakedirsNative);
    defineNative("__platform_remove", platformRemoveNative);
    defineNative("__platform_rmdir", platformRmdirNative);
    defineNative("__platform_rename", platformRenameNative);
    defineNative("__gc_mark_count", gcMarkCountNative);
    defineNative("__gc_reachable_count", gcReachableCountNative);
    defineNative("__gc_collect", gcCollectNative);
    defineNative("__gc_object_count", gcObjectCountNative);
    defineNative("__math_sqrt", mathSqrtNative);
    defineNative("__math_sin", mathSinNative);
    defineNative("__math_cos", mathCosNative);
    defineNative("__math_tan", mathTanNative);
    defineNative("__math_floor", mathFloorNative);
    defineNative("__math_ceil", mathCeilNative);
    defineNative("__ml_device", mlDeviceNative);
    defineNative("__ml_dtype", mlDTypeNative);
    defineNative("__ml_metal_available", mlMetalAvailableNative);
    defineNative("__ml_tensor", mlTensorNative);
    defineNative("__ml_parameter", mlParameterNative);
    defineNative("__ml_zeros", mlZerosNative);
    defineNative("__ml_ones", mlOnesNative);
    defineNative("__ml_full", mlFullNative);
    defineNative("__ml_arange", mlArangeNative);
    defineNative("__ml_reshape", mlReshapeNative);
    defineNative("__ml_cast", mlCastNative);
    defineNative("__ml_add", mlAddNative);
    defineNative("__ml_sub", mlSubNative);
    defineNative("__ml_mul", mlMulNative);
    defineNative("__ml_div", mlDivNative);
    defineNative("__ml_sum", mlSumNative);
    defineNative("__ml_mean", mlMeanNative);
    defineNative("__ml_max", mlMaxNative);
    defineNative("__ml_matmul", mlMatmulNative);
    defineNative("__ml_conv2d", mlConv2dNative);
    defineNative("__ml_tolist", mlToListNative);
    defineNative("__ml_relu", mlReluNative);
    defineNative("__ml_tanh", mlTanhNative);
    defineNative("__ml_sigmoid", mlSigmoidNative);
    defineNative("__ml_gelu", mlGeluNative);
    defineNative("__ml_softmax", mlSoftmaxNative);
    defineNative("__ml_layernorm", mlLayerNormNative);
    defineNative("__ml_mse_loss", mlMseLossNative);
    defineNative("__ml_backward", mlBackwardNative);
    defineNative("__ml_zero_grad", mlZeroGradNative);
    defineNative("__ml_zero_grad_tensor", mlZeroGradTensorNative);
    defineNative("__ml_sgd_step", mlSgdStepNative);
    defineNative("__ml_adam_step", mlAdamStepNative);
    defineValue("__ml_cpu", OBJ_VAL(vm.cpuDevice));
    defineValue("__ml_metal", OBJ_VAL(vm.metalDevice));
    defineValue("__ml_float32", OBJ_VAL(vm.float32DType));
    defineBuiltinMlModule();
}
