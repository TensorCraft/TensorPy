#include <stdio.h>
#include <stdint.h>

#include "tensorpy/chunk.h"
#include "tensorpy/debug.h"
#include "tensorpy/value.h"

void disassembleChunk(Chunk* chunk, const char* name) {
    printf("== %s ==\n", name);

    for (int offset = 0; offset < chunk->count;) {
        offset = disassembleInstruction(chunk, offset);
    }
}

static int simpleInstruction(const char* name, int offset) {
    printf("%s\n", name);
    return offset + 1;
}

static int byteInstruction(const char* name, Chunk* chunk, int offset) {
    uint8_t slot = chunk->code[offset + 1];
    printf("%-16s %4d\n", name, slot);
    return offset + 2;
}

static int jumpInstruction(const char* name, int sign, Chunk* chunk, int offset) {
    uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
    jump |= chunk->code[offset + 2];
    printf("%-16s %4d -> %d\n", name, offset, offset + 3 + sign * jump);
    return offset + 3;
}

static int constantInstruction(const char* name, Chunk* chunk, int offset) {
    uint8_t constant = chunk->code[offset + 1];
    printf("%-16s %4d '", name, constant);
    printValue(chunk->constants.values[constant]);
    printf("'\n");
    return offset + 2;
}

static int invokeInstruction(const char* name, Chunk* chunk, int offset) {
    uint8_t constant = chunk->code[offset + 1];
    uint8_t argCount = chunk->code[offset + 2];
    printf("%-16s (%d args) %4d '", name, argCount, constant);
    printValue(chunk->constants.values[constant]);
    printf("'\n");
    return offset + 3;
}

static int exceptInstruction(const char* name, Chunk* chunk, int offset) {
    uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
    jump |= chunk->code[offset + 2];
    uint8_t constant = chunk->code[offset + 3];
    printf("%-16s %4d -> %d", name, offset, offset + 4 + jump);
    if (constant != UINT8_MAX) {
        printf(" type=");
        printValue(chunk->constants.values[constant]);
    }
    printf("\n");
    return offset + 4;
}

int disassembleInstruction(Chunk* chunk, int offset) {
    printf("%04d ", offset);
    if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
        printf("   | ");
    } else {
        printf("%4d ", chunk->lines[offset]);
    }

    uint8_t instruction = chunk->code[offset];
    switch (instruction) {
        case OP_CONSTANT:
            return constantInstruction("OP_CONSTANT", chunk, offset);
        case OP_NIL:
            return simpleInstruction("OP_NIL", offset);
        case OP_TRUE:
            return simpleInstruction("OP_TRUE", offset);
        case OP_FALSE:
            return simpleInstruction("OP_FALSE", offset);
        case OP_POP:
            return simpleInstruction("OP_POP", offset);
        case OP_GET_LOCAL:
            return byteInstruction("OP_GET_LOCAL", chunk, offset);
        case OP_SET_LOCAL:
            return byteInstruction("OP_SET_LOCAL", chunk, offset);
        case OP_GET_GLOBAL:
            return constantInstruction("OP_GET_GLOBAL", chunk, offset);
        case OP_DEFINE_GLOBAL:
            return constantInstruction("OP_DEFINE_GLOBAL", chunk, offset);
        case OP_SET_GLOBAL:
            return constantInstruction("OP_SET_GLOBAL", chunk, offset);
        case OP_UNPACK:
            return byteInstruction("OP_UNPACK", chunk, offset);
        case OP_EQUAL:
            return simpleInstruction("OP_EQUAL", offset);
        case OP_GREATER:
            return simpleInstruction("OP_GREATER", offset);
        case OP_LESS:
            return simpleInstruction("OP_LESS", offset);
        case OP_ADD:
            return simpleInstruction("OP_ADD", offset);
        case OP_SUBTRACT:
            return simpleInstruction("OP_SUBTRACT", offset);
        case OP_MULTIPLY:
            return simpleInstruction("OP_MULTIPLY", offset);
        case OP_DIVIDE:
            return simpleInstruction("OP_DIVIDE", offset);
        case OP_NOT:
            return simpleInstruction("OP_NOT", offset);
        case OP_NEGATE:
            return simpleInstruction("OP_NEGATE", offset);
        case OP_PRINT:
            return simpleInstruction("OP_PRINT", offset);
        case OP_JUMP:
            return jumpInstruction("OP_JUMP", 1, chunk, offset);
        case OP_JUMP_IF_FALSE:
            return jumpInstruction("OP_JUMP_IF_FALSE", 1, chunk, offset);
        case OP_LOOP:
            return jumpInstruction("OP_LOOP", -1, chunk, offset);
        case OP_CALL:
            return byteInstruction("OP_CALL", chunk, offset);
        case OP_RETURN:
            return simpleInstruction("OP_RETURN", offset);
        case OP_MODULO: return simpleInstruction("OP_MODULO", offset);
        case OP_FLOOR_DIVIDE: return simpleInstruction("OP_FLOOR_DIVIDE", offset);
        case OP_POWER: return simpleInstruction("OP_POWER", offset);
        case OP_BIT_AND: return simpleInstruction("OP_BIT_AND", offset);
        case OP_BIT_OR: return simpleInstruction("OP_BIT_OR", offset);
        case OP_BIT_XOR: return simpleInstruction("OP_BIT_XOR", offset);
        case OP_BIT_NOT: return simpleInstruction("OP_BIT_NOT", offset);
        case OP_SHIFT_LEFT: return simpleInstruction("OP_SHIFT_LEFT", offset);
        case OP_SHIFT_RIGHT: return simpleInstruction("OP_SHIFT_RIGHT", offset);
        case OP_BUILD_LIST: return byteInstruction("OP_BUILD_LIST", chunk, offset);
        case OP_BUILD_TUPLE: return byteInstruction("OP_BUILD_TUPLE", chunk, offset);
        case OP_BUILD_SET: return byteInstruction("OP_BUILD_SET", chunk, offset);
        case OP_BUILD_DICT: return byteInstruction("OP_BUILD_DICT", chunk, offset);
        case OP_BUILD_SLICE: return simpleInstruction("OP_BUILD_SLICE", offset);
        case OP_GET_ITER: return simpleInstruction("OP_GET_ITER", offset);
        case OP_FOR_ITER: return jumpInstruction("OP_FOR_ITER", 1, chunk, offset);
        case OP_SUBSCRIPT: return simpleInstruction("OP_SUBSCRIPT", offset);
        case OP_STORE_SUBSCRIPT: return simpleInstruction("OP_STORE_SUBSCRIPT", offset);
        case OP_DELETE_SUBSCRIPT: return simpleInstruction("OP_DELETE_SUBSCRIPT", offset);
        case OP_CLASS: return constantInstruction("OP_CLASS", chunk, offset);
        case OP_METHOD: return constantInstruction("OP_METHOD", chunk, offset);
        case OP_GET_SUPER: return constantInstruction("OP_GET_SUPER", chunk, offset);
        case OP_SUPER_INVOKE: return invokeInstruction("OP_SUPER_INVOKE", chunk, offset);
        case OP_INVOKE: return invokeInstruction("OP_INVOKE", chunk, offset);
        case OP_GET_PROPERTY: return constantInstruction("OP_GET_PROPERTY", chunk, offset);
        case OP_IMPORT_MODULE: return constantInstruction("OP_IMPORT_MODULE", chunk, offset);
        case OP_LIST_APPEND: return byteInstruction("OP_LIST_APPEND", chunk, offset);
        case OP_SET_DEFAULTS: return byteInstruction("OP_SET_DEFAULTS", chunk, offset);
        case OP_SETUP_EXCEPT: return exceptInstruction("OP_SETUP_EXCEPT", chunk, offset);
        case OP_POP_EXCEPT: return simpleInstruction("OP_POP_EXCEPT", offset);
        case OP_RAISE: return simpleInstruction("OP_RAISE", offset);
        case OP_LIST_EXTEND: return byteInstruction("OP_LIST_EXTEND", chunk, offset);
        case OP_INVOKE_KW: return invokeInstruction("OP_INVOKE_KW", chunk, offset);
        case OP_CALL_KW: return byteInstruction("OP_CALL_KW", chunk, offset);
        case OP_SET_PROPERTY: return constantInstruction("OP_SET_PROPERTY", chunk, offset);
        default:
            printf("Unknown opcode %d\n", instruction);
            return offset + 1;
    }
}
