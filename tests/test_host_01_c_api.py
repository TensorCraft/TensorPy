import os
import shutil
import subprocess
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    compiler = os.environ.get("CC", "clang")
    with tempfile.TemporaryDirectory(prefix="tensorpy-capi-") as tmpdir:
        host_path = os.path.join(tmpdir, "host.c")
        binary_path = os.path.join(tmpdir, "host")
        host_source = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorpy/api.h"

#if TENSORPY_API_VERSION != 1
#error Unexpected TensorPy API version
#endif

#if TENSORPY_EXTENSION_ABI_VERSION != 1
#error Unexpected TensorPy extension ABI version
#endif

static TPResult host_join_tag(TPContext* context, int argCount, const TPValue* args, TPValue* result, void* userData) {
    const char* tag = (const char*)userData;
    char buffer[128];
    (void)context;

    if (argCount != 2 || args[0].type != TP_VALUE_STRING || args[1].type != TP_VALUE_STRING) {
        tpValueSetError(result, TP_RUNTIME_ERROR, "join_tag expects two strings");
        return TP_RUNTIME_ERROR;
    }

    snprintf(buffer, sizeof(buffer), "%s:%s:%s", tag, args[0].string, args[1].string);
    if (!tpValueSetString(result, buffer)) {
        return TP_RUNTIME_ERROR;
    }
    return TP_OK;
}

static TPResult host_always_fail(TPContext* context, int argCount, const TPValue* args, TPValue* result, void* userData) {
    (void)context;
    (void)argCount;
    (void)args;
    (void)userData;
    tpValueSetException(result, "ValueError", "host native boom");
    return TP_RUNTIME_ERROR;
}

static TPResult host_describe(TPContext* context, int argCount, const TPValue* args, TPValue* result, void* userData) {
    char buffer[128];
    const char* prefix = (const char*)userData;
    (void)context;

    if (argCount != 1 || args[0].type != TP_VALUE_NUMBER) {
        tpValueSetException(result, "TypeError", "describe expects one number");
        return TP_RUNTIME_ERROR;
    }

    snprintf(buffer, sizeof(buffer), "%s=%.0f", prefix, args[0].number);
    if (!tpValueSetString(result, buffer)) {
        return TP_RUNTIME_ERROR;
    }
    return TP_OK;
}

int main(void) {
    TPContext* first = tpContextCreate();
    TPContext* second;
    TPModule* module;
    TPResult result;
    TPValue value;

    if (first == NULL) {
        fprintf(stderr, "failed to create first context\n");
        return 1;
    }
    if (tpApiVersion() != TENSORPY_API_VERSION || tpExtensionAbiVersion() != TENSORPY_EXTENSION_ABI_VERSION) {
        fprintf(stderr, "public API version helpers do not match header macros\n");
        return 24;
    }

    tpValueInit(&value);

    second = tpContextCreate();
    if (second != NULL) {
        fprintf(stderr, "unexpected second active context\n");
        return 2;
    }

    result = tpContextInterpret(first, "print(1 + 2)", "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to interpret in first context\n");
        return 3;
    }

    tpValueSetNumber(&value, 12);
    if (!tpContextSetGlobal(first, "seed", &value)) {
        fprintf(stderr, "failed to seed number global\n");
        return 4;
    }
    tpValueSetString(&value, "hello");
    if (!tpContextSetGlobal(first, "greeting", &value)) {
        fprintf(stderr, "failed to seed string global\n");
        return 5;
    }
    tpValueSetBool(&value, true);
    if (!tpContextSetGlobal(first, "flag", &value)) {
        fprintf(stderr, "failed to seed bool global\n");
        return 6;
    }
    result = tpContextInterpret(first, "answer = seed + 5\nbanner = greeting + \" world\"\nflag_copy = flag", "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to interpret seeded globals\n");
        return 7;
    }
    if (!tpContextGetGlobal(first, "answer", &value) || value.type != TP_VALUE_NUMBER || value.number != 17) {
        fprintf(stderr, "failed to read numeric global\n");
        return 8;
    }
    if (!tpContextGetGlobal(first, "banner", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "hello world") != 0) {
        fprintf(stderr, "failed to read string global\n");
        return 9;
    }
    value.string[0] = 'H';
    if (!tpContextGetGlobal(first, "banner", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "hello world") != 0) {
        fprintf(stderr, "global string was not returned as an owned copy\n");
        return 25;
    }
    if (!tpContextGetGlobal(first, "flag_copy", &value) || value.type != TP_VALUE_BOOL || !value.boolean) {
        fprintf(stderr, "failed to read bool global\n");
        return 10;
    }
    result = tpContextInterpret(first, "raise ValueError(\"boom\")", "embed");
    if (result != TP_RUNTIME_ERROR) {
        fprintf(stderr, "expected runtime error result\n");
        return 11;
    }
    if (!tpContextGetLastError(first, &value) || value.type != TP_VALUE_ERROR ||
        value.exceptionType == NULL ||
        strcmp(value.exceptionType, "ValueError") != 0 ||
        strstr(value.string, "ValueError: boom") == NULL) {
        fprintf(stderr, "failed to read last error\n");
        return 12;
    }

    module = tpContextCreateModule(first, "hostdemo");
    if (module == NULL) {
        fprintf(stderr, "failed to create host module\n");
        return 13;
    }
    tpValueSetString(&value, "from-c-module");
    if (!tpModuleAddValue(module, "label", &value)) {
        fprintf(stderr, "failed to add module value\n");
        return 14;
    }
    tpValueSetString(&value, "mutable-seed");
    if (!tpModuleAddValue(module, "copied_label", &value)) {
        fprintf(stderr, "failed to add copied module value\n");
        return 26;
    }
    value.string[0] = 'X';
    if (!tpModuleAddFunction(module, "join_tag", host_join_tag, "tag")) {
        fprintf(stderr, "failed to add module function\n");
        return 15;
    }
    if (!tpModuleAddFunction(module, "always_fail", host_always_fail, NULL)) {
        fprintf(stderr, "failed to add failing module function\n");
        return 16;
    }
    if (!tpModuleAddFunction(module, "describe", host_describe, "count")) {
        fprintf(stderr, "failed to add describe module function\n");
        return 27;
    }
    tpModuleDestroy(module);
    result = tpContextInterpret(
        first,
        "import hostdemo\n"
        "mod_ok = hostdemo.label\n"
        "mod_copy_ok = hostdemo.copied_label\n"
        "join_ok = hostdemo.join_tag(\"A\", \"B\")\n"
        "describe_ok = hostdemo.describe(7)\n",
        "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to import host module\n");
        return 17;
    }
    if (!tpContextGetGlobal(first, "mod_ok", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "from-c-module") != 0) {
        fprintf(stderr, "failed to read imported module value\n");
        return 18;
    }
    if (!tpContextGetGlobal(first, "join_ok", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "tag:A:B") != 0) {
        fprintf(stderr, "failed to read imported module function result\n");
        return 19;
    }
    if (!tpContextGetGlobal(first, "mod_copy_ok", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "mutable-seed") != 0) {
        fprintf(stderr, "module value was not copied into runtime storage\n");
        return 28;
    }
    if (!tpContextGetGlobal(first, "describe_ok", &value) || value.type != TP_VALUE_STRING ||
        strcmp(value.string, "count=7") != 0) {
        fprintf(stderr, "failed to read describe function result\n");
        return 29;
    }
    result = tpContextInterpret(first, "import hostdemo\nhostdemo.always_fail()", "embed");
    if (result != TP_RUNTIME_ERROR) {
        fprintf(stderr, "expected failing host native call\n");
        return 20;
    }
    if (!tpContextGetLastError(first, &value) || value.type != TP_VALUE_ERROR ||
        value.exceptionType == NULL ||
        strcmp(value.exceptionType, "ValueError") != 0 ||
        strstr(value.string, "host native boom") == NULL) {
        fprintf(stderr, "failed to read host native error\n");
        return 21;
    }
    result = tpContextInterpret(first, "import hostdemo\nhostdemo.describe(\"bad\")", "embed");
    if (result != TP_RUNTIME_ERROR) {
        fprintf(stderr, "expected type-checked host native failure\n");
        return 30;
    }
    if (!tpContextGetLastError(first, &value) || value.type != TP_VALUE_ERROR ||
        value.exceptionType == NULL ||
        strcmp(value.exceptionType, "TypeError") != 0 ||
        strstr(value.string, "describe expects one number") == NULL) {
        fprintf(stderr, "failed to read typed host argument error\n");
        return 31;
    }
    result = tpContextInterpret(first, "non_scalar = [1, 2, 3]", "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to create non-scalar test global\n");
        return 32;
    }
    if (tpContextGetGlobal(first, "non_scalar", &value)) {
        fprintf(stderr, "unexpectedly exported non-scalar global through scalar ABI\n");
        return 33;
    }

    tpContextDestroy(first);

    first = tpContextCreate();
    if (first == NULL) {
        fprintf(stderr, "failed to recreate context after destroy\n");
        return 22;
    }

    result = tpContextInterpret(first, "print(4 + 5)", "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to interpret in recreated context\n");
        return 23;
    }

    tpContextDestroy(first);
    tpValueFree(&value);
    printf("c-api-ok\n");
    return 0;
}
'''
        with open(host_path, "w", encoding="utf-8") as handle:
            handle.write(host_source)

        compile_cmd = [
            compiler,
            "-Wall",
            "-Wextra",
            "-Iinclude",
            "-O2",
            host_path,
            "src/api.c",
            "src/object.c",
            "src/value.c",
            "src/builtins.c",
            "src/platform.c",
            "src/table.c",
            "src/chunk.c",
            "src/vm.c",
            "src/debug.c",
            "src/scanner.c",
            "src/compiler.c",
            "-o",
            binary_path,
        ]
        subprocess.run(compile_cmd, cwd=ROOT, check=True, capture_output=True, text=True)

        result = subprocess.run([binary_path], cwd=ROOT, check=True, capture_output=True, text=True)
        output = result.stdout
        if "3" not in output or "9" not in output or "c-api-ok" not in output:
            raise RuntimeError("unexpected host output: " + output)


if __name__ == "__main__":
    if shutil.which(os.environ.get("CC", "clang")) is None:
        raise RuntimeError("C compiler not found for embedding API test")
    main()
