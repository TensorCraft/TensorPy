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

#include "tensorpy/api.h"

int main(void) {
    TPContext* first = tpContextCreate();
    TPContext* second;
    TPResult result;

    if (first == NULL) {
        fprintf(stderr, "failed to create first context\n");
        return 1;
    }

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

    tpContextDestroy(first);

    first = tpContextCreate();
    if (first == NULL) {
        fprintf(stderr, "failed to recreate context after destroy\n");
        return 4;
    }

    result = tpContextInterpret(first, "print(4 + 5)", "embed");
    if (result != TP_OK) {
        fprintf(stderr, "failed to interpret in recreated context\n");
        return 5;
    }

    tpContextDestroy(first);
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
