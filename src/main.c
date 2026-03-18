#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorpy/api.h"
#include "tensorpy/common.h"
#include "tensorpy/chunk.h"
#include "tensorpy/debug.h"
#include "tensorpy/platform.h"

static bool startsWithKeyword(const char* line, const char* keyword) {
    size_t len = strlen(keyword);
    return strncmp(line, keyword, len) == 0 &&
           (line[len] == '\0' || line[len] == ' ' || line[len] == '\t' ||
            line[len] == '\n' || line[len] == '(' || line[len] == ':');
}

static bool hasAssignment(const char* line) {
    for (size_t i = 0; line[i] != '\0'; i++) {
        if (line[i] != '=') continue;
        if (line[i + 1] == '=') {
            i++;
            continue;
        }
        if (i > 0) {
            char prev = line[i - 1];
            if (prev == '!' || prev == '<' || prev == '>' || prev == '=') {
                continue;
            }
        }
        return true;
    }
    return false;
}

static bool isLikelyExpression(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (*line == '\0' || *line == '\n') return false;

    if (startsWithKeyword(line, "if") ||
        startsWithKeyword(line, "for") ||
        startsWithKeyword(line, "while") ||
        startsWithKeyword(line, "def") ||
        startsWithKeyword(line, "class") ||
        startsWithKeyword(line, "return") ||
        startsWithKeyword(line, "break") ||
        startsWithKeyword(line, "continue") ||
        startsWithKeyword(line, "pass") ||
        startsWithKeyword(line, "try") ||
        startsWithKeyword(line, "except") ||
        startsWithKeyword(line, "else") ||
        startsWithKeyword(line, "elif") ||
        startsWithKeyword(line, "raise") ||
        startsWithKeyword(line, "del") ||
        startsWithKeyword(line, "print")) {
        return false;
    }

    if (hasAssignment(line)) return false;

    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == ' ' || line[len - 1] == '\t')) {
        len--;
    }
    if (len > 0 && line[len - 1] == ':') return false;

    return true;
}

static bool isBlankLine(const char* line) {
    while (*line != '\0') {
        if (*line != ' ' && *line != '\t' && *line != '\n' && *line != '\r') {
            return false;
        }
        line++;
    }
    return true;
}

static bool lineStartsBlock(const char* line) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == ' ' || line[len - 1] == '\t')) {
        len--;
    }
    return len > 0 && line[len - 1] == ':';
}

static void repl(TPContext* context) {
    char line[1024];
    char wrapped[1152];
    char block[8192];
    block[0] = '\0';
    bool collecting = false;
    for (;;) {
        printf(collecting ? "... " : "> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            if (collecting && block[0] != '\0') {
                tpContextInterpret(context, block, "stdin");
            }
            printf("\n");
            break;
        }

        if (collecting) {
            if (isBlankLine(line)) {
                tpContextInterpret(context, block, "stdin");
                block[0] = '\0';
                collecting = false;
                continue;
            }
            strncat(block, line, sizeof(block) - strlen(block) - 1);
            continue;
        }

        if (lineStartsBlock(line)) {
            strncat(block, line, sizeof(block) - strlen(block) - 1);
            collecting = true;
            continue;
        }

        if (isLikelyExpression(line)) {
            snprintf(wrapped, sizeof(wrapped), "print(%s)", line);
            tpContextInterpret(context, wrapped, "stdin");
        } else {
            tpContextInterpret(context, line, "stdin");
        }
    }
}

static char* readFile(const char* path) {
    char* buffer = platformReadTextFile(path);
    if (buffer == NULL) {
        fprintf(stderr, "Could not read file \"%s\".\n", path);
        exit(74);
    }
    return buffer;
}

static void runFile(TPContext* context, const char* path) {
    char* source = readFile(path);
    TPResult result = tpContextInterpret(context, source, path);
    free(source);

    if (result == TP_COMPILE_ERROR) exit(65);
    if (result == TP_RUNTIME_ERROR) exit(70);
}

int main(int argc, const char* argv[]) {
    TPContext* context = tpContextCreate();
    if (context == NULL) {
        fprintf(stderr, "Failed to initialize TensorPy context.\n");
        return 70;
    }

    if (argc == 1) {
        repl(context);
    } else if (argc == 2) {
        runFile(context, argv[1]);
    } else if (argc == 3 && strcmp(argv[1], "-c") == 0) {
        tpContextInterpret(context, argv[2], "command line");
    } else {
        fprintf(stderr, "Usage: tensorpy [path]\n");
        fprintf(stderr, "       tensorpy -c \"code\"\n");
        tpContextDestroy(context);
        exit(64);
    }

    tpContextDestroy(context);
    return 0;
}
