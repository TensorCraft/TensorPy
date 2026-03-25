CC = clang
CFLAGS = -Wall -Wextra -Iinclude -O2 -pthread
SRC = src/api.c src/object.c src/value.c src/builtins.c src/platform.c src/compute.c src/table.c src/chunk.c src/vm.c src/debug.c src/scanner.c src/compiler.c src/main.c src/memory.c
METAL ?= 1

ifeq ($(METAL),1)
OBJC_SRC = src/metal.m
CFLAGS += -DTP_ENABLE_METAL=1
LDFLAGS = -framework Foundation -framework Metal -framework Accelerate
else
SRC += src/metal_stub.c
CFLAGS += -DTP_ENABLE_METAL=0
LDFLAGS = -framework Accelerate
OBJC_SRC =
endif

OBJ = $(SRC:.c=.o) $(OBJC_SRC:.m=.o)
TARGET = tensorpy
PLATFORM_TEST = platform_concurrency_test
PLATFORM_TEST_SRC = src/platform_concurrency_test.c src/platform.c
PLATFORM_TEST_OBJ = $(PLATFORM_TEST_SRC:.c=.o)
COMPUTE_TEST = compute_test
COMPUTE_TEST_SRC = src/compute_test.c src/compute.c src/platform.c
COMPUTE_TEST_OBJ = $(COMPUTE_TEST_SRC:.c=.o) $(OBJC_SRC:.m=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(PLATFORM_TEST): $(PLATFORM_TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPUTE_TEST): $(COMPUTE_TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(PLATFORM_TEST_OBJ) $(COMPUTE_TEST_OBJ) $(TARGET) $(PLATFORM_TEST) $(COMPUTE_TEST)

run: $(TARGET)
	./$(TARGET)

test-platform: $(PLATFORM_TEST)
	./$(PLATFORM_TEST)

test-compute: $(COMPUTE_TEST)
	./$(COMPUTE_TEST)

.PHONY: all clean run test-platform test-compute
