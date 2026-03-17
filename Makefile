CC = clang
CFLAGS = -Wall -Wextra -Iinclude -O2
SRC = src/object.c src/value.c src/builtins.c src/platform.c src/table.c src/chunk.c src/vm.c src/debug.c src/scanner.c src/compiler.c src/main.c
OBJ = $(SRC:.c=.o)
TARGET = tensorpy

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
