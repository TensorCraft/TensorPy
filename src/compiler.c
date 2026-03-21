#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorpy/chunk.h"
#include "tensorpy/common.h"
#include "tensorpy/compiler.h"
#include "tensorpy/object.h"
#include "tensorpy/scanner.h"
#include "tensorpy/value.h"

#ifdef DEBUG_PRINT_CODE
#include "tensorpy/debug.h"
#endif

#include <string.h>

typedef struct {
  Token current;
  Token previous;
  bool hadError;
  bool panicMode;
} Parser;

typedef enum {
  PREC_NONE,
  PREC_ASSIGNMENT,  // =
  PREC_LAMBDA,      // lambda
  PREC_CONDITIONAL, // if-else
  PREC_OR,          // or
  PREC_AND,         // and
  PREC_BIT_OR,      // |
  PREC_BIT_XOR,     // ^
  PREC_BIT_AND,     // &
  PREC_EQUALITY,    // == !=
  PREC_COMPARISON,  // < > <= >=
  PREC_SHIFT,       // << >>
  PREC_TERM,        // + -
  PREC_FACTOR,      // * / // %
  PREC_POWER,       // **
  PREC_UNARY,       // ! - ~
  PREC_CALL,        // . ()
  PREC_PRIMARY
} Precedence;

typedef void (*ParseFn)(bool canAssign);

typedef struct {
  ParseFn prefix;
  ParseFn infix;
  Precedence precedence;
} ParseRule;

Parser parser;
typedef enum { TYPE_FUNCTION, TYPE_METHOD, TYPE_SCRIPT } FunctionType;

typedef struct {
  Token name;
  int depth;
} Local;

typedef struct Loop {
  struct Loop *enclosing;
  int start;
  int scopeDepth;
  int breakJumps[64];
  int breakCount;
  bool isFor;
} Loop;

typedef struct Compiler {
  struct Compiler *enclosing;
  ObjFunction *function;
  FunctionType type;
  Loop *loop;

  Local locals[256];
  int localCount;
  int scopeDepth;
  int maxSlots;
} Compiler;

typedef struct ClassCompiler {
  struct ClassCompiler* enclosing;
  bool hasSuperclass;
  Token superclassName;
} ClassCompiler;

Compiler *current = NULL;
static ClassCompiler* currentClass = NULL;

static Chunk *currentChunk() { return &current->function->chunk; }

static void recordLocalName(int slot, Token name) {
  while (current->function->localNames.count <= slot) {
    writeValueArray(&current->function->localNames, NIL_VAL);
  }

  if (name.length == 0) {
    current->function->localNames.values[slot] = NIL_VAL;
  } else {
    current->function->localNames.values[slot] =
        OBJ_VAL(copyString(name.start, name.length));
  }
}

static void errorAt(Token *token, const char *message) {
  if (parser.hadError)
    return;
  parser.panicMode = true;
  fprintf(stderr, "[%s:%d:%d] Error", scanner.filename, token->line,
          token->column);

  if (token->type == TOKEN_EOF) {
    fprintf(stderr, " at end");
  } else if (token->type == TOKEN_ERROR) {
    // Nothing.
  } else {
    fprintf(stderr, " at '%.*s'", token->length, token->start);
  }

  fprintf(stderr, ": %s\n", message);
  parser.hadError = true;
}

static void error(const char *message) { errorAt(&parser.previous, message); }

static void errorAtCurrent(const char *message) {
  errorAt(&parser.current, message);
}

static void advance() {
  parser.previous = parser.current;

  for (;;) {
    parser.current = scanToken();
    if (parser.current.type != TOKEN_ERROR)
      break;

    errorAtCurrent(parser.current.start);
  }
}
static bool match(TokenType type) {
  if (parser.current.type != type)
    return false;
  advance();
  return true;
}

static void consume(TokenType type, const char *message) {
  if (parser.current.type == type) {
    advance();
    return;
  }

  errorAtCurrent(message);
}

static void emitByte(uint8_t byte) {
  writeChunk(currentChunk(), byte, parser.previous.line);
}

static void emitBytes(uint8_t byte1, uint8_t byte2) {
  emitByte(byte1);
  emitByte(byte2);
}

static void emitReturn() {
  if (current->type == TYPE_SCRIPT) {
    emitByte(OP_RETURN);
  } else {
    if (current->type == TYPE_METHOD) {
      emitByte(OP_GET_LOCAL);
      emitByte(1); // Return self
    } else {
      emitByte(OP_NIL);
    }
    emitByte(OP_RETURN);
  }
}

static int emitJump(uint8_t instruction) {
  emitByte(instruction);
  emitByte(0xff);
  emitByte(0xff);
  return currentChunk()->count - 2;
}

static int emitExceptSetup(void) {
  emitByte(OP_SETUP_EXCEPT);
  emitByte(0xff);
  emitByte(0xff);
  emitByte(0xff);
  return currentChunk()->count - 3;
}

static void emitLoop(int loopStart) {
  emitByte(OP_LOOP);

  int offset = currentChunk()->count - loopStart + 2;
  if (offset > UINT16_MAX)
    error("Loop body too large.");

  emitByte((offset >> 8) & 0xff);
  emitByte(offset & 0xff);
}

static void patchJump(int offset) {
  // -2 to adjust for the bytecode for the jump offset itself.
  int jump = currentChunk()->count - offset - 2;

  if (jump > UINT16_MAX) {
    error("Too far to jump.");
  }

  currentChunk()->code[offset] = (jump >> 8) & 0xff;
  currentChunk()->code[offset + 1] = jump & 0xff;
}

static void patchExceptSetup(int offset, uint8_t typeConstant) {
  int jump = currentChunk()->count - offset - 3;

  if (jump > UINT16_MAX) {
    error("Too far to jump.");
  }

  currentChunk()->code[offset] = (jump >> 8) & 0xff;
  currentChunk()->code[offset + 1] = jump & 0xff;
  currentChunk()->code[offset + 2] = typeConstant;
}

static uint8_t makeConstant(Value value) {
  int constant = addConstant(currentChunk(), value);
  if (constant > UINT8_MAX) {
    error("Too many constants in one chunk.");
    return 0;
  }

  return (uint8_t)constant;
}

static void emitConstant(Value value) {
  emitBytes(OP_CONSTANT, makeConstant(value));
}

static void initCompiler(Compiler *compiler, FunctionType type) {
  compiler->enclosing = current;
  compiler->function = NULL;
  compiler->type = type;
  compiler->localCount = 0;
  compiler->scopeDepth = 0;
  compiler->maxSlots = 0;
  compiler->function = newFunction();
  current = compiler;

  if (type != TYPE_SCRIPT) {
    current->function->name =
        copyString(parser.previous.start, parser.previous.length);
  }

  Local *local = &current->locals[current->localCount++];
  local->depth = 0;
  if (current->type == TYPE_METHOD) {
    local->name.start = "this";
    local->name.length = 4;
  } else {
    local->name.start = "";
    local->name.length = 0;
  }
  recordLocalName(0, local->name);
}

static ObjFunction *endCompiler() {
  emitReturn();
  ObjFunction *function = current->function;
  function->maxSlots = current->maxSlots;

#ifdef DEBUG_PRINT_CODE
  if (!parser.hadError) {
    disassembleChunk(currentChunk(),
                     function->name != NULL ? function->name->chars : "code");
  }
#endif

  current = current->enclosing;
  return function;
}

static void expression();
static void statement();
static void declaration();
static ParseRule *getRule(TokenType type);
static void parsePrecedence(Precedence precedence);
static uint8_t identifierConstant(Token* name);
static void storeName(Token name);
static void addLocal(Token name);
static int resolveLocal(Compiler *compiler, Token *name);
static uint8_t dottedPathConstant(Token* first, Token* last);
static void importStatement(void);
static void fromImportStatement(void);
static void super_(bool canAssign);

static bool check(TokenType type) { return parser.current.type == type; }

static bool tokenNamesEqual(Token* a, Token* b) {
  return a->length == b->length &&
         memcmp(a->start, b->start, (size_t)a->length) == 0;
}

static bool hasDuplicateParam(Token* params, int count, Token* candidate) {
  for (int i = 0; i < count; i++) {
    if (tokenNamesEqual(&params[i], candidate)) {
      return true;
    }
  }
  return false;
}

static uint8_t identifierConstant(Token* name) {
  return makeConstant(OBJ_VAL(copyString(name->start, name->length)));
}

static uint8_t dottedPathConstant(Token* first, Token* last) {
  int length = (int)((last->start + last->length) - first->start);
  char* path = (char*)malloc((size_t)length + 1);
  memcpy(path, first->start, (size_t)length);
  path[length] = '\0';
  return makeConstant(OBJ_VAL(takeString(path, length)));
}

static void storeName(Token name) {
  int arg = resolveLocal(current, &name);
  if (arg != -1) {
    emitBytes(OP_SET_LOCAL, (uint8_t)arg);
    emitByte(OP_POP);
    return;
  }

  if (current->scopeDepth > 0) {
    addLocal(name);
    current->locals[current->localCount - 1].depth = current->scopeDepth;
    emitBytes(OP_SET_LOCAL, (uint8_t)(current->localCount - 1));
    emitByte(OP_POP);
    return;
  }

  emitBytes(OP_DEFINE_GLOBAL, identifierConstant(&name));
}

static void block() {
  while (!check(TOKEN_DEDENT) && !check(TOKEN_EOF)) {
    declaration();
  }
}

static void string(bool canAssign) {
  (void)canAssign;
  const char* start = parser.previous.start;
  int length = parser.previous.length;
  int qLen = 1;
  if (length >= 6 && start[0] == start[1] && start[0] == start[2]) {
      qLen = 3;
  }
  
  const char* raw = start + qLen;
  int rawLen = length - 2 * qLen;
  
  char* processed = (char*)malloc(rawLen + 1);
  int j = 0;
  for (int i = 0; i < rawLen; i++) {
      if (raw[i] == '\\' && i + 1 < rawLen) {
          i++;
          switch (raw[i]) {
              case 'n': processed[j++] = '\n'; break;
              case 'r': processed[j++] = '\r'; break;
              case 't': processed[j++] = '\t'; break;
              case '\\': processed[j++] = '\\'; break;
              case '\'': processed[j++] = '\''; break;
              case '\"': processed[j++] = '\"'; break;
              default: processed[j++] = '\\'; processed[j++] = raw[i]; break;
          }
      } else {
          processed[j++] = raw[i];
      }
  }
  
  emitConstant(OBJ_VAL(copyString(processed, j)));
  free(processed);
}

static void bytes(bool canAssign) {
  (void)canAssign;
  emitConstant(OBJ_VAL(newBytes(parser.previous.length - 3,
                                (const uint8_t *)parser.previous.start + 2)));
}

static void expression();
static void statement();
static void tryStatement();
static void raiseStatement();
static void declaration();
static void function(FunctionType type);
static void addLocal(Token name);
static int resolveLocal(Compiler *compiler, Token *name);
static int emitJump(uint8_t instruction);
static void patchJump(int offset);
static void block();

static void lambdaExpression(bool canAssign) {
    (void)canAssign;
    
    // 1. Parse defaults/names in the OUTER scope
    Scanner backupScanner = scanner;
    Parser backupParser = parser;
    int defaultsCount = 0;
    Token paramNames[255];
    int paramCount = 0;
    bool sawVarargs = false;
    
    if (!check(TOKEN_COLON)) {
        do {
            bool isVarargs = match(TOKEN_STAR);
            consume(TOKEN_IDENTIFIER, "Expect parameter name.");
            if (hasDuplicateParam(paramNames, paramCount, &parser.previous)) {
                error("Duplicate parameter name.");
            }
            paramNames[paramCount++] = parser.previous;
            if (isVarargs) {
                if (sawVarargs) {
                    error("Can't have more than one varargs parameter.");
                }
                if (match(TOKEN_EQUAL)) {
                    error("Varargs parameter cannot have a default value.");
                }
                sawVarargs = true;
            } else if (sawVarargs) {
                error("Parameters cannot follow varargs parameter.");
            } else if (match(TOKEN_EQUAL)) {
                expression(); // Evaluated in outer scope
                defaultsCount++;
            } else if (defaultsCount > 0) {
                error("Non-default argument follows default argument.");
            }
            if (paramCount == 255) errorAtCurrent("Too many parameters.");
        } while (match(TOKEN_COMMA));
    }
    consume(TOKEN_COLON, "Expect ':' after lambda parameters.");

    // 2. Start lambda compilation
    scanner = backupScanner;
    bool hadError = parser.hadError;
    parser = backupParser;
    if (hadError) parser.hadError = true;

    Compiler compiler;
    initCompiler(&compiler, TYPE_FUNCTION);
    // Override name
    current->function->name = copyString("<lambda>", 8);

    if (!check(TOKEN_COLON)) {
        do {
            bool isVarargs = match(TOKEN_STAR);
            consume(TOKEN_IDENTIFIER, "Expect parameter name.");
            if (!isVarargs) {
                current->function->arity++;
            } else {
                current->function->hasVarargs = true;
            }
            addLocal(parser.previous);
            current->locals[current->localCount - 1].depth = current->scopeDepth;
            if (match(TOKEN_EQUAL)) {
                if (isVarargs) {
                    error("Varargs parameter cannot have a default value.");
                }
                // Skip expression
                int bracketLevel = 0;
                while (bracketLevel > 0 || (!check(TOKEN_COMMA) && !check(TOKEN_COLON))) {
                    if (check(TOKEN_LEFT_PAREN) || check(TOKEN_LEFT_BRACKET)) bracketLevel++;
                    if (check(TOKEN_RIGHT_PAREN) || check(TOKEN_RIGHT_BRACKET)) bracketLevel--;
                    advance();
                    if (parser.previous.type == TOKEN_EOF) break;
                }
            }
        } while (match(TOKEN_COMMA));
    }
    consume(TOKEN_COLON, "Expect ':' after parameters.");
    
    // Lambda body is an expression
    parsePrecedence(PREC_LAMBDA); 
    emitByte(OP_RETURN); // Lambda returns the expression value
    
    ObjFunction* lambda = endCompiler();
    emitBytes(OP_CONSTANT, makeConstant(OBJ_VAL(lambda)));
    
    for (int i = 0; i < paramCount; i++) {
        writeValueArray(&lambda->paramNames, OBJ_VAL(copyString(paramNames[i].start, paramNames[i].length)));
    }
    
    if (defaultsCount > 0) {
        emitBytes(OP_SET_DEFAULTS, (uint8_t)defaultsCount);
    }
}

static void ternaryExpression(bool canAssign) {
  (void)canAssign;
  // We already parsed 'A' (the 'true' result).
  // Now at 'if'.

  // Parse condition 'B'.
  parsePrecedence(PREC_OR);

  int elseJump = emitJump(OP_JUMP_IF_FALSE);
  emitByte(OP_POP); // Pop condition result

  // True branch: we already have 'A' on stack.
  int endJump = emitJump(OP_JUMP);

  patchJump(elseJump);
  emitByte(OP_POP); // Pop condition result
  emitByte(OP_POP); // Pop 'A'

  consume(TOKEN_ELSE, "Expect 'else' after ternary 'if'.");
  parsePrecedence(PREC_CONDITIONAL); // Parse 'C'

  patchJump(endJump);
}

static bool lookaheadComprehension() {
  Scanner backupScanner = scanner;
  int bracketLevel = 0;
  Token token = parser.current;

  for (;;) {
    if (token.type == TOKEN_EOF) break;
    
    if (token.type == TOKEN_LEFT_BRACKET || token.type == TOKEN_LEFT_PAREN ||
        token.type == TOKEN_LEFT_BRACE) {
      bracketLevel++;
    } else if (token.type == TOKEN_RIGHT_BRACKET ||
               token.type == TOKEN_RIGHT_PAREN ||
               token.type == TOKEN_RIGHT_BRACE) {
      bracketLevel--;
    }

    if (bracketLevel == 0 && token.type == TOKEN_FOR) {
      scanner = backupScanner;
      return true;
    }
    if (bracketLevel < 0) break;
    
    token = scanToken();
  }

  scanner = backupScanner;
  return false;
}

static void list(bool canAssign) {
  (void)canAssign;
  if (lookaheadComprehension()) {
    // Comprehension: [expr for var in iterable]
    Scanner startScanner = scanner;
    Parser startParser = parser;

    // 1. Skip expr tokens to reach 'for'
    int bracketLevel = 0;
    for (;;) {
      advance();
      if (parser.previous.type == TOKEN_LEFT_BRACKET ||
          parser.previous.type == TOKEN_LEFT_PAREN)
        bracketLevel++;
      if (parser.previous.type == TOKEN_RIGHT_BRACKET ||
          parser.previous.type == TOKEN_RIGHT_PAREN)
        bracketLevel--;
      if (bracketLevel == 0 && check(TOKEN_FOR))
        break;
    }

    consume(TOKEN_FOR, "Expect 'for'.");
    Token varName = parser.current;
    consume(TOKEN_IDENTIFIER, "Expect variable name.");
    consume(TOKEN_IN, "Expect 'in'.");

    // 2. Emit list and iterable
    // Reserve slots for list and iter so other locals don't overlap
    Token dummyList = {TOKEN_IDENTIFIER, ".list", 5, 0, 0};
    addLocal(dummyList);
    emitBytes(OP_BUILD_LIST, 0);

    Token dummyIter = {TOKEN_IDENTIFIER, ".iter", 5, 0, 0};
    addLocal(dummyIter);
    expression(); // iterable
    emitByte(OP_GET_ITER);

    int loopStart = currentChunk()->count;
    int exitJump = emitJump(OP_FOR_ITER);

    // 3. Define local
    addLocal(varName);
    int slot = current->localCount - 1;
    emitBytes(OP_SET_LOCAL, (uint8_t)slot);
    emitByte(OP_POP);

    // 4. Backtrack to parse 'expr'
    Scanner forScanner = scanner;
    Parser forParser = parser;
    scanner = startScanner;
    parser = startParser;

    parsePrecedence(PREC_OR);

    // 5. Append and loop
    emitBytes(OP_LIST_APPEND, 2);
    emitLoop(loopStart);
    patchJump(exitJump);
    emitByte(OP_POP); // iterator

    // 6. Restore scanner to after comprehension
    scanner = forScanner;
    parser = forParser;
    consume(TOKEN_RIGHT_BRACKET, "Expect ']' after comprehension.");
  } else {
    // Standard list
    int count = 0;
    if (!check(TOKEN_RIGHT_BRACKET)) {
      do {
        if (check(TOKEN_RIGHT_BRACKET))
          break;
        parsePrecedence(PREC_OR);
        count++;
        if (count > 255)
          error("Too many elements in list literal.");
      } while (match(TOKEN_COMMA));
    }
    consume(TOKEN_RIGHT_BRACKET, "Expect ']' after list elements.");
    emitBytes(OP_BUILD_LIST, (uint8_t)count);
  }
}
static void dict(bool canAssign) {
  (void)canAssign;
  int count = 0;
  bool isDict = false;
  if (!check(TOKEN_RIGHT_BRACE)) {
    // We need to parse first element to decide if it's a set or dict
    parsePrecedence(PREC_OR);
    if (match(TOKEN_COLON)) {
      isDict = true;
      expression();
      count++;
      while (match(TOKEN_COMMA)) {
        if (check(TOKEN_RIGHT_BRACE))
          break;
        expression();
        consume(TOKEN_COLON, "Expect ':' after key.");
        expression();
        count++;
      }
    } else {
      // Set
      count++;
      while (match(TOKEN_COMMA)) {
        if (check(TOKEN_RIGHT_BRACE))
          break;
        expression();
        count++;
      }
    }
  } else {
    isDict = true; // {} is empty dict
  }
  consume(TOKEN_RIGHT_BRACE, "Expect '}' after elements.");
  if (isDict) {
    emitBytes(OP_BUILD_DICT, (uint8_t)count);
  } else {
    emitBytes(OP_BUILD_SET, (uint8_t)count);
  }
}

static void subscript(bool canAssign) {
  bool isSlice = false;
  if (match(TOKEN_COLON)) {
    emitConstant(NIL_VAL); // start
    isSlice = true;
  } else {
    expression(); // start or index
    if (match(TOKEN_COLON)) {
      isSlice = true;
    }
  }

  if (isSlice) {
    if (match(TOKEN_COLON)) {
      emitConstant(NIL_VAL); // stop
      if (check(TOKEN_RIGHT_BRACKET)) {
        emitConstant(NIL_VAL); // step
      } else {
        expression(); // step
      }
    } else if (check(TOKEN_RIGHT_BRACKET)) {
      emitConstant(NIL_VAL); // stop
      emitConstant(NIL_VAL); // step
    } else {
      expression(); // stop
      if (match(TOKEN_COLON)) {
        if (check(TOKEN_RIGHT_BRACKET)) {
          emitConstant(NIL_VAL); // step
        } else {
          expression(); // step
        }
      } else {
        emitConstant(NIL_VAL); // step
      }
    }
    emitByte(OP_BUILD_SLICE);
    consume(TOKEN_RIGHT_BRACKET, "Expect ']' after slice.");
  } else {
    consume(TOKEN_RIGHT_BRACKET, "Expect ']' after subscript.");
  }

  if (canAssign && match(TOKEN_EQUAL)) {
    expression();
    emitByte(OP_STORE_SUBSCRIPT);
  } else {
    emitByte(OP_SUBSCRIPT);
  }
}

static void addLocal(Token name) {
  int slot = current->localCount;
  Local *local = &current->locals[current->localCount++];
  if (current->localCount > current->maxSlots) current->maxSlots = current->localCount;
  local->name = name;
  local->depth = -1; // "uninitialized"
  recordLocalName(slot, name);
}

static int resolveLocal(Compiler *compiler, Token *name) {
  for (int i = compiler->localCount - 1; i >= 0; i--) {
    Local *local = &compiler->locals[i];
    if (name->length == local->name.length &&
        memcmp(name->start, local->name.start, name->length) == 0) {
      return i;
    }
  }
  return -1;
}

static void namedVariable(Token name, bool canAssign) {
  uint8_t getOp, setOp;
  int arg = resolveLocal(current, &name);
  if (arg != -1) {
    getOp = OP_GET_LOCAL;
    setOp = OP_SET_LOCAL;
  } else {
    getOp = OP_GET_GLOBAL;
    setOp = OP_SET_GLOBAL;
    arg = makeConstant(OBJ_VAL(copyString(name.start, name.length)));
  }

  if (canAssign &&
      (match(TOKEN_EQUAL) || match(TOKEN_PLUS_EQUAL) ||
       match(TOKEN_MINUS_EQUAL) || match(TOKEN_STAR_EQUAL) ||
       match(TOKEN_SLASH_EQUAL) || match(TOKEN_PERCENT_EQUAL) ||
       match(TOKEN_STAR_STAR_EQUAL) || match(TOKEN_SLASH_SLASH_EQUAL))) {

    TokenType op = parser.previous.type;

    if (op != TOKEN_EQUAL) {
      if (arg == -1 && current->type != TYPE_SCRIPT) {
        error("Can't use compound assignment on undefined local variable.");
      }
      emitBytes(getOp, (uint8_t)arg);
      expression();
      switch (op) {
      case TOKEN_PLUS_EQUAL:
        emitByte(OP_ADD);
        break;
      case TOKEN_MINUS_EQUAL:
        emitByte(OP_SUBTRACT);
        break;
      case TOKEN_STAR_EQUAL:
        emitByte(OP_MULTIPLY);
        break;
      case TOKEN_SLASH_EQUAL:
        emitByte(OP_DIVIDE);
        break;
      case TOKEN_PERCENT_EQUAL:
        emitByte(OP_MODULO);
        break;
      case TOKEN_STAR_STAR_EQUAL:
        emitByte(OP_POWER);
        break;
      case TOKEN_SLASH_SLASH_EQUAL:
        emitByte(OP_FLOOR_DIVIDE);
        break;
      default:
        break;
      }
    } else {
      expression();
      if (setOp == OP_SET_GLOBAL && current->type != TYPE_SCRIPT) {
        addLocal(name);
        current->locals[current->localCount - 1].depth = current->scopeDepth;
        arg = current->localCount - 1;
        setOp = OP_SET_LOCAL;
      }
    }
    emitBytes(setOp, (uint8_t)arg);
  } else {
    emitBytes(getOp, (uint8_t)arg);
  }
}

static void variable(bool canAssign) {
  namedVariable(parser.previous, canAssign);
}

static uint8_t makeStarIndexConstant(uint8_t* starIndices, uint8_t starCount) {
  ObjTuple* tuple = newTuple();
  for (int i = 0; i < starCount; i++) {
    writeValueArray(&tuple->items, NUMBER_VAL(starIndices[i]));
  }
  return makeConstant(OBJ_VAL(tuple));
}

static void argumentList(uint8_t *posCount, uint8_t *kwCount, uint8_t* kwSourceCount,
                         bool* hasStar, uint8_t* starConst,
                         bool* hasKwStar, uint8_t* kwStarConst) {
  *posCount = 0;
  *kwCount = 0;
  *kwSourceCount = 0;
  *hasStar = false;
  *starConst = 0;
  *hasKwStar = false;
  *kwStarConst = 0;
  Value kwNames[255];
  uint8_t starIndices[255];
  uint8_t starCount = 0;
  uint8_t kwStarIndices[255];
  uint8_t kwStarCount = 0;

  if (parser.current.type != TOKEN_RIGHT_PAREN) {
    do {
      if (match(TOKEN_STAR_STAR)) {
        expression();
        kwStarIndices[kwStarCount++] = *kwSourceCount;
        (*kwSourceCount)++;
        *hasKwStar = true;
        if (*posCount + *kwSourceCount == 255) {
          error("Can't have more than 255 arguments.");
        }
        continue;
      }

      if (match(TOKEN_STAR)) {
        if (*kwCount > 0)
          error("Positional argument cannot follow keyword argument.");
        expression();
        starIndices[starCount++] = *posCount;
        (*posCount)++;
        *hasStar = true;
        if (*posCount + *kwCount == 255) {
          error("Can't have more than 255 arguments.");
        }
        continue;
      }

      bool isKw = false;
      if (parser.current.type == TOKEN_IDENTIFIER) {
        // Peek ahead for '='
        Scanner backup = scanner;
        Token next = scanToken();
        scanner = backup;
        if (next.type == TOKEN_EQUAL)
          isKw = true;
      }

      if (isKw) {
        Token nameToken = parser.current;
        advance(); // identifier
        advance(); // =
        expression();
        kwNames[(*kwCount)++] =
            OBJ_VAL(copyString(nameToken.start, nameToken.length));
        (*kwSourceCount)++;
      } else {
        if (*kwCount > 0)
          error("Positional argument cannot follow keyword argument.");
        expression();
        (*posCount)++;
      }
      if (*posCount + *kwCount == 255) {
        error("Can't have more than 255 arguments.");
      }
    } while (match(TOKEN_COMMA));
  }
  consume(TOKEN_RIGHT_PAREN, "Expect ')' after arguments.");

  if (*kwCount > 0) {
    // Push kw names as constants
    for (int i = 0; i < *kwCount; i++) {
      emitConstant(kwNames[i]);
    }
  }
  if (*hasStar) {
    *starConst = makeStarIndexConstant(starIndices, starCount);
  }
  if (*hasKwStar) {
    *kwStarConst = makeStarIndexConstant(kwStarIndices, kwStarCount);
  }
}

static void call(bool canAssign) {
  (void)canAssign;
  uint8_t posCount, kwCount, kwSourceCount, starConst, kwStarConst;
  bool hasStar, hasKwStar;
  argumentList(&posCount, &kwCount, &kwSourceCount, &hasStar, &starConst, &hasKwStar, &kwStarConst);
  if (hasKwStar || (kwCount > 0 && hasStar)) {
    emitBytes(OP_CALL_EX, posCount);
    emitByte(kwSourceCount);
    emitByte(kwCount);
    emitByte(starConst);
    emitByte(kwStarConst);
  } else if (kwCount > 0) {
    emitBytes(OP_CALL_KW, posCount);
    emitByte(kwCount);
  } else if (hasStar) {
    emitBytes(OP_CALL_STAR, posCount);
    emitByte(starConst);
  } else {
    emitBytes(OP_CALL, posCount);
  }
}

static void dot(bool canAssign) {
  consume(TOKEN_IDENTIFIER, "Expect property name after '.'.");
  uint8_t name = makeConstant(
      OBJ_VAL(copyString(parser.previous.start, parser.previous.length)));

  if (canAssign && match(TOKEN_EQUAL)) {
    expression();
    emitBytes(OP_SET_PROPERTY, name);
  } else if (match(TOKEN_LEFT_PAREN)) {
    uint8_t posCount, kwCount, kwSourceCount, starConst, kwStarConst;
    bool hasStar, hasKwStar;
    argumentList(&posCount, &kwCount, &kwSourceCount, &hasStar, &starConst, &hasKwStar, &kwStarConst);
    if (hasKwStar || (kwCount > 0 && hasStar)) {
        emitBytes(OP_INVOKE_EX, name);
        emitByte(posCount);
        emitByte(kwSourceCount);
        emitByte(kwCount);
        emitByte(starConst);
        emitByte(kwStarConst);
    } else if (kwCount > 0) {
        emitBytes(OP_INVOKE_KW, name);
        emitByte(posCount);
        emitByte(kwCount);
    } else if (hasStar) {
        emitBytes(OP_INVOKE_STAR, name);
        emitByte(posCount);
        emitByte(starConst);
    } else {
        emitBytes(OP_INVOKE, name);
        emitByte(posCount);
    }
  } else {
    emitBytes(OP_GET_PROPERTY, name);
  }
}

static void and_(bool canAssign) {
  (void)canAssign;
  int endJump = emitJump(OP_JUMP_IF_FALSE);
  emitByte(OP_POP);
  parsePrecedence(PREC_AND);
  patchJump(endJump);
}

static void or_(bool canAssign) {
  (void)canAssign;
  int elseJump = emitJump(OP_JUMP_IF_FALSE);
  int endJump = emitJump(OP_JUMP);

  patchJump(elseJump);
  emitByte(OP_POP);

  parsePrecedence(PREC_OR);
  patchJump(endJump);
}

static void binary(bool canAssign) {
  (void)canAssign;
  TokenType operatorType = parser.previous.type;
  ParseRule *rule = getRule(operatorType);

  if (operatorType == TOKEN_NOT) {
    consume(TOKEN_IN, "Expect 'in' after 'not'.");
    parsePrecedence((Precedence)(rule->precedence + 1));
    emitByte(OP_CONTAINS);
    emitByte(OP_NOT);
    return;
  }

  parsePrecedence((Precedence)(rule->precedence + 1));

  switch (operatorType) {
  case TOKEN_BANG_EQUAL:
    emitBytes(OP_EQUAL, OP_NOT);
    break;
  case TOKEN_EQUAL_EQUAL:
    emitByte(OP_EQUAL);
    break;
  case TOKEN_GREATER:
    emitByte(OP_GREATER);
    break;
  case TOKEN_GREATER_EQUAL:
    emitBytes(OP_LESS, OP_NOT);
    break;
  case TOKEN_LESS:
    emitByte(OP_LESS);
    break;
  case TOKEN_LESS_EQUAL:
    emitBytes(OP_GREATER, OP_NOT);
    break;
  case TOKEN_PLUS:
    emitByte(OP_ADD);
    break;
  case TOKEN_MINUS:
    emitByte(OP_SUBTRACT);
    break;
  case TOKEN_STAR:
    emitByte(OP_MULTIPLY);
    break;
  case TOKEN_SLASH:
    emitByte(OP_DIVIDE);
    break;
  case TOKEN_PERCENT:
    emitByte(OP_MODULO);
    break;
  case TOKEN_SLASH_SLASH:
    emitByte(OP_FLOOR_DIVIDE);
    break;
  case TOKEN_STAR_STAR:
    emitByte(OP_POWER);
    break;
  case TOKEN_AMPERSAND:
    emitByte(OP_BIT_AND);
    break;
  case TOKEN_PIPE:
    emitByte(OP_BIT_OR);
    break;
  case TOKEN_CARET:
    emitByte(OP_BIT_XOR);
    break;
  case TOKEN_LEFT_SHIFT:
    emitByte(OP_SHIFT_LEFT);
    break;
  case TOKEN_RIGHT_SHIFT:
    emitByte(OP_SHIFT_RIGHT);
    break;
  case TOKEN_IS:
    emitByte(OP_IS);
    break;
  case TOKEN_IN:
    emitByte(OP_CONTAINS);
    break;
  default:
    return; // Unreachable.
  }
}

static void literal(bool canAssign) {
  (void)canAssign;
  switch (parser.previous.type) {
  case TOKEN_FALSE:
    emitByte(OP_FALSE);
    break;
  case TOKEN_NIL:
    emitByte(OP_NIL);
    break;
  case TOKEN_TRUE:
    emitByte(OP_TRUE);
    break;
  default:
    return; // Unreachable.
  }
}

static void grouping(bool canAssign) {
  (void)canAssign;
  expression();
  if (match(TOKEN_COMMA)) {
    int count = 1;
    if (!check(TOKEN_RIGHT_PAREN)) {
      do {
        if (check(TOKEN_RIGHT_PAREN))
          break;
        expression();
        count++;
      } while (match(TOKEN_COMMA));
    }
    consume(TOKEN_RIGHT_PAREN, "Expect ')' after tuple.");
    emitBytes(OP_BUILD_TUPLE, (uint8_t)count);
  } else {
    consume(TOKEN_RIGHT_PAREN, "Expect ')' after expression.");
  }
}

static void number(bool canAssign) {
  (void)canAssign;
  bool isFloat = false;
  for (int i = 0; i < parser.previous.length; i++) {
    char c = parser.previous.start[i];
    if (c == '.' || c == 'e' || c == 'E') {
      isFloat = true;
      break;
    }
  }
  if (isFloat) {
    double value = strtod(parser.previous.start, NULL);
    emitConstant(NUMBER_VAL(value));
  } else {
    ObjInt* integer = newIntFromString(parser.previous.start, parser.previous.length);
    if (integer == NULL) {
      double value = strtod(parser.previous.start, NULL);
      emitConstant(NUMBER_VAL(value));
      return;
    }
    emitConstant(OBJ_VAL(integer));
  }
}

static void unary(bool canAssign) {
  (void)canAssign;
  TokenType operatorType = parser.previous.type;

  // Compile the operand.
  parsePrecedence(PREC_UNARY);

  // Emit the operator instruction.
  switch (operatorType) {
  case TOKEN_BANG:
  case TOKEN_NOT:
    emitByte(OP_NOT);
    break;
  case TOKEN_MINUS:
    emitByte(OP_NEGATE);
    break;
  case TOKEN_TILDE:
    emitByte(OP_BIT_NOT);
    break;
  default:
    return; // Unreachable.
  }
}

static void super_(bool canAssign) {
  (void)canAssign;
  if (currentClass == NULL) {
    error("Can't use 'super' outside of a class.");
    return;
  }
  if (!currentClass->hasSuperclass) {
    error("Can't use 'super' in a class with no superclass.");
    return;
  }

  consume(TOKEN_LEFT_PAREN, "Expect '(' after 'super'.");
  consume(TOKEN_RIGHT_PAREN, "Expect ')' after 'super('.");
  consume(TOKEN_DOT, "Expect '.' after super().");
  consume(TOKEN_IDENTIFIER, "Expect superclass method name.");

  uint8_t name = identifierConstant(&parser.previous);
  namedVariable(currentClass->superclassName, false);
  emitBytes(OP_GET_LOCAL, 1);

  if (match(TOKEN_LEFT_PAREN)) {
    uint8_t posCount, kwCount, kwSourceCount, starConst, kwStarConst;
    bool hasStar, hasKwStar;
    argumentList(&posCount, &kwCount, &kwSourceCount, &hasStar, &starConst, &hasKwStar, &kwStarConst);
    if (kwCount > 0 || hasStar || hasKwStar) {
      error("super() calls currently support only positional arguments.");
      return;
    }
    emitBytes(OP_SUPER_INVOKE, name);
    emitByte(posCount);
  } else {
    emitBytes(OP_GET_SUPER, name);
  }
}

ParseRule rules[] = {
    [TOKEN_LEFT_PAREN] = {grouping, call, PREC_CALL},
    [TOKEN_RIGHT_PAREN] = {NULL, NULL, PREC_NONE},
    [TOKEN_LEFT_BRACE] = {dict, NULL, PREC_NONE},
    [TOKEN_RIGHT_BRACE] = {NULL, NULL, PREC_NONE},
    [TOKEN_LEFT_BRACKET] = {list, subscript, PREC_CALL},
    [TOKEN_RIGHT_BRACKET] = {NULL, NULL, PREC_NONE},
    [TOKEN_COMMA] = {NULL, NULL, PREC_NONE},
    [TOKEN_DOT] = {NULL, dot, PREC_CALL},
    [TOKEN_MINUS] = {unary, binary, PREC_TERM},
    [TOKEN_PLUS] = {NULL, binary, PREC_TERM},
    [TOKEN_SEMICOLON] = {NULL, NULL, PREC_NONE},
    [TOKEN_SLASH] = {NULL, binary, PREC_FACTOR},
    [TOKEN_STAR] = {NULL, binary, PREC_FACTOR},
    [TOKEN_PERCENT] = {NULL, binary, PREC_FACTOR},
    [TOKEN_SLASH_SLASH] = {NULL, binary, PREC_FACTOR},
    [TOKEN_STAR_STAR] = {NULL, binary, PREC_POWER},
    [TOKEN_AMPERSAND] = {NULL, binary, PREC_BIT_AND},
    [TOKEN_PIPE] = {NULL, binary, PREC_BIT_OR},
    [TOKEN_CARET] = {NULL, binary, PREC_BIT_XOR},
    [TOKEN_TILDE] = {unary, NULL, PREC_NONE},
    [TOKEN_LEFT_SHIFT] = {NULL, binary, PREC_SHIFT},
    [TOKEN_RIGHT_SHIFT] = {NULL, binary, PREC_SHIFT},
    [TOKEN_COLON] = {NULL, NULL, PREC_NONE},
    [TOKEN_BANG] = {unary, NULL, PREC_NONE},
    [TOKEN_BANG_EQUAL] = {NULL, binary, PREC_EQUALITY},
    [TOKEN_EQUAL] = {NULL, NULL, PREC_NONE},
    [TOKEN_EQUAL_EQUAL] = {NULL, binary, PREC_EQUALITY},
    [TOKEN_GREATER] = {NULL, binary, PREC_COMPARISON},
    [TOKEN_GREATER_EQUAL] = {NULL, binary, PREC_COMPARISON},
    [TOKEN_LESS] = {NULL, binary, PREC_COMPARISON},
    [TOKEN_LESS_EQUAL] = {NULL, binary, PREC_COMPARISON},
    [TOKEN_IDENTIFIER] = {variable, NULL, PREC_NONE},
    [TOKEN_STRING] = {string, NULL, PREC_NONE},
    [TOKEN_BYTES] = {bytes, NULL, PREC_NONE},
    [TOKEN_NUMBER] = {number, NULL, PREC_NONE},
    [TOKEN_AND] = {NULL, and_, PREC_AND},
    [TOKEN_CLASS] = {NULL, NULL, PREC_NONE},
    [TOKEN_ELSE] = {NULL, NULL, PREC_NONE},
    [TOKEN_FALSE] = {literal, NULL, PREC_NONE},
    [TOKEN_FOR] = {NULL, NULL, PREC_NONE},
    [TOKEN_DEF] = {NULL, NULL, PREC_NONE},
    [TOKEN_DEL] = {NULL, NULL, PREC_NONE},
    [TOKEN_IF] = {NULL, ternaryExpression, PREC_CONDITIONAL},
    [TOKEN_NIL] = {literal, NULL, PREC_NONE},
    [TOKEN_NOT] = {unary, binary, PREC_COMPARISON},
    [TOKEN_LAMBDA] = {lambdaExpression, NULL, PREC_NONE},
    [TOKEN_OR] = {NULL, or_, PREC_OR},
    [TOKEN_PRINT] = {NULL, NULL, PREC_NONE},
    [TOKEN_RETURN] = {NULL, NULL, PREC_NONE},
    [TOKEN_SUPER] = {super_, NULL, PREC_NONE},
    [TOKEN_THIS] = {NULL, NULL, PREC_NONE},
    [TOKEN_TRUE] = {literal, NULL, PREC_NONE},
    [TOKEN_VAR] = {NULL, NULL, PREC_NONE},
    [TOKEN_WHILE] = {NULL, NULL, PREC_NONE},
    [TOKEN_IS] = {NULL, binary, PREC_EQUALITY},
    [TOKEN_IN] = {NULL, binary, PREC_COMPARISON},
    [TOKEN_ERROR] = {NULL, NULL, PREC_NONE},
    [TOKEN_EOF] = {NULL, NULL, PREC_NONE},
};

static void parsePrecedence(Precedence precedence) {
  advance();
  ParseFn prefixRule = getRule(parser.previous.type)->prefix;
  if (prefixRule == NULL) {
    error("Expect expression.");
    return;
  }

  bool canAssign = precedence <= PREC_ASSIGNMENT;
  prefixRule(canAssign);

  while (precedence <= getRule(parser.current.type)->precedence) {
    advance();
    ParseFn infixRule = getRule(parser.previous.type)->infix;
    infixRule(canAssign);
  }

  if (canAssign && match(TOKEN_EQUAL)) {
    error("Invalid assignment target.");
  }
}

static ParseRule *getRule(TokenType type) { return &rules[type]; }

static void expression() { parsePrecedence(PREC_ASSIGNMENT); }

static void function(FunctionType type) {
  // 1. Parse defaults/names in the OUTER scope
  Scanner backupScanner = scanner;
  Parser backupParser = parser;
  int defaultsCount = 0;
  Token paramNames[255];
  int paramCount = 0;
  bool sawVarargs = false;

  consume(TOKEN_LEFT_PAREN, "Expect '(' after function name.");
  if (!check(TOKEN_RIGHT_PAREN)) {
    do {
      bool isVarargs = match(TOKEN_STAR);
      consume(TOKEN_IDENTIFIER, "Expect parameter name.");
      if (hasDuplicateParam(paramNames, paramCount, &parser.previous)) {
        error("Duplicate parameter name.");
      }
      paramNames[paramCount++] = parser.previous;
      if (isVarargs) {
        if (sawVarargs) {
          error("Can't have more than one varargs parameter.");
        }
        if (match(TOKEN_EQUAL)) {
          error("Varargs parameter cannot have a default value.");
        }
        sawVarargs = true;
      } else if (sawVarargs) {
        error("Parameters cannot follow varargs parameter.");
      } else if (match(TOKEN_EQUAL)) {
        expression(); // Evaluated in outer scope
        defaultsCount++;
      } else if (defaultsCount > 0) {
        error("Non-default argument follows default argument.");
      }
      if (paramCount == 255) errorAtCurrent("Too many parameters.");
    } while (match(TOKEN_COMMA));
  }
  consume(TOKEN_RIGHT_PAREN, "Expect ')' after parameters.");

  // 2. Start function compilation
  scanner = backupScanner;
  bool hadError = parser.hadError;
  parser = backupParser;
  if (hadError) parser.hadError = true;

  Compiler compiler;
  initCompiler(&compiler, type);

  consume(TOKEN_LEFT_PAREN, "Expect '(' after function name.");
  if (!check(TOKEN_RIGHT_PAREN)) {
    do {
      bool isVarargs = match(TOKEN_STAR);
      consume(TOKEN_IDENTIFIER, "Expect parameter name.");
      if (!isVarargs) {
        current->function->arity++;
      } else {
        current->function->hasVarargs = true;
      }
      addLocal(parser.previous);
      current->locals[current->localCount - 1].depth = current->scopeDepth;
      if (match(TOKEN_EQUAL)) {
        if (isVarargs) {
          error("Varargs parameter cannot have a default value.");
        }
        // Skip expression as it was already parsed in outer scope
        int bracketLevel = 0;
        while (bracketLevel > 0 ||
               (!check(TOKEN_COMMA) && !check(TOKEN_RIGHT_PAREN))) {
          if (check(TOKEN_LEFT_PAREN) || check(TOKEN_LEFT_BRACKET))
            bracketLevel++;
          if (check(TOKEN_RIGHT_PAREN) || check(TOKEN_RIGHT_BRACKET))
            bracketLevel--;
          advance();
          if (parser.previous.type == TOKEN_EOF)
            break;
        }
      }
    } while (match(TOKEN_COMMA));
  }
  consume(TOKEN_RIGHT_PAREN, "Expect ')' after parameters.");

  consume(TOKEN_COLON, "Expect ':' after parameters.");
  while (match(TOKEN_NEWLINE))
    ;

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after function body.");
  } else {
    statement();
  }

  ObjFunction *fn = endCompiler();
  emitBytes(OP_CONSTANT, makeConstant(OBJ_VAL(fn)));

  for (int i = 0; i < paramCount; i++) {
    writeValueArray(&fn->paramNames, OBJ_VAL(copyString(paramNames[i].start, paramNames[i].length)));
  }

  if (defaultsCount > 0) {
    emitBytes(OP_SET_DEFAULTS, (uint8_t)defaultsCount);
  }
}

static void returnStatement() {
  if (current->type == TYPE_SCRIPT) {
    error("Can't return from top-level code.");
  }

  if (match(TOKEN_NEWLINE) || check(TOKEN_DEDENT) || check(TOKEN_EOF) || check(TOKEN_SEMICOLON)) {
    emitReturn();
  } else {
    expression();
    emitByte(OP_RETURN);
  }
}

static void expressionStatement() {
  expression();
  emitByte(OP_POP);
}

typedef struct {
  Token names[255];
  int count;
} UnpackTargetList;

static bool parseUnpackTargetElements(TokenType closing, UnpackTargetList* out) {
  do {
    consume(TOKEN_IDENTIFIER, "Expect variable name in unpack target.");
    if (out->count == 255) {
      error("Too many unpack targets.");
      return false;
    }
    out->names[out->count++] = parser.previous;
  } while (match(TOKEN_COMMA) && !check(closing));
  return true;
}

static bool tryParseUnpackAssignment(void) {
  Scanner savedScanner = scanner;
  Parser savedParser = parser;
  UnpackTargetList targets;
  int i;

  targets.count = 0;

  if (check(TOKEN_LEFT_PAREN) || check(TOKEN_LEFT_BRACKET)) {
    TokenType closing = check(TOKEN_LEFT_PAREN) ? TOKEN_RIGHT_PAREN : TOKEN_RIGHT_BRACKET;
    advance();
    if (!parseUnpackTargetElements(closing, &targets)) {
      scanner = savedScanner;
      parser = savedParser;
      return false;
    }
    if (targets.count < 2 || !match(closing) || !match(TOKEN_EQUAL)) {
      scanner = savedScanner;
      parser = savedParser;
      return false;
    }
  } else if (check(TOKEN_IDENTIFIER)) {
    advance();
    targets.names[targets.count++] = parser.previous;
    if (!match(TOKEN_COMMA)) {
      scanner = savedScanner;
      parser = savedParser;
      return false;
    }
    do {
      consume(TOKEN_IDENTIFIER, "Expect variable name in unpack target.");
      targets.names[targets.count++] = parser.previous;
    } while (match(TOKEN_COMMA));
    if (!match(TOKEN_EQUAL)) {
      scanner = savedScanner;
      parser = savedParser;
      return false;
    }
  } else {
    return false;
  }

  expression();
  emitBytes(OP_UNPACK, (uint8_t)targets.count);
  for (i = targets.count - 1; i >= 0; i--) {
    storeName(targets.names[i]);
  }
  return true;
}

static void delStatement() {
  advance();
  ParseFn prefixRule = getRule(parser.previous.type)->prefix;
  if (prefixRule == NULL) {
    error("Expect delete target.");
    return;
  }
  prefixRule(false);

  while (match(TOKEN_DOT)) {
    dot(false);
  }

  consume(TOKEN_LEFT_BRACKET, "Expect '[' after delete target.");
  expression();
  consume(TOKEN_RIGHT_BRACKET, "Expect ']' after delete target.");
  emitByte(OP_DELETE_SUBSCRIPT);
}

static void ifStatement() {
  expression();
  consume(TOKEN_COLON, "Expect ':' after condition.");
  while (match(TOKEN_NEWLINE))
    ;

  int thenJump = emitJump(OP_JUMP_IF_FALSE);
  emitByte(OP_POP);

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after block.");
  } else {
    statement();
  }

  int elseJump = emitJump(OP_JUMP);

  patchJump(thenJump);
  emitByte(OP_POP);

  if (match(TOKEN_ELIF)) {
    ifStatement();
  } else if (match(TOKEN_ELSE)) {
    consume(TOKEN_COLON, "Expect ':' after 'else'.");
    while (match(TOKEN_NEWLINE))
      ;

    if (match(TOKEN_INDENT)) {
      block();
      consume(TOKEN_DEDENT, "Expect dedent after block.");
    } else {
      statement();
    }
  }

  patchJump(elseJump);
}

static void whileStatement() {
  Loop loop;
  loop.start = currentChunk()->count;
  loop.scopeDepth = current->scopeDepth;
  loop.breakCount = 0;
  loop.isFor = false;
  loop.enclosing = current->loop;
  current->loop = &loop;

  expression();
  consume(TOKEN_COLON, "Expect ':' after condition.");
  while (match(TOKEN_NEWLINE))
    ;

  int exitJump = emitJump(OP_JUMP_IF_FALSE);
  emitByte(OP_POP);

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after block.");
  } else {
    statement();
  }

  emitLoop(loop.start);

  patchJump(exitJump);
  emitByte(OP_POP);

  for (int i = 0; i < loop.breakCount; i++) {
    patchJump(loop.breakJumps[i]);
  }
  current->loop = loop.enclosing;
}

static void forStatement() {
  Loop loop;
  loop.breakCount = 0;
  loop.isFor = true;
  loop.enclosing = current->loop;
  current->loop = &loop;

  consume(TOKEN_IDENTIFIER, "Expect variable name after 'for'.");
  Token loopVar = parser.previous;

  consume(TOKEN_IN, "Expect 'in' after variable name.");

  expression(); // The iterable
  consume(TOKEN_COLON, "Expect ':' after iterable in 'for' loop.");
  while (match(TOKEN_NEWLINE))
    ;

  // Get the iterator
  emitByte(OP_GET_ITER);

  int localsBefore = current->localCount;
  Token dummy; dummy.start = ""; dummy.length = 0;
  addLocal(dummy);
  current->locals[current->localCount - 1].depth = current->scopeDepth;

  loop.start = currentChunk()->count;
  int exitJump = emitJump(OP_FOR_ITER);

  // Store loop variable
  int arg = resolveLocal(current, &loopVar);
  if (arg != -1) {
    emitBytes(OP_SET_LOCAL, (uint8_t)arg);
  } else if (current->type != TYPE_SCRIPT) {
    addLocal(loopVar);
    current->locals[current->localCount - 1].depth = current->scopeDepth;
    arg = current->localCount - 1;
    emitBytes(OP_SET_LOCAL, (uint8_t)arg);
  } else {
    arg = makeConstant(OBJ_VAL(copyString(loopVar.start, loopVar.length)));
    emitBytes(OP_SET_GLOBAL, (uint8_t)arg);
  }
  emitByte(OP_POP);

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after block.");
  } else {
    statement();
  }

  emitLoop(loop.start);

  patchJump(exitJump);
  emitByte(OP_POP); // Pop the iterator
  
  for (int i = 0; i < loop.breakCount; i++) {
    patchJump(loop.breakJumps[i]);
  }
  current->localCount = localsBefore;
  current->loop = loop.enclosing;
}

static void breakStatement() {
  if (current->loop == NULL) {
    error("Can't use 'break' outside of a loop.");
    return;
  }
  if (current->loop->isFor) {
    emitByte(OP_POP); // Pop the iterator
  }
  int jump = emitJump(OP_JUMP);
  if (current->loop->breakCount < 64) {
    current->loop->breakJumps[current->loop->breakCount++] = jump;
  } else {
    error("Too many breaks in one loop.");
  }
}

static void continueStatement() {
  if (current->loop == NULL) {
    error("Can't use 'continue' outside of a loop.");
    return;
  }
  emitLoop(current->loop->start);
}

static void synchronize() {
  parser.panicMode = false;

  while (parser.current.type != TOKEN_EOF) {
    if (parser.previous.type == TOKEN_NEWLINE)
      return;
    switch (parser.current.type) {
    case TOKEN_CLASS:
    case TOKEN_DEF:
    case TOKEN_FOR:
    case TOKEN_IF:
    case TOKEN_WHILE:
    case TOKEN_RETURN:
      return;
    default:; // Do nothing.
    }
    advance();
  }
}

static void simpleStatement() {
  if (match(TOKEN_RETURN)) {
    returnStatement();
  } else if (match(TOKEN_BREAK)) {
    breakStatement();
  } else if (match(TOKEN_CONTINUE)) {
    continueStatement();
  } else if (match(TOKEN_PASS)) {
    // Do nothing
  } else if (match(TOKEN_DEL)) {
    delStatement();
  } else if (match(TOKEN_RAISE)) {
    raiseStatement();
  } else if (match(TOKEN_IMPORT)) {
    importStatement();
  } else if (match(TOKEN_FROM)) {
    fromImportStatement();
  } else if (tryParseUnpackAssignment()) {
    return;
  } else {
    expressionStatement();
  }
}

static void tryStatement() {
  int tryJump = emitExceptSetup();
  
  consume(TOKEN_COLON, "Expect ':' after 'try'.");
  while (match(TOKEN_NEWLINE))
    ;

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after try block.");
  } else {
    statement();
  }

  emitByte(OP_POP_EXCEPT);
  int skipExcept = emitJump(OP_JUMP);

  patchJump(tryJump);

  while (match(TOKEN_NEWLINE))
    ;

  consume(TOKEN_EXCEPT, "Expect 'except' after try block.");

  uint8_t typeConstant = UINT8_MAX;
  bool hasBinding = false;
  Token bindingName;
  if (match(TOKEN_COLON)) {
    // Catch all.
  } else {
    consume(TOKEN_IDENTIFIER, "Expect exception type after 'except'.");
    typeConstant = makeConstant(
        OBJ_VAL(copyString(parser.previous.start, parser.previous.length)));
    if (match(TOKEN_AS)) {
      consume(TOKEN_IDENTIFIER, "Expect binding name after 'as'.");
      hasBinding = true;
      bindingName = parser.previous;
    }
    consume(TOKEN_COLON, "Expect ':' after exception type.");
  }

  patchExceptSetup(tryJump, typeConstant);

  while (match(TOKEN_NEWLINE))
    ;
  
  if (hasBinding) {
    storeName(bindingName);
  } else {
    emitByte(OP_POP);
  }

  if (match(TOKEN_INDENT)) {
    block();
    consume(TOKEN_DEDENT, "Expect dedent after except block.");
  } else {
    statement();
  }

  patchJump(skipExcept);
}

static void raiseStatement() {
  if (match(TOKEN_NEWLINE) || check(TOKEN_EOF) || check(TOKEN_SEMICOLON) || check(TOKEN_DEDENT)) {
    emitByte(OP_NIL);
  } else {
    expression();
  }
  emitByte(OP_RAISE);
}

static void importStatement() {
  consume(TOKEN_IDENTIFIER, "Expect module name after 'import'.");
  Token firstModuleName = parser.previous;
  Token lastModuleName = parser.previous;
  Token bindName = firstModuleName;
  bool hasDots = false;
  bool hasAlias = false;
  while (match(TOKEN_DOT)) {
    consume(TOKEN_IDENTIFIER, "Expect module name after '.'.");
    lastModuleName = parser.previous;
    hasDots = true;
  }
  if (match(TOKEN_AS)) {
    consume(TOKEN_IDENTIFIER, "Expect alias after 'as'.");
    bindName = parser.previous;
    hasAlias = true;
  }

  emitBytes(OP_IMPORT_MODULE, dottedPathConstant(&firstModuleName, &lastModuleName));
  if (hasDots && !hasAlias) {
    emitByte(OP_POP);
    emitBytes(OP_IMPORT_MODULE, identifierConstant(&firstModuleName));
  }
  storeName(bindName);
}

static void fromImportStatement() {
  consume(TOKEN_IDENTIFIER, "Expect module name after 'from'.");
  Token firstModuleName = parser.previous;
  Token lastModuleName = parser.previous;
  while (match(TOKEN_DOT)) {
    consume(TOKEN_IDENTIFIER, "Expect module name after '.'.");
    lastModuleName = parser.previous;
  }
  consume(TOKEN_IMPORT, "Expect 'import' after module name.");
  consume(TOKEN_IDENTIFIER, "Expect imported name.");
  Token importedName = parser.previous;
  Token bindName = importedName;
  if (match(TOKEN_AS)) {
    consume(TOKEN_IDENTIFIER, "Expect alias after 'as'.");
    bindName = parser.previous;
  }

  emitBytes(OP_IMPORT_MODULE, dottedPathConstant(&firstModuleName, &lastModuleName));
  emitBytes(OP_GET_PROPERTY, identifierConstant(&importedName));
  storeName(bindName);
}

static void statement() {
  if (match(TOKEN_IF)) {
    ifStatement();
  } else if (match(TOKEN_WHILE)) {
    whileStatement();
  } else if (match(TOKEN_FOR)) {
    forStatement();
  } else if (match(TOKEN_TRY)) {
    tryStatement();
  } else {
    for (;;) {
      simpleStatement();
      if (match(TOKEN_SEMICOLON)) {
        if (check(TOKEN_NEWLINE) || check(TOKEN_EOF) || check(TOKEN_DEDENT)) break;
        continue;
      }
      break;
    }
  }
}

static void method() {
  consume(TOKEN_IDENTIFIER, "Expect method name.");
  uint8_t constant = makeConstant(
      OBJ_VAL(copyString(parser.previous.start, parser.previous.length)));
  function(TYPE_METHOD);
  emitBytes(OP_METHOD, constant);
}

static void classDeclaration() {
  ClassCompiler classCompiler;
  consume(TOKEN_IDENTIFIER, "Expect class name.");
  Token className = parser.previous;
  uint8_t nameConstant = makeConstant(
      OBJ_VAL(copyString(parser.previous.start, parser.previous.length)));

  classCompiler.enclosing = currentClass;
  classCompiler.hasSuperclass = false;
  currentClass = &classCompiler;

  emitBytes(OP_CLASS, nameConstant);
  emitBytes(OP_DEFINE_GLOBAL, nameConstant);

  namedVariable(className, false); // Push class back on stack to attach methods

  if (match(TOKEN_LEFT_PAREN)) {
    Token superName;
    consume(TOKEN_IDENTIFIER, "Expect superclass name.");
    superName = parser.previous;
    consume(TOKEN_RIGHT_PAREN, "Expect ')' after superclass name.");
    classCompiler.hasSuperclass = true;
    classCompiler.superclassName = superName;
    namedVariable(superName, false);
    emitByte(OP_INHERIT);
  }

  consume(TOKEN_COLON, "Expect ':' after class name.");
  consume(TOKEN_NEWLINE, "Expect newline after class declaration.");
  consume(TOKEN_INDENT, "Expect block after class declaration.");

  while (!check(TOKEN_DEDENT) && !check(TOKEN_EOF)) {
    if (match(TOKEN_DEF)) {
      method();
    } else if (match(TOKEN_PASS)) {
      consume(TOKEN_NEWLINE, "Expect newline after 'pass'.");
    } else {
      error("Expect method declaration in class.");
      advance();
    }
  }

  consume(TOKEN_DEDENT, "Expect dedent after class block.");
  emitByte(OP_POP); // Pop the class object
  currentClass = classCompiler.enclosing;
}

static void declaration() {
  if (match(TOKEN_CLASS)) {
    classDeclaration();
  } else if (match(TOKEN_DEF)) {
    consume(TOKEN_IDENTIFIER, "Expect function name.");
    uint8_t global = makeConstant(
        OBJ_VAL(copyString(parser.previous.start, parser.previous.length)));
    function(TYPE_FUNCTION);
    emitBytes(OP_DEFINE_GLOBAL, global);
  } else if (match(TOKEN_NEWLINE)) {
    // Just skip extra newlines
  } else {
    statement();
  }

  if (parser.panicMode)
    synchronize();
}

ObjFunction *compile(const char *source, const char *filename) {
  initScanner(source, filename);

  Compiler compiler;
  initCompiler(&compiler, TYPE_SCRIPT);

  parser.hadError = false;
  parser.panicMode = false;

  advance();

  while (!match(TOKEN_EOF)) {
    declaration();
  }

  ObjFunction *function = endCompiler();
  return parser.hadError ? NULL : function;
}
