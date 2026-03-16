#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "tensorpy/common.h"
#include "tensorpy/scanner.h"

Scanner scanner;

void initScanner(const char* source, const char* filename) {
    scanner.start = source;
    scanner.current = source;
    scanner.filename = filename;
    scanner.lineStart = source;
    scanner.line = 1;
    scanner.indentStack[0] = 0;
    scanner.indentLevel = 1;
    scanner.pendingTokens = 0;
    scanner.isAtLineStart = true;
    scanner.bracketLevel = 0;
}

static bool isAlpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

static bool isAtEnd() {
    return *scanner.current == '\0';
}

static char advance() {
    scanner.current++;
    return scanner.current[-1];
}

static char peek() {
    return *scanner.current;
}

static char peekNext() {
    if (isAtEnd()) return '\0';
    return scanner.current[1];
}

static bool match(char expected) {
    if (isAtEnd()) return false;
    if (*scanner.current != expected) return false;
    scanner.current++;
    return true;
}

static Token makeToken(TokenType type) {
    Token token;
    token.type = type;
    token.start = scanner.start;
    token.length = (int)(scanner.current - scanner.start);
    token.line = scanner.line;
    token.column = (int)(scanner.start - scanner.lineStart) + 1;
    return token;
}

static Token errorToken(const char* message) {
    Token token;
    token.type = TOKEN_ERROR;
    token.start = message;
    token.length = (int)strlen(message);
    token.line = scanner.line;
    token.column = (int)(scanner.current - scanner.lineStart) + 1;
    return token;
}


static TokenType checkKeyword(int start, int length, const char* rest, TokenType type) {
    if (scanner.current - scanner.start == start + length &&
        memcmp(scanner.start + start, rest, length) == 0) {
        return type;
    }

    return TOKEN_IDENTIFIER;
}

static TokenType identifierType() {
    switch (scanner.start[0]) {
        case 'a': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'n': return checkKeyword(2, 1, "d", TOKEN_AND);
                    case 's': return (scanner.current - scanner.start == 2) ? TOKEN_AS : checkKeyword(2, 3, "ync", TOKEN_ASYNC);
                }
            }
            break;
        case 'b': return checkKeyword(1, 4, "reak", TOKEN_BREAK);
        case 'c': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'l': return checkKeyword(2, 3, "ass", TOKEN_CLASS);
                    case 'o': return checkKeyword(2, 6, "ntinue", TOKEN_CONTINUE);
                }
            }
            break;
        case 'd': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'e':
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 'f': return checkKeyword(3, 0, "", TOKEN_DEF);
                                case 'l': return checkKeyword(3, 0, "", TOKEN_DEL);
                            }
                        }
                        break;
                }
            }
            break;
        case 'e':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'l': 
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 's': return checkKeyword(3, 1, "e", TOKEN_ELSE);
                                case 'i': return checkKeyword(3, 1, "f", TOKEN_ELIF);
                            }
                        }
                        break;
                    case 'x': return checkKeyword(2, 4, "cept", TOKEN_EXCEPT);
                }
            }
            break;
        case 'F':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'a': return checkKeyword(2, 3, "lse", TOKEN_FALSE);
                }
            }
            break;
        case 'f':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'o': return checkKeyword(2, 1, "r", TOKEN_FOR);
                    case 'r': return checkKeyword(2, 2, "om", TOKEN_FROM);
                    case 'i': return checkKeyword(2, 5, "nally", TOKEN_FINALLY);
                }
            }
            break;
        case 'i': 
            if (scanner.current - scanner.start > 1) {
              switch (scanner.start[1]) {
                case 'f': return checkKeyword(2, 0, "", TOKEN_IF);
                case 'm': return checkKeyword(2, 4, "port", TOKEN_IMPORT);
                case 's': return checkKeyword(2, 0, "", TOKEN_IS);
                case 'n': return checkKeyword(2, 0, "", TOKEN_IN);
              }
            }
            break;
        case 'l': return checkKeyword(1, 5, "ambda", TOKEN_LAMBDA);
        case 'N':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'o': 
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 'n': return checkKeyword(3, 1, "e", TOKEN_NIL);
                            }
                        }
                        break;
                }
            }
            break;
        case 'n':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'o': 
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 't': return checkKeyword(3, 0, "", TOKEN_NOT);
                            }
                        }
                        break;
                }
            }
            break;
        case 'o': return checkKeyword(1, 1, "r", TOKEN_OR);
        case 'p': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'a': return checkKeyword(2, 2, "ss", TOKEN_PASS);
                    // treat 'print' as identifier for Python 3 style
                }
            }
            break;
        case 'r': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'e': return checkKeyword(2, 4, "turn", TOKEN_RETURN);
                    case 'a': return checkKeyword(2, 3, "ise", TOKEN_RAISE);
                }
            }
            break;
        case 'T':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'r': 
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 'u': return checkKeyword(3, 1, "e", TOKEN_TRUE);
                            }
                        }
                        break;
                }
            }
            break;
        case 't':
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'r': 
                        if (scanner.current - scanner.start > 2) {
                            switch (scanner.start[2]) {
                                case 'y': return checkKeyword(3, 0, "", TOKEN_TRY);
                            }
                        }
                        break;
                }
            }
            break;
        case 'w': 
            if (scanner.current - scanner.start > 1) {
                switch (scanner.start[1]) {
                    case 'h': return checkKeyword(2, 3, "ile", TOKEN_WHILE);
                    case 'i': return checkKeyword(2, 2, "th", TOKEN_WITH);
                }
            }
            break;
        case 'y': return checkKeyword(1, 4, "ield", TOKEN_YIELD);
    }

    return TOKEN_IDENTIFIER;
}

static Token identifier() {
    while (isAlpha(peek()) || isDigit(peek())) advance();
    return makeToken(identifierType());
}

static Token number() {
    while (isDigit(peek())) advance();

    if (peek() == '.' && isDigit(peekNext())) {
        advance();
        while (isDigit(peek())) advance();
    }

    return makeToken(TOKEN_NUMBER);
}

static Token string(char quote) {
    bool isTriple = false;
    if (peek() == quote && peekNext() == quote) {
        isTriple = true;
        advance();
        advance();
    }

    while (!isAtEnd()) {
        if (peek() == quote) {
            if (isTriple) {
                if (peekNext() == quote && scanner.current[2] == quote) {
                    advance();
                    advance();
                    advance();
                    return makeToken(TOKEN_STRING);
                }
            } else {
                advance();
                return makeToken(TOKEN_STRING);
            }
        }
        
        if (peek() == '\\') {
            advance();
            if (!isAtEnd()) advance();
        } else {
            if (peek() == '\n') {
                scanner.line++;
                scanner.lineStart = scanner.current + 1;
            }
            advance();
        }
    }

    if (isAtEnd()) return errorToken("Unterminated string.");
    return makeToken(TOKEN_STRING); // Should not reach
}

static Token bytes(char quote) {
    while (peek() != quote && !isAtEnd()) {
        if (peek() == '\\') {
            advance();
            if (!isAtEnd()) advance();
        } else {
            if (peek() == '\n') {
                scanner.line++;
                scanner.lineStart = scanner.current + 1;
            }
            advance();
        }
    }

    if (isAtEnd()) return errorToken("Unterminated bytes literal.");

    advance();
    return makeToken(TOKEN_BYTES);
}

static void skipSpaces() {
    for (;;) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r') {
            advance();
        } else if (c == '#') {
            while (peek() != '\n' && !isAtEnd()) {
                advance();
            }
        } else {
            break;
        }
    }
}

Token scanToken() {
    if (scanner.pendingTokens > 0) {
        scanner.pendingTokens--;
        return makeToken(TOKEN_INDENT);
    }
    if (scanner.pendingTokens < 0) {
        scanner.pendingTokens++;
        return makeToken(TOKEN_DEDENT);
    }

    if (scanner.isAtLineStart) {
        scanner.isAtLineStart = false;
        if (scanner.bracketLevel > 0) {
            return scanToken();
        }
        int spaces = 0;
        for (;;) {
            if (peek() == ' ') {
                advance();
                spaces++;
            } else if (peek() == '\t') {
                advance();
                spaces += 4;
            } else {
                break;
            }
        }

        if (peek() == '\n' || (peek() == '#' && scanner.bracketLevel == 0) || isAtEnd()) {
            if (isAtEnd()) {
                while (scanner.indentLevel > 1) {
                    scanner.indentLevel--;
                    scanner.pendingTokens--;
                }
                if (scanner.pendingTokens < 0) return scanToken();
                return makeToken(TOKEN_EOF);
            }
            if (peek() == '\n') {
                advance();
                scanner.line++;
                scanner.lineStart = scanner.current;
            } else if (peek() == '#') {
                while (!isAtEnd() && peek() != '\n') advance();
                if (peek() == '\n') {
                    advance();
                    scanner.line++;
                    scanner.lineStart = scanner.current;
                }
            }
            scanner.isAtLineStart = true;
            return scanToken();
        }

        if (spaces > scanner.indentStack[scanner.indentLevel - 1]) {
            scanner.indentStack[scanner.indentLevel++] = spaces;
            return makeToken(TOKEN_INDENT);
        }

        while (spaces < scanner.indentStack[scanner.indentLevel - 1]) {
            scanner.indentLevel--;
            scanner.pendingTokens--;
        }

        if (scanner.pendingTokens < 0) {
            scanner.pendingTokens++;
            return makeToken(TOKEN_DEDENT);
        }
    }

    skipSpaces();
    scanner.start = scanner.current;

    if (isAtEnd()) {
        while (scanner.indentLevel > 1) {
            scanner.indentLevel--;
            scanner.pendingTokens--;
        }
        if (scanner.pendingTokens < 0) return scanToken();
        return makeToken(TOKEN_EOF);
    }

    char c = advance();
    if (isAlpha(c)) {
        if (c == 'b' && (peek() == '"' || peek() == '\'')) {
            char quote = advance();
            return bytes(quote);
        }
        return identifier();
    }
    if (isDigit(c)) return number();

    switch (c) {
        case '(': scanner.bracketLevel++; return makeToken(TOKEN_LEFT_PAREN);
        case ')': if (scanner.bracketLevel > 0) scanner.bracketLevel--; return makeToken(TOKEN_RIGHT_PAREN);
        case '{': scanner.bracketLevel++; return makeToken(TOKEN_LEFT_BRACE);
        case '}': if (scanner.bracketLevel > 0) scanner.bracketLevel--; return makeToken(TOKEN_RIGHT_BRACE);
        case '[': scanner.bracketLevel++; return makeToken(TOKEN_LEFT_BRACKET);
        case ']': if (scanner.bracketLevel > 0) scanner.bracketLevel--; return makeToken(TOKEN_RIGHT_BRACKET);
        case ';': return makeToken(TOKEN_SEMICOLON);
        case ',': return makeToken(TOKEN_COMMA);
        case '.': return makeToken(TOKEN_DOT);
        case '-': return makeToken(match('=') ? TOKEN_MINUS_EQUAL : TOKEN_MINUS);
        case '+': return makeToken(match('=') ? TOKEN_PLUS_EQUAL : TOKEN_PLUS);
        case '/':
            if (match('/')) {
                return makeToken(match('=') ? TOKEN_SLASH_SLASH_EQUAL : TOKEN_SLASH_SLASH);
            }
            return makeToken(match('=') ? TOKEN_SLASH_EQUAL : TOKEN_SLASH);
        case '*':
            if (match('*')) {
                return makeToken(match('=') ? TOKEN_STAR_STAR_EQUAL : TOKEN_STAR_STAR);
            }
            return makeToken(match('=') ? TOKEN_STAR_EQUAL : TOKEN_STAR);
        case '%': return makeToken(match('=') ? TOKEN_PERCENT_EQUAL : TOKEN_PERCENT);
        case '&': return makeToken(TOKEN_AMPERSAND);
        case '|': return makeToken(TOKEN_PIPE);
        case '^': return makeToken(TOKEN_CARET);
        case '~': return makeToken(TOKEN_TILDE);
        case ':': return makeToken(TOKEN_COLON);
        case '\n':
            scanner.line++;
            scanner.lineStart = scanner.current;
            scanner.isAtLineStart = true;
            if (scanner.bracketLevel > 0) return scanToken();
            return makeToken(TOKEN_NEWLINE);
        case '!':
            return makeToken(match('=') ? TOKEN_BANG_EQUAL : TOKEN_BANG);
        case '=':
            return makeToken(match('=') ? TOKEN_EQUAL_EQUAL : TOKEN_EQUAL);
        case '<':
            if (match('<')) return makeToken(TOKEN_LEFT_SHIFT);
            return makeToken(match('=') ? TOKEN_LESS_EQUAL : TOKEN_LESS);
        case '>':
            if (match('>')) return makeToken(TOKEN_RIGHT_SHIFT);
            return makeToken(match('=') ? TOKEN_GREATER_EQUAL : TOKEN_GREATER);
        case '"': return string('"');
        case '\'': return string('\'');
    }

    if (c < 0) return scanToken();

    return errorToken("Unexpected character.");
}
