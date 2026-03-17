class JSON:

    def __init__(self, s):
        self.s = s
        self.i = 0

    def parse(self):
        self.skip()
        value = self.value()
        self.skip()
        if self.i != len(self.s):
            raise ValueError("Extra data")
        return value

    def skip(self):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == " " or c == "\n" or c == "\t" or c == "\r":
                self.i = self.i + 1
            else:
                break

    def value(self):
        self.skip()
        if self.i >= len(self.s):
            raise ValueError("Unexpected end of JSON input")
        c = self.s[self.i]

        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c == "\"":
            return self.string()
        if c == "t":
            self.i = self.i + 4
            return True
        if c == "f":
            self.i = self.i + 5
            return False
        if c == "n":
            self.i = self.i + 4
            return None
        return self.number()

    def string(self):
        self.i = self.i + 1
        out = ""

        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "\"":
                self.i = self.i + 1
                return out
            if c == "\\":
                self.i = self.i + 1
                esc = self.s[self.i]
                if esc == "n":
                    out = out + "\n"
                elif esc == "t":
                    out = out + "\t"
                elif esc == "r":
                    out = out + "\r"
                else:
                    out = out + esc
                self.i = self.i + 1
            else:
                out = out + c
                self.i = self.i + 1

        return out

    def number(self):
        sign = 1
        if self.s[self.i] == "-":
            sign = -1
            self.i = self.i + 1

        value = 0
        saw_digit = False
        while self.i < len(self.s):
            c = self.s[self.i]
            digit = _digit_value(c)
            if digit != -1:
                value = value * 10 + digit
                self.i = self.i + 1
                saw_digit = True
            else:
                break

        if not saw_digit:
            raise ValueError("Invalid number")

        if self.i < len(self.s) and self.s[self.i] == ".":
            self.i = self.i + 1
            factor = 0.1
            saw_fraction = False
            while self.i < len(self.s):
                c = self.s[self.i]
                digit = _digit_value(c)
                if digit != -1:
                    value = value + digit * factor
                    factor = factor / 10
                    self.i = self.i + 1
                    saw_fraction = True
                else:
                    break
            if not saw_fraction:
                raise ValueError("Invalid number")

        return value * sign

    def arr(self):
        self.i = self.i + 1
        out = []
        self.skip()

        if self.s[self.i] == "]":
            self.i = self.i + 1
            return out

        while True:
            out.append(self.value())
            self.skip()
            if self.i >= len(self.s):
                raise ValueError("Unterminated array")
            if self.s[self.i] == "]":
                self.i = self.i + 1
                break
            if self.s[self.i] != ",":
                raise ValueError("Expected ',' or ']' in array")
            self.i = self.i + 1

        return out

    def obj(self):
        self.i = self.i + 1
        out = {}
        self.skip()

        if self.s[self.i] == "}":
            self.i = self.i + 1
            return out

        while True:
            if self.i >= len(self.s) or self.s[self.i] != "\"":
                raise ValueError("Expected string key")
            key = self.string()
            self.skip()
            if self.i >= len(self.s) or self.s[self.i] != ":":
                raise ValueError("Expected ':' after key")
            self.i = self.i + 1
            out[key] = self.value()
            self.skip()
            if self.i >= len(self.s):
                raise ValueError("Unterminated object")
            if self.s[self.i] == "}":
                self.i = self.i + 1
                break
            if self.s[self.i] != ",":
                raise ValueError("Expected ',' or '}' in object")
            self.i = self.i + 1

        return out


def loads(s):
    parser = JSON(s)
    return parser.parse()


def _digit_value(c):
    digits = "0123456789"
    return digits.find(c)


def _escape_string(s):
    out = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\"":
            out = out + "\\\""
        elif c == "\\":
            out = out + "\\\\"
        elif c == "\n":
            out = out + "\\n"
        elif c == "\t":
            out = out + "\\t"
        elif c == "\r":
            out = out + "\\r"
        else:
            out = out + c
        i = i + 1
    return out


def dumps(value):
    t = type(value)

    if value is None:
        return "null"
    if t == "bool":
        if value:
            return "true"
        return "false"
    if t == "float":
        return str(value)
    if t == "str":
        return "\"" + _escape_string(value) + "\""
    if t == "list" or t == "tuple":
        parts = []
        for item in value:
            parts.append(dumps(item))
        return "[" + ",".join(parts) + "]"
    if t == "dict":
        parts = []
        for key in value.keys():
            parts.append(dumps(str(key)) + ":" + dumps(value[key]))
        return "{" + ",".join(parts) + "}"

    return dumps(str(value))
