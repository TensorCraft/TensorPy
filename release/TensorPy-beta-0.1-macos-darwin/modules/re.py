_ASCII_ORDER = "\t\n\r !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"


def _is_digit(c):
    return "0123456789".find(c) != -1


def _is_word(c):
    if _is_digit(c):
        return True
    if "abcdefghijklmnopqrstuvwxyz".find(c) != -1:
        return True
    if "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(c) != -1:
        return True
    return c == "_"


def _is_space(c):
    return c == " " or c == "\t" or c == "\n" or c == "\r"


def _in_range(c, start, end):
    ci = _ASCII_ORDER.find(c)
    si = _ASCII_ORDER.find(start)
    ei = _ASCII_ORDER.find(end)
    if ci == -1 or si == -1 or ei == -1:
        return False
    return ci >= si and ci <= ei


def _escaped_atom(ch):
    if ch == "d":
        return {"type": "digit"}
    if ch == "w":
        return {"type": "word"}
    if ch == "s":
        return {"type": "space"}
    return {"type": "literal", "value": ch}


def _slice_text(text, start, end):
    out = ""
    i = start
    while i < end and i < len(text):
        out = out + text[i]
        i = i + 1
    return out


def _parse_class(pattern, i):
    negate = False
    if i < len(pattern) and pattern[i] == "^":
        negate = True
        i = i + 1

    chars = []
    ranges = []

    while i < len(pattern):
        c = pattern[i]
        if c == "]":
            return ({"type": "class", "negate": negate, "chars": chars, "ranges": ranges}, i + 1)
        if c == "\\":
            i = i + 1
            if i >= len(pattern):
                chars.append("\\")
                break
            esc = pattern[i]
            chars.append("\\" + esc)
            i = i + 1
        elif i + 2 < len(pattern) and pattern[i + 1] == "-" and pattern[i + 2] != "]":
            ranges.append((pattern[i], pattern[i + 2]))
            i = i + 3
        else:
            chars.append(c)
            i = i + 1

    return ({"type": "class", "negate": negate, "chars": chars, "ranges": ranges}, i)


def _parse_atom(pattern, i):
    c = pattern[i]
    if c == ".":
        return ({"type": "dot"}, i + 1)
    if c == "\\":
        if i + 1 >= len(pattern):
            return ({"type": "literal", "value": "\\"}, i + 1)
        return (_escaped_atom(pattern[i + 1]), i + 2)
    if c == "[":
        return _parse_class(pattern, i + 1)
    return ({"type": "literal", "value": c}, i + 1)


def _class_contains(atom, c):
    chars = atom["chars"]
    for item in chars:
        if len(item) == 2 and item[0] == "\\":
            esc = item[1]
            if esc == "d" and _is_digit(c):
                return True
            if esc == "w" and _is_word(c):
                return True
            if esc == "s" and _is_space(c):
                return True
            if esc != "d" and esc != "w" and esc != "s" and c == esc:
                return True
        elif c == item:
            return True

    for pair in atom["ranges"]:
        if _in_range(c, pair[0], pair[1]):
            return True
    return False


def _atom_matches(atom, c):
    t = atom["type"]
    if t == "literal":
        return c == atom["value"]
    if t == "dot":
        return c != "\n"
    if t == "digit":
        return _is_digit(c)
    if t == "word":
        return _is_word(c)
    if t == "space":
        return _is_space(c)
    if t == "class":
        inside = _class_contains(atom, c)
        if atom["negate"]:
            return not inside
        return inside
    return False


def _match_sequence(pattern, pi, text, ti):
    if pi >= len(pattern):
        return ti

    if pattern[pi] == "$" and pi + 1 == len(pattern):
        if ti == len(text):
            return ti
        return -1

    parsed = _parse_atom(pattern, pi)
    atom = parsed[0]
    next_pi = parsed[1]

    quant = ""
    if next_pi < len(pattern):
        q = pattern[next_pi]
        if q == "*" or q == "+" or q == "?":
            quant = q
            next_pi = next_pi + 1

    if quant == "":
        if ti < len(text) and _atom_matches(atom, text[ti]):
            return _match_sequence(pattern, next_pi, text, ti + 1)
        return -1

    if quant == "?":
        if ti < len(text) and _atom_matches(atom, text[ti]):
            end = _match_sequence(pattern, next_pi, text, ti + 1)
            if end != -1:
                return end
        return _match_sequence(pattern, next_pi, text, ti)

    min_count = 0
    if quant == "+":
        min_count = 1

    cur = ti
    count = 0
    while cur < len(text) and _atom_matches(atom, text[cur]):
        cur = cur + 1
        count = count + 1

    if count < min_count:
        return -1

    min_cur = ti + min_count
    while cur >= min_cur:
        end = _match_sequence(pattern, next_pi, text, cur)
        if end != -1:
            return end
        cur = cur - 1

    return -1


def _search_bounds(pattern, text, start_pos):
    pi = 0
    anchored = False
    if len(pattern) > 0 and pattern[0] == "^":
        anchored = True
        pi = 1

    if anchored:
        if start_pos != 0:
            return None
        end = _match_sequence(pattern, pi, text, 0)
        if end == -1:
            return None
        return (0, end)

    pos = start_pos
    while pos <= len(text):
        end = _match_sequence(pattern, pi, text, pos)
        if end != -1:
            return (pos, end)
        pos = pos + 1
    return None


class Match:

    def __init__(self, text, start, end):
        self.text = text
        self._start = start
        self._end = end

    def group(self, index=0):
        if index != 0:
            raise ValueError
        return _slice_text(self.text, self._start, self._end)

    def start(self):
        return self._start

    def end(self):
        return self._end

    def span(self):
        return (self._start, self._end)


class Pattern:

    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, text):
        bounds = _search_bounds(self.pattern, text, 0)
        if bounds is None or bounds[0] != 0:
            return None
        return Match(text, bounds[0], bounds[1])

    def search(self, text):
        bounds = _search_bounds(self.pattern, text, 0)
        if bounds is None:
            return None
        return Match(text, bounds[0], bounds[1])

    def fullmatch(self, text):
        bounds = _search_bounds(self.pattern, text, 0)
        if bounds is None or bounds[0] != 0 or bounds[1] != len(text):
            return None
        return Match(text, bounds[0], bounds[1])

    def findall(self, text):
        out = []
        pos = 0
        while pos <= len(text):
            bounds = _search_bounds(self.pattern, text, pos)
            if bounds is None:
                break
            out.append(_slice_text(text, bounds[0], bounds[1]))
            if bounds[1] == bounds[0]:
                if bounds[1] >= len(text):
                    break
                pos = bounds[1] + 1
            else:
                pos = bounds[1]
        return out

    def split(self, text, maxsplit=0):
        out = []
        pos = 0
        splits = 0
        while pos <= len(text):
            if maxsplit != 0 and splits >= maxsplit:
                break
            bounds = _search_bounds(self.pattern, text, pos)
            if bounds is None:
                break
            out.append(_slice_text(text, pos, bounds[0]))
            splits = splits + 1
            if bounds[1] == bounds[0]:
                if bounds[1] >= len(text):
                    pos = bounds[1]
                    break
                pos = bounds[1] + 1
            else:
                pos = bounds[1]
        out.append(_slice_text(text, pos, len(text)))
        return out

    def sub(self, repl, text, count=0):
        return self.subn(repl, text, count)[0]

    def subn(self, repl, text, count=0):
        out = ""
        pos = 0
        replaced = 0
        while pos <= len(text):
            if count != 0 and replaced >= count:
                break
            bounds = _search_bounds(self.pattern, text, pos)
            if bounds is None:
                break
            out = out + _slice_text(text, pos, bounds[0]) + repl
            replaced = replaced + 1
            if bounds[1] == bounds[0]:
                if bounds[1] >= len(text):
                    pos = bounds[1]
                    break
                out = out + _slice_text(text, bounds[1], bounds[1] + 1)
                pos = bounds[1] + 1
            else:
                pos = bounds[1]
        out = out + _slice_text(text, pos, len(text))
        return (out, replaced)


def compile(pattern):
    return Pattern(pattern)


def match(pattern, text):
    return compile(pattern).match(text)


def search(pattern, text):
    return compile(pattern).search(text)


def fullmatch(pattern, text):
    return compile(pattern).fullmatch(text)


def findall(pattern, text):
    return compile(pattern).findall(text)


def split(pattern, text, maxsplit=0):
    return compile(pattern).split(text, maxsplit)


def sub(pattern, repl, text, count=0):
    return compile(pattern).sub(repl, text, count)


def subn(pattern, repl, text, count=0):
    return compile(pattern).subn(repl, text, count)
