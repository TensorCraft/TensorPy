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


def _copy_groups(groups):
    out = []
    i = 0
    while i < len(groups):
        out.append(groups[i])
        i = i + 1
    return out


def _blank_groups(count):
    out = []
    i = 0
    while i <= count:
        out.append(None)
        i = i + 1
    return out


def _count_groups(pattern):
    count = 0
    in_class = False
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i = i + 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif not in_class and c == "(":
            count = count + 1
        i = i + 1
    return count


def _group_index(pattern, group_start):
    count = 0
    in_class = False
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i = i + 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif not in_class and c == "(":
            count = count + 1
            if i == group_start:
                return count
        i = i + 1
    return -1


def _find_group_end(pattern, start):
    depth = 1
    i = start
    in_class = False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i = i + 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif not in_class and c == "(":
            depth = depth + 1
        elif not in_class and c == ")":
            depth = depth - 1
            if depth == 0:
                return i
        i = i + 1
    return -1


def _split_alternatives(pattern, start, end):
    parts = []
    part_start = start
    depth = 0
    in_class = False
    i = start
    while i < end:
        c = pattern[i]
        if c == "\\":
            i = i + 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif not in_class and c == "(":
            depth = depth + 1
        elif not in_class and c == ")":
            depth = depth - 1
        elif not in_class and depth == 0 and c == "|":
            parts.append((part_start, i))
            part_start = i + 1
        i = i + 1
    parts.append((part_start, end))
    return parts


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


def _match_one(pattern, group_bounds, group_index, atom, text, current_ti, current_groups):
    if not (group_bounds is None):
        inner_groups = _copy_groups(current_groups)
        result = _match_pattern_range(pattern, group_bounds[0], group_bounds[1], text, current_ti, inner_groups)
        if result is None:
            return None
        group_end_ti = result[0]
        result_groups = result[1]
        result_groups[group_index] = (current_ti, group_end_ti)
        return (group_end_ti, result_groups)
    if current_ti < len(text) and _atom_matches(atom, text[current_ti]):
        return (current_ti + 1, current_groups)
    return None


def _match_pattern_range(pattern, start, end, text, ti, groups):
    parts = _split_alternatives(pattern, start, end)
    if len(parts) == 1:
        return _match_sequence(pattern, start, end, text, ti, groups)

    for part in parts:
        result = _match_sequence(pattern, part[0], part[1], text, ti, _copy_groups(groups))
        if not (result is None):
            return result
    return None


def _match_sequence(pattern, pi, end_pi, text, ti, groups):
    if pi >= end_pi:
        return (ti, groups)

    if pattern[pi] == "$" and pi + 1 == end_pi:
        if ti == len(text):
            return (ti, groups)
        return None

    atom = None
    group_bounds = None
    group_index = -1
    if pattern[pi] == "(":
        group_end = _find_group_end(pattern, pi + 1)
        if group_end == -1 or group_end >= end_pi:
            return None
        group_bounds = (pi + 1, group_end)
        group_index = _group_index(pattern, pi)
        next_pi = group_end + 1
    else:
        parsed = _parse_atom(pattern, pi)
        atom = parsed[0]
        next_pi = parsed[1]

    quant = ""
    if next_pi < end_pi:
        q = pattern[next_pi]
        if q == "*" or q == "+" or q == "?":
            quant = q
            next_pi = next_pi + 1

    if quant == "":
        single = _match_one(pattern, group_bounds, group_index, atom, text, ti, _copy_groups(groups))
        if not (single is None):
            return _match_sequence(pattern, next_pi, end_pi, text, single[0], single[1])
        return None

    if quant == "?":
        single = _match_one(pattern, group_bounds, group_index, atom, text, ti, _copy_groups(groups))
        if not (single is None):
            result = _match_sequence(pattern, next_pi, end_pi, text, single[0], single[1])
            if not (result is None):
                return result
        return _match_sequence(pattern, next_pi, end_pi, text, ti, _copy_groups(groups))

    min_count = 0
    if quant == "+":
        min_count = 1

    states = [(ti, _copy_groups(groups))]
    while True:
        current_state = states[len(states) - 1]
        next_state = _match_one(pattern, group_bounds, group_index, atom, text, current_state[0], current_state[1])
        if next_state is None or next_state[0] == current_state[0]:
            break
        states.append(next_state)

    count = len(states) - 1
    if count < min_count:
        return None

    idx = count
    while idx >= min_count:
        result = _match_sequence(pattern, next_pi, end_pi, text, states[idx][0], _copy_groups(states[idx][1]))
        if not (result is None):
            return result
        idx = idx - 1

    return None


def _search_match(pattern, text, start_pos):
    pi = 0
    anchored = False
    group_count = _count_groups(pattern)
    if len(pattern) > 0 and pattern[0] == "^":
        anchored = True
        pi = 1

    if anchored:
        if start_pos != 0:
            return None
        result = _match_pattern_range(pattern, pi, len(pattern), text, 0, _blank_groups(group_count))
        if result is None:
            return None
        groups = result[1]
        groups[0] = (0, result[0])
        return (0, result[0], groups)

    pos = start_pos
    while pos <= len(text):
        result = _match_pattern_range(pattern, pi, len(pattern), text, pos, _blank_groups(group_count))
        if not (result is None):
            groups = result[1]
            groups[0] = (pos, result[0])
            return (pos, result[0], groups)
        pos = pos + 1
    return None


class Match:

    def __init__(self, text, start, end, groups):
        self.text = text
        self._start = start
        self._end = end
        self._groups = groups

    def group(self, index=0):
        if index < 0 or index >= len(self._groups):
            raise ValueError
        bounds = self._groups[index]
        if bounds is None:
            return None
        return _slice_text(self.text, bounds[0], bounds[1])

    def start(self, index=0):
        if index < 0 or index >= len(self._groups):
            raise ValueError
        bounds = self._groups[index]
        if bounds is None:
            return -1
        return bounds[0]

    def end(self, index=0):
        if index < 0 or index >= len(self._groups):
            raise ValueError
        bounds = self._groups[index]
        if bounds is None:
            return -1
        return bounds[1]

    def span(self, index=0):
        if index < 0 or index >= len(self._groups):
            raise ValueError
        bounds = self._groups[index]
        if bounds is None:
            return (-1, -1)
        return bounds


class Pattern:

    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, text):
        result = _search_match(self.pattern, text, 0)
        if result is None or result[0] != 0:
            return None
        return Match(text, result[0], result[1], result[2])

    def search(self, text):
        result = _search_match(self.pattern, text, 0)
        if result is None:
            return None
        return Match(text, result[0], result[1], result[2])

    def fullmatch(self, text):
        result = _search_match(self.pattern, text, 0)
        if result is None or result[0] != 0 or result[1] != len(text):
            return None
        return Match(text, result[0], result[1], result[2])

    def findall(self, text):
        out = []
        pos = 0
        while pos <= len(text):
            result = _search_match(self.pattern, text, pos)
            if result is None:
                break
            out.append(_slice_text(text, result[0], result[1]))
            if result[1] == result[0]:
                if result[1] >= len(text):
                    break
                pos = result[1] + 1
            else:
                pos = result[1]
        return out

    def split(self, text, maxsplit=0):
        out = []
        pos = 0
        splits = 0
        while pos <= len(text):
            if maxsplit != 0 and splits >= maxsplit:
                break
            result = _search_match(self.pattern, text, pos)
            if result is None:
                break
            out.append(_slice_text(text, pos, result[0]))
            splits = splits + 1
            if result[1] == result[0]:
                if result[1] >= len(text):
                    pos = result[1]
                    break
                pos = result[1] + 1
            else:
                pos = result[1]
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
            result = _search_match(self.pattern, text, pos)
            if result is None:
                break
            out = out + _slice_text(text, pos, result[0]) + repl
            replaced = replaced + 1
            if result[1] == result[0]:
                if result[1] >= len(text):
                    pos = result[1]
                    break
                out = out + _slice_text(text, result[1], result[1] + 1)
                pos = result[1] + 1
            else:
                pos = result[1]
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
