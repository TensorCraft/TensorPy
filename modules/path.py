import os


def isabs(value):
    return len(value) > 0 and value[0] == os.sep


def join(*parts):
    out = ""
    for part in parts:
        if part == "":
            continue
        if isabs(part):
            out = part
            continue
        if out == "" or out == os.sep:
            if out == os.sep:
                out = out + part
            else:
                out = part
        elif out.endswith(os.sep):
            out = out + part
        else:
            out = out + os.sep + part
    if out == "":
        return "."
    return normpath(out)


def normpath(value):
    if value == "":
        return "."

    absolute = isabs(value)
    pieces = value.split(os.sep)
    stack = []

    for piece in pieces:
        if piece == "" or piece == ".":
            continue
        if piece == "..":
            if len(stack) > 0 and stack[len(stack) - 1] != "..":
                stack.pop()
            elif not absolute:
                stack.append(piece)
        else:
            stack.append(piece)

    result = os.sep.join(stack)
    if absolute:
        result = os.sep + result

    if result == "":
        if absolute:
            return os.sep
        return "."
    return result


def abspath(value):
    if isabs(value):
        return normpath(value)
    return normpath(join(os.getcwd(), value))


def basename(value):
    normalized = normpath(value)
    if normalized == os.sep:
        return os.sep

    pieces = normalized.split(os.sep)
    return pieces[len(pieces) - 1]


def dirname(value):
    normalized = normpath(value)
    if normalized == os.sep:
        return os.sep

    pieces = normalized.split(os.sep)
    if len(pieces) <= 1:
        return "."

    pieces.pop()
    result = os.sep.join(pieces)
    if result == "":
        return os.sep
    return result


def splitext(value):
    base = basename(value)
    dirpart = dirname(value)
    pieces = base.split(".")

    if len(pieces) <= 1:
        return (value, "")

    if len(pieces) == 2 and pieces[0] == "":
        return (value, "")

    root = ".".join(pieces[:len(pieces) - 1])
    ext = "." + pieces[len(pieces) - 1]
    if dirpart == ".":
        return (root, ext)
    if dirpart == os.sep:
        return (os.sep + root, ext)
    return (join(dirpart, root), ext)
