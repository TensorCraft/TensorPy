def _is_list(value):
    return type(value) == "list"


def zeros(count):
    out = []
    i = 0
    while i < count:
        out.append(0)
        i = i + 1
    return out


def full(count, value):
    out = []
    i = 0
    while i < count:
        out.append(value)
        i = i + 1
    return out


def shape(value):
    if not _is_list(value):
        return []
    if len(value) == 0:
        return [0]

    out = [len(value)]
    for item in shape(value[0]):
        out.append(item)
    return out


def add(a, b):
    out = []
    if _is_list(b):
        i = 0
        while i < len(a):
            out.append(a[i] + b[i])
            i = i + 1
        return out
    for item in a:
        out.append(item + b)
    return out


def mul(a, b):
    out = []
    if _is_list(b):
        i = 0
        while i < len(a):
            out.append(a[i] * b[i])
            i = i + 1
        return out
    for item in a:
        out.append(item * b)
    return out


def matmul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = []
    i = 0
    while i < rows:
        row = []
        j = 0
        while j < cols:
            total = 0
            k = 0
            while k < inner:
                total = total + a[i][k] * b[k][j]
                k = k + 1
            row.append(total)
            j = j + 1
        out.append(row)
        i = i + 1
    return out
