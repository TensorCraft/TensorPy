import io


def _parse_line(line, delimiter=","):
    row = []
    field = ""
    i = 0
    in_quotes = False

    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == "\"":
                if i + 1 < len(line) and line[i + 1] == "\"":
                    field = field + "\""
                    i = i + 2
                    continue
                in_quotes = False
            else:
                field = field + ch
        else:
            if ch == "\"":
                in_quotes = True
            elif ch == delimiter:
                row.append(field)
                field = ""
            else:
                field = field + ch
        i = i + 1

    if in_quotes:
        raise ValueError("Unterminated quoted field")

    row.append(field)
    return row


def loads(text, delimiter=","):
    rows = []
    for line in text.split("\n"):
        if line == "":
            continue
        rows.append(_parse_line(line, delimiter))
    return rows


def read_rows(path, delimiter=","):
    return loads(io.read_text(path), delimiter)


def read_dicts(path, delimiter=","):
    rows = read_rows(path, delimiter)
    if len(rows) == 0:
        return []

    header = rows[0]
    out = []
    i = 1
    while i < len(rows):
        row = rows[i]
        item = {}
        j = 0
        while j < len(header):
            key = header[j]
            value = ""
            if j < len(row):
                value = row[j]
            item[key] = value
            j = j + 1
        out.append(item)
        i = i + 1
    return out
