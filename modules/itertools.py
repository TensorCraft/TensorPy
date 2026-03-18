def chain(*iterables):
    out = []
    for iterable in iterables:
        for item in iterable:
            out.append(item)
    return out


def repeat(value, count):
    out = []
    i = 0
    while i < count:
        out.append(value)
        i = i + 1
    return out


def take(count, iterable):
    out = []
    i = 0
    for item in iterable:
        if i >= count:
            break
        out.append(item)
        i = i + 1
    return out


def batched(iterable, size):
    batch = []
    out = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            out.append(batch)
            batch = []
    if len(batch) > 0:
        out.append(batch)
    return out
