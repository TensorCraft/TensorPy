class Counter:
    def __init__(self, items=None):
        self.data = {}
        if items is not None:
            self.update(items)

    def update(self, items):
        for item in items:
            self.data[item] = self.data.get(item, 0) + 1
        return None

    def get(self, key, default=0):
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def most_common(self):
        out = []
        for key in self.data:
            out.append((key, self.data[key]))
        return sorted(out)


class defaultdict:
    def __init__(self, factory):
        self.default_factory = factory
        self.data = {}

    def get(self, key):
        if key not in self.data:
            self.data[key] = self.default_factory()
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value
        return None

    def items(self):
        return self.data.items()


def flatten(items):
    out = []
    for group in items:
        for item in group:
            out.append(item)
    return out


def chunked(items, size):
    out = []
    current = []
    for item in items:
        current.append(item)
        if len(current) == size:
            out.append(current)
            current = []
    if len(current) > 0:
        out.append(current)
    return out
