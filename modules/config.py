import io
import json


def loads(text):
    return json.loads(text)


def load(path):
    return json.loads(io.read_text(path))


def get(mapping, key, default=None):
    return mapping.get(key, default)


def require(mapping, key):
    if key not in mapping:
        raise KeyError("missing config key: " + str(key))
    return mapping[key]


def merge(base, override):
    out = {}
    out.update(base)
    out.update(override)
    return out
