_registry = {}


def set(name, value):
    _registry[name] = value
    return None


def get(name, default=None):
    return _registry.get(name, default)


def has(name):
    return name in _registry


def call(name, *args):
    if name not in _registry:
        raise KeyError("missing host binding: " + name)
    target = _registry[name]
    if not callable(target):
        raise TypeError("host binding is not callable: " + name)
    return target(*args)


def keys():
    return _registry.keys()
