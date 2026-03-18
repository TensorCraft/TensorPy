def get(name, default=None):
    value = __platform_getenv(name)
    if value is None:
        return default
    return value


def exists(name):
    return __platform_getenv(name) is not None


def require(name):
    value = __platform_getenv(name)
    if value is None:
        raise OSError("missing environment variable: " + name)
    return value
