def type_name(value):
    return type(value)


def is_callable(value):
    return callable(value)


def is_function(value):
    kind = type(value)
    return kind == "function" or kind == "builtin_function_or_method"


def is_class(value):
    return type(value) == "type"


def is_module(value):
    return type(value) == "module"


def members(value):
    out = []
    for name in dir(value):
        out.append((name, getattr(value, name)))
    return out
