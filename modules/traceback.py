def format_exception(exc):
    message = ""
    if hasattr(exc, "message"):
        message = exc.message
    if message == "":
        return type(exc)
    return type(exc) + ": " + message


def print_exception(exc):
    print(format_exception(exc))
    return None


def format_exception_only(exc):
    return [format_exception(exc)]


def as_dict(exc):
    message = ""
    if hasattr(exc, "message"):
        message = exc.message
    return {"type": type(exc), "message": message, "text": format_exception(exc)}
