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
