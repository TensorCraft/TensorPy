DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

_state = {"level": INFO}


def set_level(level):
    _state["level"] = level


def get_level():
    return _state["level"]


def basicConfig(level=INFO):
    return set_level(level)


def _emit(level_name, level_value, name, message):
    if level_value < _state["level"]:
        return None
    if name is None:
        print("[" + level_name + "] " + str(message))
    else:
        print("[" + level_name + "] " + name + ": " + str(message))
    return None


def _format_exception(exc):
    if hasattr(exc, "message") and exc.message != "":
        return type(exc) + ": " + exc.message
    return str(exc)


def debug(message):
    return _emit("DEBUG", DEBUG, None, message)


def info(message):
    return _emit("INFO", INFO, None, message)


def warn(message):
    return _emit("WARN", WARN, None, message)


warning = warn


def error(message):
    return _emit("ERROR", ERROR, None, message)


def exception(exc):
    return _emit("ERROR", ERROR, None, _format_exception(exc))


class Logger:
    def __init__(self, name):
        self.name = name

    def debug(self, message):
        return _emit("DEBUG", DEBUG, self.name, message)

    def info(self, message):
        return _emit("INFO", INFO, self.name, message)

    def warn(self, message):
        return _emit("WARN", WARN, self.name, message)

    def warning(self, message):
        return _emit("WARN", WARN, self.name, message)

    def error(self, message):
        return _emit("ERROR", ERROR, self.name, message)

    def exception(self, exc):
        return _emit("ERROR", ERROR, self.name, _format_exception(exc))


def getLogger(name):
    return Logger(name)
