name = __platform_name()
sep = "/"


def getcwd():
    return __platform_getcwd()


def listdir(path="."):
    return __platform_listdir(path)


def exists(path):
    return __platform_exists(path)


def isdir(path):
    return __platform_isdir(path)


def isfile(path):
    return __platform_isfile(path)
