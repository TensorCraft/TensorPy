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


def getenv(name, default=None):
    value = __platform_getenv(name)
    if value is None:
        return default
    return value


def _raise_os_error(action, path):
    raise OSError(action + " failed: " + path)


def mkdir(path, exist_ok=False):
    if __platform_mkdir(path):
        return None
    if exist_ok and isdir(path):
        return None
    _raise_os_error("mkdir", path)


def makedirs(path, exist_ok=False):
    if __platform_makedirs(path):
        return None
    if exist_ok and isdir(path):
        return None
    _raise_os_error("makedirs", path)


def remove(path):
    if __platform_remove(path):
        return None
    _raise_os_error("remove", path)


def rmdir(path):
    if __platform_rmdir(path):
        return None
    _raise_os_error("rmdir", path)


def rename(src, dst):
    if __platform_rename(src, dst):
        return None
    raise OSError("rename failed: " + src + " -> " + dst)


def replace(src, dst):
    return rename(src, dst)


def system(command):
    return __platform_system(command)


def _parent_dir(path):
    pieces = path.split(sep)
    if len(pieces) <= 1:
        return ""
    pieces.pop()
    return sep.join(pieces)


def removedirs(path):
    current = path
    while current != "" and current != "." and current != sep:
        if not isdir(current):
            break
        rmdir(current)
        current = _parent_dir(current)
    return None
