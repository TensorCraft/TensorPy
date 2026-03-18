def read_text(path):
    data = __platform_read_text(path)
    if data is None:
        raise OSError("read_text failed: " + path)
    return data


def read_bytes(path):
    data = __platform_read_bytes(path)
    if data is None:
        raise OSError("read_bytes failed: " + path)
    return data


def write_text(path, text):
    if not __platform_write_text(path, text):
        raise OSError("write_text failed: " + path)
    return None


def write_bytes(path, data):
    if not __platform_write_bytes(path, data):
        raise OSError("write_bytes failed: " + path)
    return None
