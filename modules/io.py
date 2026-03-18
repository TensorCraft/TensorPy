import os


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


def append_text(path, text):
    current = ""
    if os.exists(path):
        current = read_text(path)
    return write_text(path, current + text)


def append_bytes(path, data):
    current = b""
    if os.exists(path):
        current = read_bytes(path)
    return write_bytes(path, (current.decode() + data.decode()).encode())


def read_lines(path):
    return read_text(path).split("\n")


def write_lines(path, lines):
    return write_text(path, "\n".join(lines))
