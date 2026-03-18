import io
import logging
import os
import path
import sys
import traceback

base = "__tensorpy_v1_polish"
leaf = path.join(base, "a", "b")
text_path = path.join(base, "log.txt")
bytes_path = path.join(base, "data.bin")
renamed_path = path.join(base, "renamed.txt")
system_path = "__tensorpy_system.txt"

if os.exists(bytes_path):
    os.remove(bytes_path)
if os.exists(text_path):
    os.remove(text_path)
if os.exists(renamed_path):
    os.remove(renamed_path)
if os.exists(system_path):
    os.remove(system_path)
if os.exists(leaf):
    os.rmdir(leaf)
if os.exists(path.join(base, "a")):
    os.rmdir(path.join(base, "a"))
if os.exists(base):
    os.rmdir(base)

os.makedirs(leaf)
print(path.exists(base))
print(path.isdir(leaf))
print(path.isfile(text_path))
print(path.split("alpha/beta.txt"))
print(path.relpath(path.join(base, "a", "b"), base))
print(os.getenv("HOME") is not None)

io.write_lines(text_path, ["one", "two"])
io.append_text(text_path, "\nthree")
print(io.read_lines(text_path))

io.write_bytes(bytes_path, b"A")
io.append_bytes(bytes_path, b"BC")
print(io.read_bytes(bytes_path).decode())

os.replace(text_path, renamed_path)
print(path.basename(renamed_path))

print(os.system("printf system-ok > __tensorpy_system.txt"))
print(io.read_text(system_path))

logging.basicConfig(logging.ERROR)
logging.info("skip")
logging.exception(ValueError("boom"))

try:
    io.read_text(path.join(base, "missing.txt"))
except OSError as e:
    print(traceback.format_exception_only(e))
    print(traceback.as_dict(e)["type"])
    print(traceback.as_dict(e)["message"].find("read_text failed") >= 0)

print(sys.getdefaultencoding())
print(sys.byteorder)
print(sys.executable)
print(len(sys.path) >= 3)

os.remove(renamed_path)
os.remove(bytes_path)
os.remove(system_path)
os.removedirs(leaf)
print(os.exists(base))
