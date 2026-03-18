import io
import logging
import os
import path
import sys
import traceback

base = "__tensorpy_v1_tmp"
nested = path.join(base, "nested", "deeper")
text_path = path.join(base, "hello.txt")
bytes_path = path.join(base, "data.bin")
renamed_path = path.join(base, "renamed.txt")

if os.exists(renamed_path):
    os.remove(renamed_path)
if os.exists(bytes_path):
    os.remove(bytes_path)
if os.exists(text_path):
    os.remove(text_path)
if os.exists(nested):
    os.rmdir(nested)
if os.exists(path.join(base, "nested")):
    os.rmdir(path.join(base, "nested"))
if os.exists(base):
    os.rmdir(base)

os.makedirs(nested)
print(os.isdir(base))
print(os.isdir(nested))

io.write_text(text_path, "hello v1")
print(io.read_text(text_path))

io.write_bytes(bytes_path, b"ABC")
print(io.read_bytes(bytes_path).decode())

os.rename(text_path, renamed_path)
print(os.isfile(renamed_path))
print(path.basename(renamed_path))
print(path.dirname(renamed_path))
print(path.splitext("archive.tar.gz")[0])
print(path.splitext("archive.tar.gz")[1])
print(path.normpath("alpha/./beta/../gamma"))
print(path.join("alpha", "beta", "gamma"))
print(path.abspath(".").endswith("TensorPy"))

print(sys.implementation)
print(sys.platform)
print(len(sys.argv))

logging.set_level(logging.INFO)
logging.debug("skip-me")
logging.info("ready")
logger = logging.getLogger("demo")
logger.warn("warned")

try:
    io.read_text(path.join(base, "missing.txt"))
except OSError as e:
    print(traceback.format_exception(e))
    traceback.print_exception(e)

os.remove(renamed_path)
os.remove(bytes_path)
os.rmdir(nested)
os.rmdir(path.join(base, "nested"))
os.rmdir(base)
print(os.exists(base))
