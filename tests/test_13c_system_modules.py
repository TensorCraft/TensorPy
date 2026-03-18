import os
import random
import time

cwd = os.getcwd()
print(type(cwd))
print(len(cwd) > 0)
print(os.name)
print(os.sep)
entries = os.listdir(".")
print("modules" in entries)
print(os.exists("modules"))
print(os.isdir("modules"))
print(os.isfile("README.md"))
print(os.isfile("missing-tensorpy-file.txt"))

r = random.random()
print(r >= 0)
print(r <= 1)

value = random.randint(2, 5)
print(value >= 2)
print(value <= 5)

now = time.time()
mono = time.monotonic()
print(now >= 0)
print(mono >= 0)
