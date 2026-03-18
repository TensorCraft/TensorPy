import collections
import config
import env
import functools
import host
import io
import itertools
import os


counter = collections.Counter(["a", "b", "a", "c", "a"])
print(counter.get("a"))
print(collections.flatten([[1, 2], [3], [4, 5]]))
print(collections.chunked([1, 2, 3, 4, 5], 2))

defaulted = collections.defaultdict(list)
defaulted.get("nums").append(1)
defaulted.get("nums").append(2)
print(defaulted.items())

print(itertools.chain([1, 2], [3], [4, 5]))
print(itertools.repeat("x", 3))
print(itertools.take(3, [9, 8, 7, 6]))
print(itertools.batched([1, 2, 3, 4, 5], 2))


def add3(a, b, c):
    return a + b + c


add10 = functools.partial(add3, 10)
print(add10(20, 30))
double_then_str = functools.compose(str, lambda x: x * 2)
print(double_then_str(6))

host.set("value", 42)
host.set("adder", add3)
print(host.has("value"))
print(host.get("value"))
print(host.call("adder", 1, 2, 3))

cfg_path = "__tensorpy_config.json"
io.write_text(cfg_path, "{\"name\":\"tensorpy\",\"ok\":true}")
cfg = config.load(cfg_path)
print(config.get(cfg, "name"))
print(config.require(cfg, "ok"))
print(config.merge({"a": 1}, {"b": 2}))

print(env.exists("HOME"))
home = env.get("HOME", "")
print(len(home) > 0)

os.remove(cfg_path)
