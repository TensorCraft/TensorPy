import json

data = json.loads("{\"a\":1,\"b\":[2,3],\"c\":true}")
print(data["a"])
print(data["b"][1])
print(json.dumps({"ok": True, "nums": [1, 2]}))

from json import loads as jl
print(jl("[1,2,3]")[2])

from json import JSON
parser = JSON("{\"x\":5}")
print(parser.parse()["x"])
