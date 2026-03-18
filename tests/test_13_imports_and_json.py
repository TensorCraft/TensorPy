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
print(json.loads("{\"n\":1.5e2,\"s\":\"line\\ntext\",\"u\":\"\\u0041\"}")["n"])
print(json.loads("{\"n\":1.5e2,\"s\":\"line\\ntext\",\"u\":\"\\u0041\"}")["s"])
print(json.loads("{\"n\":1.5e2,\"s\":\"line\\ntext\",\"u\":\"\\u0041\"}")["u"])
print(json.dumps({"ctrl": "a\b\f"}))
print(json.dumps({"unicode": "A" + chr(1) + chr(233)}))

try:
    json.loads("{\"a\": 1} trailing")
except ValueError as e:
    print(e.message)

try:
    json.loads("[1 2]")
except ValueError as e:
    print(e.message)

try:
    json.loads("\"\\x\"")
except ValueError as e:
    print(e.message)

try:
    json.loads("tru")
except ValueError as e:
    print(e.message)

try:
    json.loads("01")
except ValueError as e:
    print(e.message)

try:
    json.loads("1e")
except ValueError as e:
    print(e.message)

try:
    json.loads("\"\u0001\"")
except ValueError as e:
    print(e.message)
