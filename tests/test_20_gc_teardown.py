import json
import re


def make_closure(seed):
    values = [seed, seed + 1, {"label": "v" + str(seed)}]

    def inner(extra):
        return (values, extra)

    return inner


class Box:
    def __init__(self, payload):
        self.payload = payload
        self.items = [payload, payload(99)]

    def view(self):
        return self.items


closures = []
boxes = []

for n in [1, 2, 3, 4, 5]:
    fn = make_closure(n)
    closures.append(fn)
    boxes.append(Box(fn))

payload = {
    "closures": closures,
    "boxes": boxes,
    "json": json.loads("{\"a\":1,\"b\":[2,3],\"c\":true}"),
    "match": re.match("a|b", "b"),
}

print(len(payload["closures"]))
print(len(payload["boxes"]))
print(payload["json"]["b"][1])
print(payload["match"].group(0))
print("gc-teardown-ok")
