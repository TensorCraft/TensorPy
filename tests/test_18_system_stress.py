class Node:
    def __init__(self, name, children=None, weight=0):
        self.name = name
        self.children = children if children else []
        self.weight = weight

    def add(self, child):
        self.children.append(child)

    def collect(self, result, depth=0):
        label = self.name.upper().replace("_", "-")
        result.append((depth, label, self.weight))
        for child in self.children:
            child.collect(result, depth + 1)

    def walk(self):
        result = []
        self.collect(result)
        return result


root = Node("root_node", weight=1)
left = Node("left_branch", weight=2)
right = Node("right_branch", weight=3)
right.add(Node("leaf_a", weight=5))
right.add(Node("leaf_b", weight=8))
root.add(left)
root.add(right)

records = root.walk()
totals = {}
seen = set()

for record in records:
    depth = record[0]
    name = record[1]
    weight = record[2]
    key = name.strip().lower().split("-")[0]
    totals[key] = totals.get(key, 0) + weight
    if key not in seen:
        seen.add(key)

summary = []
for key in sorted(totals.keys()):
    summary.append((key, totals[key], key in seen, not (key not in seen)))


def render(rows):
    lines = []
    for row in rows:
        key = row[0]
        total = row[1]
        present = row[2]
        mirrored = row[3]
        marker = "ok" if present and mirrored else "bad"
        lines.append(key + ":" + str(total) + ":" + marker)
    return "|".join(lines)


report = render(summary)
print(report)

try:
    cache = {"root": report}
    del cache["root"]
    print(cache.get("root", "missing"))
    raise KeyError
except KeyError:
    print("stress-keyerror")
