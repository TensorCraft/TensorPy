print(round(1.2))
print(round(1.7))
print(ord("A"))
print(chr(66))


class Item:
    def __init__(self):
        self.value = 3

    def ping(self):
        return "pong"


item = Item()
print(len(dir(item)) > 0)

print(reversed([1, 2, 3]))
print(zip([1, 2], ["a", "b", "c"]))
print(map(lambda x: x * 2, [1, 2, 3]))
print(filter(lambda x: x > 1, [0, 1, 2, 3]))
