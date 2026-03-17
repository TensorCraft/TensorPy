class Box:
    def __init__(self, value):
        self.value = value

    def double(self):
        return self.value * 2


box = Box(7)

print(isinstance(box, Box))
print(isinstance([], "list"))
print(isinstance("x", "str"))
print(isinstance(1, "float"))

print(getattr(box, "value"))
print(getattr(box, "missing", 99))
print(hasattr(box, "value"))
print(hasattr(box, "missing"))

setattr(box, "value", 11)
print(box.value)
print(getattr(box, "double")())
