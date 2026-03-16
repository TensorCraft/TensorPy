print(2 in [1, 2, 3])
print(4 in [1, 2, 3])
print("he" in "hello")
print("zz" not in "hello")
print("a" in {"a": 1, "b": 2})
print(3 in {1, 2, 3})
print(not False)
print(not 0)
print(not [])

l = [0, 1, 2, 3, 4]
print(l)
print(l[1:3])
print(l[:3])
print(l[2:])
print(l[:])
print(l[::-1])

del l[1]
print(l)

mapping = {"a": 1, "b": 2}
del mapping["a"]
print(mapping.get("a"))
