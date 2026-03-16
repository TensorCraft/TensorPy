print("Testing Lists:")
l = [1, 2, 3]
print("l =", l)
print("l[0] =", l[0])
l[1] = 20
print("l after l[1]=20 =", l)

print("\nTesting Tuples:")
t = (1, 2, 3)
print("t =", t)
print("t[2] =", t[2])

print("\nTesting Dicts:")
d = {"a": 1, "b": 2}
print("d =", d)
print("d['a'] =", d["a"])
d["c"] = 3
print("d after d['c']=3 =", d)
print(d.get("a"))
print(d.get("missing"))
print(d.get("missing", 3))

print("\nTesting Sets:")
s = {1, 2, 2, 3}
print("s =", s)

print("\nNested lists:")
l = [[1, 2], [3, 4]]
print(l)
print(l[0])
print(l[1][1])

print("List of dicts:")
ld = [{"a": 1}, {"b": 2}]
print(ld)
print(ld[1]["b"])
