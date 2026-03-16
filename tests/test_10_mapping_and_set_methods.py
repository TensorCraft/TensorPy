d = {"a": 1, "b": 2}
print(d.keys())
print(d.values())
print(d.items())
print(d.pop("a"))
print(d.setdefault("z", 7))
print(d.setdefault("z", 9))
print(d.copy())
print(d.fromkeys(["k1", "k2"], 5))
d.update({"c": 3})
print(d.popitem())
print(d.get("c"))
d.clear()
print(d)

st = {1, 2}
st.discard(3)
st.add(4)
print(st.copy())
print(st.union({5, 6}))
print(st.intersection({2, 8, 4}))
print(st.difference({1, 4}))
print(st.issubset({1, 2, 4, 9}))
print(st.issuperset({1, 4}))
st.update({9, 10})
print(st)
st.clear()
print(st)

try:
    {}.popitem()
except KeyError:
    print("dict-popitem-empty")

try:
    {}.pop("x")
except KeyError:
    print("dict-pop-missing")

try:
    {1}.remove(9)
except KeyError:
    print("set-remove-missing")
