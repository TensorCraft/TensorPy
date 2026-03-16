l = [1, 2]
l.append(3)
print(l)
l.remove(2)
print(l)
print(l.pop())
print(l)

l = [1, 2, 3]
l.insert(1, 9)
print(l)
print(l.count(9))
print(l.index(9))
print(l.copy())
l.reverse()
print(l)
print(l.pop(1))
print(l)
l.clear()
print(l)

l2 = [0, 1, 2, 3, 4]
l2[1:4] = [8, 9]
print(l2)

l3 = [{"id": 3}, {"id": 1}, {"id": 2}]
l3.sort(key=lambda x: x["id"], reverse=True)
print(l3)

print((1, 2, 1, 3).count(1))
print((1, 2, 1, 3).index(3))

try:
    [].pop()
except IndexError:
    print("list-pop-empty")

try:
    [1, 2].remove(9)
except ValueError:
    print("list-remove-missing")

try:
    (1, 2).index(9)
except ValueError:
    print("tuple-index-missing")
