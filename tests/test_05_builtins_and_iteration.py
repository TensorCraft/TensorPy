print("len([1, 2, 3]):")
print(len([1, 2, 3]))

print("len('hello'):")
print(len("hello"))

print("type(None):")
print(type(None))

print("type(123):")
print(type(123))

print("type('hello'):")
print(type("hello"))

print("type([]):")
print(type([]))

print("sorted([3, 1, 4, 1, 5, 9]):")
print(sorted([3, 1, 4, 1, 5, 9]))

print("sorted(['banana', 'apple', 'cherry']):")
print(sorted(["banana", "apple", "cherry"]))

l = [10, 20, 30]
for x in l:
    print(x)

t = (40, 50)
for y in t:
    print(y)

d = {"a": 1, "b": 2}
for k in d:
    print(k)

s = {100, 200}
for item in s:
    print(item)

for c in "hey":
    print(c)

print("range(5):")
for x in range(5):
    print(x)

print("range(2, 5):")
for x in range(2, 5):
    print(x)

print("range(1, 10, 2):")
for x in range(1, 10, 2):
    print(x)

print("enumerate(['a', 'b', 'c']):")
for item in enumerate(["a", "b", "c"]):
    print(item[0])
    print(item[1])

print("range(5, 0, -1):")
for x in range(5, 0, -1):
    print(x)
