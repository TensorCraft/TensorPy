a, b = [1, 2]
print(a)
print(b)

x, y = ("left", "right")
print(x)
print(y)

[m, n] = [10, 20]
print(m)
print(n)

(p, q) = "hi"
print(p)
print(q)

r, s = b"AZ"
print(r)
print(s)

def pair():
    return [7, 8]


u, v = pair()
print(u)
print(v)

try:
    a, b = [1]
except ValueError:
    print("unpack-short")

try:
    a, b = [1, 2, 3]
except ValueError:
    print("unpack-long")

try:
    a, b = 42
except TypeError:
    print("unpack-noniter")
