def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


print("10 + 20 =", add(10, 20))
print("10 * 20 =", multiply(10, 20))


def recursive_factorial(n):
    if n <= 1:
        return 1
    return n * recursive_factorial(n - 1)


print("factorial(5) =", recursive_factorial(5))

x = 100


def test(x):
    print("inner x:", x)
    x = 50
    print("updated inner x:", x)


test(10)
print("outer x after test:", x)


def scope_test():
    y = 5
    if True:
        y = 10
        print("y in if:", y)
    print("y after if:", y)


scope_test()

g1 = 10
g2 = 20
print(g1 + g2)
g1 = 5
print(g1 + g2)
g3 = g1 * g2 + 2
print(g3)
