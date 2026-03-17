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


def combine(a, b=2, c=3):
    return a + b + c


print("combine(1, c=9) =", combine(1, c=9))
print("combine(c=5, a=4, b=6) =", combine(c=5, a=4, b=6))

try:
    combine()
except RuntimeError as e:
    print("missing arg:", e.message)

try:
    combine(1, a=2)
except RuntimeError as e:
    print("duplicate arg:", e.message)

try:
    combine(1, d=4)
except RuntimeError as e:
    print("unexpected kw:", e.message)


def make_adder(a):
    offset = a

    def add_inner(b):
        return offset + b

    return add_inner


add5 = make_adder(5)
add10 = make_adder(10)
print("closure add5 =", add5(7))
print("closure add10 =", add10(3))


def make_pair_sum():
    nums = [2, 4]

    def pair_sum():
        return nums[0] + nums[1]

    return pair_sum


print("closure list sum =", make_pair_sum()())
