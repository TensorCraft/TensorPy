def add_item(x, items=[]):
    items.append(x)
    return items


print(len(add_item(1)))
print(len(add_item(2)))

f = lambda x, y=10: x + y
print(f(5))
print(f(5, 20))

g = lambda x=1: lambda y=2, x=x: x + y
print(g()())
print(g(10)())
print(g(10)(20))

h = lambda _: _ + 1
print(h(10))

chooser = lambda a, b: a if a > b else b
print(chooser(5, 10))
print(chooser(15, 10))


def collect(head, *rest):
    print(head)
    print(len(rest))
    if len(rest) > 0:
        print(rest[0])
    if len(rest) > 1:
        print(rest[1])
    return rest


print(collect(1))
print(collect(1, 2, 3))
print(collect(head=7))

pack = lambda first, *items: (first, len(items))
print(pack(10))
print(pack(10, 20, 30))
