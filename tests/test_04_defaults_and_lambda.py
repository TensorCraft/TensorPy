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
