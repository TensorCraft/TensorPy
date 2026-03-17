class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show(self):
        print("Point:")
        print(self.x)
        print(self.y)


p = Point(10, 20)
p.show()


class Person:
    def __init__(self, name, age=18):
        self.name = name
        self.age = age

    def greet(self, prefix="Hello"):
        print(prefix + " " + self.name + ", you are " + self.age)


p1 = Person("Antigravity")
print(p1.name)
print(p1.age)

p2 = Person("Alice", "25")
print(p2.name)
print(p2.age)

p1.greet()
p2.greet("Hi")


class Foo:
    def __init__(self, val):
        self.val = val

    def show(self):
        print(self.val)


li = [Foo(10), Foo(20)]
li[0].show()
li[1].show()


class Collector:
    def gather(self, head, *rest):
        print(head)
        print(len(rest))
        return rest


c = Collector()
print(c.gather(1))
print(c.gather(1, 2, 3))
print(c.gather(head=4))
