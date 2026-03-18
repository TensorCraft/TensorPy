class A:
    def hi(self):
        return "A"


class B(A):
    def bye(self):
        return "B"


class C(B):
    def hi(self):
        return "C"


obj = B()
print(obj.hi())
print(obj.bye())

c = C()
print(c.hi())
print(c.bye())
