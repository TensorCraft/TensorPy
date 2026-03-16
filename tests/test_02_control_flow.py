x = 10
if x > 5:
    print("x is greater than 5")
    if x == 10:
        print("x is exactly 10")
else:
    print("x is 5 or less")

i = 0
while i < 3:
    print("while loop iteration:", i)
    i = i + 1

y = 2
if y > 5:
    print("if block")
elif y > 0:
    print("elif block")
else:
    print("else block")

print("Testing break/continue")
for i in [1, 2, 3]:
    if i == 2:
        continue
    print(i)
print("Done")

x = 1 if True else 2
print(x)
y = 1 if False else 2
print(y)

print(True and False)
print(True or False)
print(not True)
print(1 and 2)
print(0 or "hello")
