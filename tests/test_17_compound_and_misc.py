x = 10
x += 5
print("x is 15:", x)
x -= 3
print("x is 12:", x)
x *= 2
print("x is 24:", x)
x /= 4
print("x is 6:", x)
x %= 4
print("x is 2:", x)
x **= 3
print("x is 8:", x)
x //= 3
print("x is 2:", x)


def test_local_compound():
    y = 100
    y += 50
    print("local y is 150:", y)
    return y


z = test_local_compound()
print("result z is 150:", z)

b = b"hello world"
print(b)
