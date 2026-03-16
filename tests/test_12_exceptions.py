print("Try block test start")

try:
    print("Inside try")
    a = 1 / 0
    print("Should not be here")
except:
    print("Caught exception!")

print("After try-except")


def test_nested():
    print("Nested test start")
    try:
        print("Outer try")
        try:
            print("Inner try")
            raise "Custom Error"
        except:
            print("Inner catch")
            raise "Reraised Error"
    except:
        print("Outer catch")
    print("Nested test end")


test_nested()

try:
    value = {"x": 1}
    print(value["missing"])
except KeyError:
    print("Caught key error")

try:
    items = [1]
    print(items[3])
except IndexError:
    print("Caught index error")

try:
    raise ValueError
except ValueError:
    print("Caught typed value error")

try:
    try:
        raise ValueError
    except KeyError:
        print("Should not catch")
except ValueError:
    print("Reraised typed value error")
