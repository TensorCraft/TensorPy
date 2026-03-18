value = b"abc"


try:
    print(value.index(122))
except ValueError:
    print("bytes-index-missing-a")


try:
    print(value.split(b""))
except ValueError:
    print("bytes-split-empty-a")


try:

    print("abc".index("z"))
except ValueError:
    print("str-index-missing-a")


try:
    print(value.index(122))
except ValueError:
    print("bytes-index-missing-b")


try:


    print(value.split(b""))
except ValueError:
    print("bytes-split-empty-b")
