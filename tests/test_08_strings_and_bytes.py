print("abc".count("a"))
print("banana".count("na"))
print("banana".index("na"))
print("123".isdigit())
print("abc".isalpha())
print("abc".islower())
print("ABC".isupper())
print("hello world".title())
print("hELLo".swapcase())
print("hello".capitalize())
print("hello".encode())
print("x".center(5))
print("x".ljust(4))
print("x".rjust(4))
print("  hi".lstrip())
print("hi  ".rstrip())
print("tensorpy".startswith("tensor"))
print("tensorpy".endswith("py"))
print("banana".find("na"))
print("banana".replace("na", "NA"))
print("a,b,c".split(","))

try:
    print("abc".index("z"))
except ValueError:
    print("str-index-missing")

b = b"hello"
print(b)
print(type(b))
print(len(b))
print(b.decode())
print(b.hex())
print(b[1])
print(b[1:4])
print(b.count(108))
print(b.index(111))
print(b.find(111))
print(b.startswith(b"he"))
print(b.endswith(b"lo"))
print(b.replace(b"ll", b"XX"))
print(b.split(b"l"))
print(b"-".join([b"a", b"b", b"c"]))
print(101 in b)
print(b"ell" in b)
print(b"  hi  ".strip())
print(b"  hi".lstrip())
print(b"hi  ".rstrip())

try:
    print(b.index(120))
except ValueError:
    print("bytes-index-missing")

try:
    print(b.split(b""))
except ValueError:
    print("bytes-split-empty")
