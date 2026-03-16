print("Test 1 start")

# Only a comment

    # Indented comment

print("Test 1 end")

print("Test 2 start"); a = 1; b = 2; print(a + b)
print("Test 2 end")

s1 = 'hello "world"'
s2 = "hello 'world'"
s3 = "line1\nline2\rline3"
print(s1)
print(s2)
print(s3)

s4 = """This is a
multiline string
with "double quotes" and 'single quotes'"""
s5 = '''Another
one
here'''
print(s4)
print(s5)

print("Outer Start")


def test():
    print("Start")

    # This is a comment on a blank line

    if True:
        print("Inside if")

    print("End")


test()
print("Outer End")
