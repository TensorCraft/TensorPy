import re

m = re.match("ab+c", "abbbc xyz")
print(m.group())
print(m.span())

m2 = re.search("c.t", "xx cat yy")
print(m2.group())
print(m2.start())
print(m2.end())

print(re.fullmatch("[a-z]+\\d+", "tensor42").group())
print(re.findall("\\d+", "a12 b34 c5"))
print(re.split("[,; ]+", "a,b; c"))
print(re.sub("\\d+", "#", "a12 b34 c5"))
print(re.subn("\\d+", "#", "a12 b34 c5"))

p = re.compile("^h.llo$")
print(p.match("hello").group())
print(p.fullmatch("hallo").group())
print(p.search("ohallo") is None)

print(re.search("[^0-9]+", "123abc456").group())
print(re.search("\\s+", "a \t b").group())
print(re.match("colou?r", "color").group())
print(re.match("colou?r", "colour").group())
