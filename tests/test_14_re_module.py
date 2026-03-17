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
print(re.fullmatch("cat|dog", "dog").group())

g1 = re.fullmatch("(ab)+", "abab")
print(g1.group())
print(g1.group(1))
print(g1.span(1))

g2 = re.fullmatch("([a-z]+)(\\d+)", "tensor42")
print(g2.group(1))
print(g2.group(2))
print(g2.start(2))
print(g2.end(2))

g3 = re.fullmatch("a(bc)?d", "ad")
print(g3.group(1) is None)

print(re.fullmatch("(ab){2,3}", "ababab").group())
print(re.fullmatch("a{2}", "aa").group())
print(re.fullmatch("[0-9]{2,4}", "123").group())
print(re.findall("([a-z]+)(\\d+)", "a12 b34"))
