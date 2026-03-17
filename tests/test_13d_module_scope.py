import module_scope_demo as demo

print(demo.get())
print(demo.set_value(4))
print(demo.get())

try:
    print(counter)
except:
    print("module-scope-isolated")

counter = 100
print(counter)
print(demo.get())

import module_scope_demo as demo2
print(demo2.get())
