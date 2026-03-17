import module_scope_demo as demo
import module_scope_demo as demo2

print(demo is demo2)
print(demo.set_value(9))
print(demo2.get())

import pkg.mod as mod1
import pkg.mod as mod2
print(mod1 is mod2)
print(mod1.read())
