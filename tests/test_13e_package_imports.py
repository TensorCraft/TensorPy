import pkg.mod
print(pkg.name)
print(pkg.mod.value)
print(pkg.mod.read())

from pkg import mod
print(mod.value)
print(pkg.mod is mod)

from pkg.mod import value as exported_value
print(exported_value)

import pkg.consumer as consumer
print(consumer.read_via_import())
print(consumer.same_module(mod))
