import pkg.mod


def read_via_import():
    return pkg.mod.read()


def same_module(other):
    return pkg.mod is other
