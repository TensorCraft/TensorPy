def random():
    return __platform_random()


def randint(a, b):
    span = b - a + 1
    value = __platform_random()
    return a + __math_floor(value * span)
