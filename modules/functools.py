def partial(func, *bound_args):
    def wrapped(*rest):
        args = []
        for item in bound_args:
            args.append(item)
        for item in rest:
            args.append(item)
        return func(*args)

    return wrapped


def compose(f, g):
    def wrapped(value):
        return f(g(value))

    return wrapped
