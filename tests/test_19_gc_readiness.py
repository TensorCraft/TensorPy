def expect(condition, message):
    if not condition:
        raise RuntimeError(message)


baseline = __gc_mark_count()


def make_payload():
    captured = ["x", {"k": "v"}]

    def inner():
        return captured

    return inner


fn = make_payload()
reachable_fn = __gc_reachable_count(fn)
expect(reachable_fn > 5, "closure marking missed captured environment")

payload = {"fn": fn, "items": [fn, ("tail",)]}
reachable_payload = __gc_reachable_count(payload)
expect(reachable_payload > reachable_fn, "container edges were not marked")


class Holder:
    def __init__(self, payload):
        self.payload = payload

    def call(self):
        return self.payload()


holder = Holder(fn)
reachable_holder = __gc_reachable_count(holder)
expect(reachable_holder > reachable_payload, "instance/class edges were not marked")

bound = holder.call
reachable_bound = __gc_reachable_count(bound)
expect(reachable_bound >= reachable_holder, "bound method edges were not marked")

after = __gc_mark_count()
expect(after > baseline, "root marking did not find newly reachable globals")

print("gc-mark-ok")
