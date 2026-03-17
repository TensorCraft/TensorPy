def expect(condition, message):
    if not condition:
        raise RuntimeError(message)


baseline = __gc_object_count()


def churn():
    i = 0
    while i < 300:
        [i, {"box": [i, i + 1]}, (i,)]
        i = i + 1


churn()
after = __gc_object_count()

expect(after < baseline + 120, "automatic GC did not keep heap growth bounded")

freed = __gc_collect()
expect(freed >= 0, "explicit GC should still be available after auto collection")

print("gc-auto-ok")
