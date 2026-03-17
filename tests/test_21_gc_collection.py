def expect(condition, message):
    if not condition:
        raise RuntimeError(message)


baseline = __gc_object_count()


def churn():
    i = 0
    while i < 20:
        [i, {"box": [i, i + 1]}, (i,)]
        i = i + 1


keeper = {"name": "rooted", "items": [1, 2, 3]}
keeper_before = __gc_reachable_count(keeper)

churn()
mid = __gc_object_count()
expect(mid > baseline, "expected temporary objects before collection")

freed = __gc_collect()
after = __gc_object_count()
keeper_after = __gc_reachable_count(keeper)

expect(freed > 0, "collector did not free unreachable objects")
expect(after < mid, "object count did not decrease after collection")
expect(keeper_before == keeper_after, "collector dropped rooted objects")

print("gc-collect-ok")
