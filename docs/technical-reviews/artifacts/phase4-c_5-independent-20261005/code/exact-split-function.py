# Original scripts/course_experiments/common.py:50-70
def split_records(records, seed=42, family="family"):
    """Keep all descendants of a family in one split; never split individual answers."""
    groups = {}
    seen = set()
    for record in records:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        key = str(record.get(family, hashlib.sha256(encoded.encode()).hexdigest()))
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three distinct families are required for train/validation/test")
    random.Random(seed).shuffle(keys)
    a = min(max(1, int(len(keys) * 0.8)), len(keys) - 2)
    b = min(max(a + 1, int(len(keys) * 0.9)), len(keys) - 1)
    return {
        name: [record for key in selected for record in groups[key]]
        for name, selected in (("train", keys[:a]), ("validation", keys[a:b]), ("test", keys[b:]))
    }

