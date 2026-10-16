"""先建立世界家族，再衍生B/C題目；ID不放進模型輸入。"""

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "finite-worlds-v1"
SPLITS = ("train", "calibration", "policy", "test")
TASKS = ("card", "address", "app", "boxes", "sum", "none")
MEANINGS = {
    "card": "查詢卡片申請進度",
    "address": "說明地址填寫規則",
    "app": "協助App重設密碼",
    "boxes": "追蹤兩盒的數值",
    "sum": "計算三個整數的和",
    "none": "以上皆不適合",
    "yes": "必要資訊全部齊備",
    "no": "必要資訊尚未齊備",
    "grade0": "必要欄位全缺",
    "grade1": "必要欄位部分齊備",
    "grade2": "必要欄位全部齊備",
}
QUESTIONS = {
    "choice": "哪個任務適合這段材料？",
    "info": "依該流程的入口契約，必要資訊是否齊備？",
    "rubric": "依必要欄位的完整程度，選出對應級別。",
}
CANDIDATES = {"choice": TASKS, "info": ("yes", "no"), "rubric": ("grade0", "grade1", "grade2")}
REQUIRED = {
    "card": ("application",),
    "address": (),
    "app": ("device",),
    "boxes": ("a", "b", "events", "query"),
    "sum": ("x", "y", "z"),
}
EVENTS = [(box, value) for box in ("A", "B") for value in range(8)] + [("swap", 0)]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def family(task, facts):
    return task + ":" + digest(facts)


def split_of(family_id, salt):
    bucket = int(hashlib.sha256((salt + ":" + family_id).encode()).hexdigest()[:12], 16) % 1000
    return SPLITS[0 if bucket < 760 else 1 if bucket < 840 else 2 if bucket < 920 else 3]


def answer(world):
    if world["task"] == "sum":
        return world["x"] + world["y"] + world["z"]
    a, b = world["a"], world["b"]
    for box, value in world["events"]:
        if box == "swap":
            a, b = b, a
        elif box == "A":
            a = value
        else:
            b = value
    return a if world["query"] == "A" else b


def present(value):
    return value is not None


def render(world, omitted=(), shifted=False):
    """固定公開語法；0是值，?才表示缺欄。沒有case ID或答案。"""

    def value(key):
        return "?" if key in omitted or not present(world.get(key)) else str(world[key])

    task = world["task"]
    if task == "boxes":
        events = (
            "?"
            if "events" in omitted
            else ",".join("交換" if box == "swap" else box + "設" + str(n) for box, n in world["events"])
        )
        prefix = "盒子狀態" if shifted else "兩盒追蹤"
        return f"{prefix};A={value('a')};B={value('b')};事件={events};查詢={value('query')}"
    if task == "sum":
        prefix = "整數加總" if shifted else "三數加法"
        return f"{prefix};x={value('x')};y={value('y')};z={value('z')}"
    if task == "card":
        prefix = "查卡片目前進度" if shifted else "卡片進度"
        return prefix + ";申請編號=" + value("application")
    if task == "address":
        return "我想瞭解地址怎麼填" if shifted else "地址填寫規則"
    if task == "app":
        prefix = "App密碼忘記了" if shifted else "App重設密碼"
        return prefix + ";裝置=" + value("device")
    return "幫我預報明天天氣" if shifted else "天氣預測"


def box_strata(world):
    source = {"A": "initial", "B": "initial"}
    write_seen = False
    write_then_swap = False
    for box, _ in world["events"]:
        if box == "swap":
            write_then_swap = write_then_swap or write_seen
            source["A"], source["B"] = source["B"], source["A"]
        else:
            write_seen = True
            source[box] = "write"
    writes = [(box, n) for box, n in world["events"] if box != "swap"]
    return {
        "length": len(world["events"]),
        "query": world["query"],
        "answer_from_initial": source[world["query"]] == "initial",
        "last_write_to_other_box": bool(writes) and writes[-1][0] != world["query"],
        "write_then_swap": write_then_swap,
        "repeated_swap": sum(box == "swap" for box, _ in world["events"]) >= 2,
    }


def arithmetic_worlds():
    for triple in itertools.combinations_with_replacement(range(7), 3):
        fid = family("sum", triple)
        for permutation in sorted(set(itertools.permutations(triple))):
            yield {"task": "sum", "x": permutation[0], "y": permutation[1], "z": permutation[2], "family_id": fid}


def choose_salt():
    """只用事前標籤覆蓋選salt，尚未訓練、更沒有測試成績可挑。"""
    worlds = list(arithmetic_worlds())
    for i in range(10000):
        salt = f"{VERSION}-coverage-{i}"
        labels = {answer(w) for w in worlds if split_of(w["family_id"], salt) == "train"}
        families = Counter(split_of(fid, salt) for fid in {w["family_id"] for w in worlds})
        if labels == set(range(19)) and all(families[s] >= 4 for s in SPLITS[1:]):
            return salt
    raise RuntimeError("事前算術覆蓋條件沒有成立；不可默默換評估規則")


def box_worlds(per_length=6000, stress_per_length=800, seed=6021):
    rng = random.Random(seed)
    for length in range(1, 7):
        if length == 1:
            facts = [(a, b, (event,)) for a in range(8) for b in range(8) for event in EVENTS]
        else:
            count = per_length if length <= 4 else stress_per_length
            unique = set()
            while len(unique) < count:
                unique.add((rng.randrange(8), rng.randrange(8), tuple(rng.choice(EVENTS) for _ in range(length))))
            facts = sorted(unique)
        for a, b, events in facts:
            fid = family("boxes", (a, b, events))
            for query in ("A", "B"):
                yield {"task": "boxes", "a": a, "b": b, "events": events, "query": query, "family_id": fid}


def service_worlds(count=700, seed=1349):
    rng = random.Random(seed)
    for task in ("card", "address", "app", "none"):
        for i in range(count):
            facts = {"task": task, "entity": f"{task}-{i}"}
            if task == "card":
                facts["application"] = f"{rng.randrange(100000, 999999)}"
            if task == "app":
                facts["device"] = ("手機", "平板", "電腦")[i % 3]
            yield {**facts, "family_id": family(task, facts)}


def judgment(world, omitted, kind, candidates):
    if kind == "choice":
        return world["task"] if world["task"] in candidates else "none"
    needed = REQUIRED[world["task"]]
    available = sum(key not in omitted and present(world.get(key)) for key in needed)
    if kind == "info":
        return "yes" if available == len(needed) else "no"
    return "grade2" if available == len(needed) else "grade0" if available == 0 else "grade1"


def c_rows(world, split, shifted=False):
    task = world["task"]
    variants = [()]
    if task in REQUIRED:
        variants += [(key,) for key in REQUIRED[task]]
    for omitted in variants:
        state = render(world, omitted, shifted)
        for kind in QUESTIONS if task != "none" else ("choice",):
            candidates = list(CANDIDATES[kind])
            gold = judgment(world, omitted, kind, candidates)
            base = {
                "family_id": world["family_id"],
                "case_id": digest(world["family_id"])[:20],
                "split": split,
                "task": task,
                "state": state,
                "question": kind,
                "candidates": candidates,
                "gold": gold,
                "omitted": list(omitted),
                "shifted": shifted,
            }
            yield {**base, "id": digest(base)}
            if kind == "choice" and not omitted and task != "none":
                reduced = [x for x in candidates if x != task]
                missing = {**base, "candidates": reduced, "gold": "none", "missing_best": True}
                yield {**missing, "id": digest(missing)}


def build_rows(per_length=6000, stress_per_length=800, services=700):
    salt = choose_salt()
    boxes = list(box_worlds(per_length, stress_per_length))
    sums = list(arithmetic_worlds())
    b = []
    for world in boxes + sums:
        split = (
            "long_shift"
            if world["task"] == "boxes" and len(world["events"]) > 4
            else split_of(world["family_id"], salt)
        )
        base = {
            "family_id": world["family_id"],
            "case_id": digest(world["family_id"])[:20],
            "split": split,
            "task": world["task"],
            "state": render(world),
            "gold": answer(world),
            "world": world,
        }
        if world["task"] == "boxes":
            base["strata"] = box_strata(world)
        b.append({**base, "id": digest((base["family_id"], base["state"]))})
        if split == "test":
            shifted = {**base, "split": "wording_shift", "state": render(world, shifted=True)}
            b.append({**shifted, "id": digest((base["family_id"], shifted["state"]))})
            if world["task"] == "boxes":
                renamed = {**base, "split": "entity_shift", "state": base["state"].replace("A", "P").replace("B", "Q")}
                b.append({**renamed, "id": digest((base["family_id"], renamed["state"]))})
    # C沿用同一批world與split；固定抽樣減少重複訓練成本，不依測試成績挑題。
    c_worlds = [w for w in boxes if len(w["events"]) <= 4 and int(digest(w["family_id"])[:8], 16) % 5 == 0]
    c_worlds += sums + list(service_worlds(services))
    c = []
    for world in c_worlds:
        split = split_of(world["family_id"], salt)
        c.extend(c_rows(world, split))
        if split == "test":
            c.extend(c_rows(world, "wording_shift", shifted=True))
    return (
        b,
        c,
        {
            "version": VERSION,
            "split_salt": salt,
            "generator_seed": 6021,
            "per_length": per_length,
            "stress_per_length": stress_per_length,
            "services": services,
            "salt_selection": "Before training, require all19 arithmetic answer labels in train and >=4 arithmetic families in each other core split. No metric used.",
            "case_id_policy": "Metadata only, never supplied to model; semantic duplicate checks cannot be passed by random IDs.",
            "family_policy": "All B/C/question/candidate descendants inherit the same world family split. Sorted arithmetic triples share one family; query changes do not change box family.",
        },
    )


def observable_key(row, product):
    if product == "b":
        return canonical(row["state"])
    return canonical((row["state"], row["question"], sorted(row["candidates"])))


def audit_rows(b, c):
    result = {}
    combined_families = defaultdict(set)
    for product, rows in (("b", b), ("c", c)):
        observations = defaultdict(lambda: {"splits": set(), "labels": set(), "rows": 0})
        counts = Counter()
        families = defaultdict(set)
        for row in rows:
            if row["split"] not in SPLITS:
                continue
            key = observable_key(row, product)
            observations[key]["splits"].add(row["split"])
            observations[key]["labels"].add(row["gold"])
            observations[key]["rows"] += 1
            combined_families[row["family_id"]].add(row["split"])
            counts[(row["split"], row["task"]) if product == "b" else (row["split"], row["question"])] += 1
            families[row["split"]].add(row["family_id"])
        conflicts = [key for key, value in observations.items() if len(value["labels"]) > 1]
        shared = [value for value in observations.values() if len(value["splits"]) > 1]
        if conflicts:
            raise ValueError(f"{product}同一可觀測題目有衝突答案")
        if product == "b" and shared:
            raise ValueError("B核心輸入跨切分重複，不能以metadata ID掩蓋")
        result[product] = {
            "counts": {"/".join(k): n for k, n in sorted(counts.items())},
            "family_counts": {s: len(families[s]) for s in SPLITS},
            "unique_observations": len(observations),
            "cross_split_shared_observations": len(shared),
            "rows_in_shared_observations": sum(x["rows"] for x in shared),
            "conflicting_labels": len(conflicts),
            "scope": "B complete core inputs are observation-disjoint. C includes repeated, same-label fixed-grammar generic observations; heldout families do not imply unseen templates or meanings.",
        }
    if any(len(splits) != 1 for splits in combined_families.values()):
        raise ValueError("B/C同一底層家族跨split")
    result["joint_family_overlap"] = 0
    result["arithmetic_labels"] = {
        s: dict(sorted(Counter(row["gold"] for row in b if row["split"] == s and row["task"] == "sum").items()))
        for s in SPLITS
    }
    result["box_strata"] = {
        s: dict(
            Counter(
                name
                for row in b
                if row["split"] == s and row["task"] == "boxes"
                for name, present in row["strata"].items()
                if type(present) is bool and present
            )
        )
        for s in SPLITS
    }
    return result


def write_dataset(directory, **options):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    b, c, config = build_rows(**options)
    audit = audit_rows(b, c)
    files = {}
    for product, rows in (("b", b), ("c", c)):
        path = directory / f"{product}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }
    manifest = {"config": config, "audit": audit, "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def read_dataset(directory, product, split):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    path = directory / f"{product}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]["sha256"]:
        raise ValueError("資料不符合固定manifest")
    return [row for line in path.read_text().splitlines() if (row := json.loads(line))["split"] == split]
