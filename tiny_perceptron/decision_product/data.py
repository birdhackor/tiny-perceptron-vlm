"""可見事實分組、不同目標裝置與中英文改寫；規則僅用於產生標籤。"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass

ACTIONS = ("stop", "wait", "record", "play", "save")
NAMES = tuple("ABCDEFGH")
KINDS = ("choice", "info", "rubric")
SOURCES = ("compact", "sentence", "chinese", "status")
BOUNDARIES = {
    "train": (0, 800),
    "dev": (800, 880),
    "calibration": (880, 920),
    "policy": (920, 960),
    "final": (960, 1000),
}


@dataclass(frozen=True)
class Example:
    state: str
    question: str
    candidates: tuple[tuple[str, str], ...]
    gold: str
    kind: str
    family: str
    source: str
    panel: str

    def to_dict(self):
        return asdict(self)


def family_key(facts):
    # 名稱、排列、文字、被問的裝置與候選不能把同一可見世界拆到不同 split。
    canonical = sorted(tuple(row) for row in facts)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def partition(key):
    value = int(hashlib.sha256(("c-product-v3-visible-facts-1:" + key).encode()).hexdigest(), 16) % 1000
    return next(name for name, (lo, hi) in BOUNDARIES.items() if lo <= value < hi)


def action_gold(fact):
    task, permission, ready = fact
    if permission == -1 or ready == -1:
        return "none"
    if permission == 0:
        return "stop"
    return "wait" if ready == 0 else ("record", "play", "save")[task]


def info_gold(fact, for_choice=False):
    task, permission, ready = fact
    required = (permission, ready) if for_choice or task == 0 else (permission,) if task == 1 else (ready,)
    return "yes" if -1 not in required else "no"


def rubric_gold(fact):
    task, permission, ready = fact
    if -1 in (permission, ready):
        return "none"
    score = (
        permission + ready if task == 0 else (2 if permission else ready) if task == 1 else (2 if ready else permission)
    )
    return str(score)


def make_families(split, count, seed, min_devices=4, max_devices=6):
    if split not in BOUNDARIES or count < 1 or not 1 <= min_devices <= max_devices <= 6:
        raise ValueError("split 或 count 不合法")
    rng, result, seen = random.Random(seed), [], set()
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > max(20000, count * 500):
            raise ValueError("此 family 空間不足以產生要求的不同樣本")
        # 目標的六種動作結果平衡，避免 stop/none 掩蓋少見動作。
        desired = rng.randrange(6)
        task = rng.randrange(3)
        if desired == 0:
            target = (task, 0, rng.randrange(2))
        elif desired == 1:
            target = (task, 1, 0)
        elif desired < 5:
            target = (desired - 2, 1, 1)
        else:
            values = [rng.randrange(2), rng.randrange(2)]
            values[rng.randrange(2)] = -1
            target = (task, *values)
        facts = [target]
        for _ in range(rng.randint(min_devices, max_devices) - 1):
            facts.append(
                (rng.randrange(3), rng.choices((-1, 0, 1), (1, 5, 5))[0], rng.choices((-1, 0, 1), (1, 5, 5))[0])
            )
        key = family_key(facts)
        if key in seen or partition(key) != split:
            continue
        seen.add(key)
        result.append({"key": key, "facts": facts})
    return result


def render_state(facts, names, order, source, rng):
    rows = []
    for index in order:
        task, p, r = facts[index]
        action = ("record", "play", "save")[task]
        permission, ready = ({-1: "unknown", 0: "no", 1: "yes"}[value] for value in (p, r))
        name = names[index]
        if source == "compact":
            fields = [f"task={action}", f"permission={permission}", f"ready={ready}"]
            rng.shuffle(fields)
            rows.append(f"{name}: " + "; ".join(fields) + ".")
        elif source == "sentence":
            fields = [f"permission is {permission}", f"readiness is {ready}", f"requested task is {action}"]
            rng.shuffle(fields)
            rows.append(f"Device {name}: " + ", ".join(fields) + ".")
        elif source == "chinese":
            cn = {-1: "未知", 0: "否", 1: "是"}
            fields = [f"任務{action}", f"許可{cn[p]}", f"就緒{cn[r]}"]
            rng.shuffle(fields)
            rows.append(f"裝置{name}：" + "；".join(fields) + "。")
        elif source == "status":
            fields = [f"requested action {action}", f"readiness {ready}", f"permission {permission}"]
            rng.shuffle(fields)
            rows.append(f"Status for {name}: " + ", ".join(fields) + ".")
        else:
            raise ValueError("未登記的 source")
    return " ".join(rows)


def make_example(family, rng, kind=None, source=None, panel=None, focused=False):
    facts = family["facts"]
    names = rng.sample(NAMES, len(facts))
    order = rng.sample(range(len(facts)), len(facts))
    source = source or rng.choice(SOURCES)
    kind = kind or rng.choice(KINDS)
    name, target = names[0], facts[0]
    state = render_state(facts, names, order, source, rng)
    chinese = source == "chinese"
    if kind == "choice":
        gold = action_gold(target)
        removed = rng.random() < 0.1 if panel is None else panel == "removed"
        available = [(entity, action) for entity in names for action in ACTIONS]
        correct = (name, gold)
        available = [item for item in available if item != correct]
        # 候選數獨立抽樣；缺答案也可能有七項，完整答案也可能只有一項。
        count = rng.randint(1, min(7, len(available) + (gold != "none" and not removed)))
        needed = count - (gold != "none" and not removed)
        if focused:
            local = [item for item in available if item[0] == name]
            options = rng.sample(local, min(len(local), needed))
            if len(options) < needed:
                options += rng.sample([item for item in available if item not in options], needed - len(options))
        else:
            options = rng.sample(available, needed)
        if gold != "none" and not removed:
            options.append(correct)
        rng.shuffle(options)
        candidates = tuple(
            (f"option-{i}", f"裝置{entity}：{action}" if chinese else f"Device {entity}: {action}")
            for i, (entity, action) in enumerate(options)
        )
        gold_id = f"option-{options.index(correct)}" if correct in options and gold != "none" else "none"
        question = (
            f"選擇裝置{name}的動作。許可與就緒都必須已知；許可否先stop；否則就緒否wait；否則執行其任務。"
            if chinese
            else f"Choose {name}'s action. Both permission and ready must be known. No permission: stop; otherwise not ready: wait; otherwise requested task."
        )
        panel = "missing" if gold == "none" else "removed" if removed else "complete"
    elif kind == "info":
        gold_id = info_gold(target)
        question = (
            f"裝置{name}的必要資訊齊全嗎？record需要許可與就緒；play需要許可；save需要就緒。"
            if chinese
            else f"Is required information known for {name}? record needs permission and ready; play needs permission; save needs ready."
        )
        candidates = (
            (("yes", "必要資訊齊全"), ("no", "必要資訊缺漏"))
            if chinese
            else (("yes", "Required information is sufficient"), ("no", "Required information is missing"))
        )
        candidates = tuple(rng.sample(candidates, len(candidates)))
        panel = "missing" if -1 in target else "complete"
    elif kind == "rubric":
        gold_id = rubric_gold(target)
        question = (
            f"評分裝置{name}。兩欄須已知。record計算是的欄位數；play許可是得2否則看就緒0或1；save就緒是得2否則看許可0或1。"
            if chinese
            else f"Grade {name}. Both fields must be known. record: count yes fields; play: 2 if permission else ready(0/1); save: 2 if ready else permission(0/1)."
        )
        candidates = tuple(
            (str(i), f"裝置{name}得{i}分" if chinese else f"Device {name} has grade {i}")
            for i in rng.sample(range(3), 3)
        )
        panel = "missing" if gold_id == "none" else "complete"
    else:
        raise ValueError("未支援的 kind")
    return Example(state, question, candidates, gold_id, kind, family["key"], source, panel)


def evaluation_examples(families, seed):
    rng = random.Random(seed)
    # 同一 family 的三題是相關觀察；統計一律按 kind 分開計算。
    return [
        make_example(family, rng, kind=kind, source=SOURCES[i % len(SOURCES)])
        for i, family in enumerate(families)
        for kind in KINDS
    ]
