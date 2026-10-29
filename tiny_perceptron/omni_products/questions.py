"""C 家族內的可見事實、三型代表題與次要變體；不沿用舊 parse_state。"""

import random

from .choice import Candidate, Question
from .data import SPLIT_SEED, c_splits, fingerprint, visible_c_facts

ACTIONS = ("record", "play", "save")
ALL_ACTIONS = ("stop", "wait", *ACTIONS)


def state_text(facts, source="compact"):
    rows = []
    for name, fact in zip(("A", "B", "C"), facts):
        permission = {True: "yes", False: "no", None: "unknown"}[fact["permission"]]
        ready = {True: "yes", False: "no", None: "unknown"}[fact["ready"]]
        task = ACTIONS[fact["task"]]
        if source == "compact":
            rows.append(f"{name}: task={task}; permission={permission}; ready={ready}.")
        elif source == "sentence":
            rows.append(f"Device {name} requests {task}. Its permission is {permission}; its readiness is {ready}.")
        elif source == "chinese":
            chinese = {True: "是", False: "否", None: "未知"}
            rows.append(f"裝置{name}：任務{task}；許可{chinese[fact['permission']]}；就緒{chinese[fact['ready']]}。")
        elif source == "held-out":
            rows.append(f"Status for {name}: requested action {task}, readiness {ready}, permission {permission}.")
        else:
            raise ValueError("未登記 source")
    return " ".join(rows)


def choice_gold(fact):
    if fact["permission"] is None or fact["ready"] is None:
        return "none"
    if not fact["permission"]:
        return "stop"
    if not fact["ready"]:
        return "wait"
    return ACTIONS[fact["task"]]


def choice_question(state, fact, entity="A", removed=False, wording=0):
    gold = choice_gold(fact)
    candidates = tuple(
        Candidate(action, f"Device {entity}: {action}") for action in ALL_ACTIONS if not removed or action != gold
    )
    text = f"Choose {entity}'s action. Require permission and ready known; no permission: stop; not ready: wait; else requested task."
    if wording == 1:
        text = f"For {entity}, pick action: both fields required. permission=no -> stop; ready=no -> wait; otherwise do the requested task."
    return Question(state, "choice", text, candidates, "none" if removed else gold).validate()


def info_question(state, fact, entity="B", for_choice=False):
    required = (
        ("permission", "ready")
        if for_choice or fact["task"] == 0
        else (("permission",) if fact["task"] == 1 else ("ready",))
    )
    enough = all(fact[key] is not None for key in required)
    if for_choice:
        text = f"Are permission and ready both known for {entity}'s action?"
    else:
        text = f"Does {entity} have all required fields known? record needs permission+ready; play needs permission; save needs ready."
    return Question(
        state,
        "info",
        text,
        (Candidate("yes", "Required information is sufficient"), Candidate("no", "Required information is missing")),
        "yes" if enough else "no",
    ).validate()


def rubric_question(state, fact, entity="C"):
    permission, ready = fact["permission"], fact["ready"]
    if permission is None or ready is None:
        raise ValueError("rubric 的兩個欄位必須已知")
    if fact["task"] == 0:
        score = int(permission) + int(ready)
    elif fact["task"] == 1:
        score = 2 if permission else int(ready)
    else:
        score = 2 if ready else int(permission)
    text = f"Grade {entity}: record=count yes fields; play=2 if permission else ready(0/1); save=2 if ready else permission(0/1)."
    return Question(
        state, "rubric", text, tuple(Candidate(str(i), f"Device {entity} has grade {i}") for i in range(3)), str(score)
    ).validate()


def family_questions(family, index=0, source="compact", variant=0, primary_panel="complete"):
    # B 的 sufficiency 事先平衡，不能拿它替 A 的對應 info 預測。
    facts = visible_c_facts(family, index, primary_panel)
    state = state_text(facts, source)
    choice = choice_question(state, facts[0], removed=primary_panel == "removed", wording=variant % 2)
    rng = random.Random(SPLIT_SEED + variant + int(fingerprint(family)[:8], 16))
    candidates = list(choice.candidates)
    rng.shuffle(candidates)
    choice = Question(choice.state, choice.kind, choice.text, tuple(candidates), choice.gold).validate()
    return {
        "choice": choice,
        "info": info_question(state, facts[1]),
        "rubric": rubric_question(state, facts[2]),
        "choice_info": info_question(state, facts[0], "A", for_choice=True),
    }


def panel_c(split):
    if split not in {"calibration", "policy", "final", "dev"}:
        raise ValueError("僅提供預定的 reliability／development panel")
    rows = []
    for index, family in enumerate(c_splits()[split]):
        panel = (
            "missing"
            if index >= 54 and split in {"policy", "final"}
            else "removed"
            if index >= 48 and split in {"policy", "final"}
            else "complete"
        )
        # final30 完整 source 留出；其餘 source 僅稱開發來源新組合。
        source = "held-out" if split == "final" and index >= 30 else ("compact", "sentence", "chinese")[index % 3]
        rows.append(
            {
                "family": fingerprint(family),
                "parent": list(family),
                "source": source,
                "panel": panel,
                "questions": family_questions(family, index, source, primary_panel=panel),
            }
        )
    return rows


def training_questions():
    rows = []
    for index, family in enumerate(c_splits()["train"]):
        for variant, source in enumerate(("compact", "sentence", "chinese")):
            panel = ("complete", "removed", "missing")[index % 3]
            rows.extend(family_questions(family, index + variant, source, variant, panel).values())
    return rows
