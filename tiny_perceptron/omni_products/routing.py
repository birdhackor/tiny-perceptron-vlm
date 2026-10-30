"""Auto100 的可見文字家族與有限 B adapter；router 不讀媒體或 gold。"""

import itertools
import random
from dataclasses import dataclass

from .choice import Candidate, Question
from .data import SPLIT_SEED, c_splits, fingerprint, visible_c_facts
from .questions import family_questions, state_text

PRODUCTS = (
    Candidate(
        "A",
        "Limited assistant: two-number calculator, clothing, spatial images, OCR, voice, and gray video+audio shape.",
    ),
    Candidate("B", "Fixed looped solver: sum of exactly three integers, each 0 through 6; or bounded two-box events."),
    Candidate(
        "C",
        "Service decision: choose device action or judge information/grade from explicit permission and ready facts.",
    ),
)


@dataclass(frozen=True)
class Request:
    text: str
    image_present: bool = False
    audio_present: bool = False
    video_present: bool = False

    def router_state(self):
        if not all(type(value) is bool for value in (self.image_present, self.audio_present, self.video_present)):
            raise ValueError("附件 presence 必須是 bool")
        return f"request: {self.text}\nimage={int(self.image_present)} audio={int(self.audio_present)} video={int(self.video_present)}"


def routing_question(request, gold=None):
    if type(request) is not Request:
        raise ValueError("router 只收公開文字和 presence；不接受媒體內容／gold task")
    return Question(
        request.router_state(),
        "choice",
        "Which available product supports this request? Choose none if outside all three finite contracts.",
        PRODUCTS,
        gold,
    ).validate()


def b_sum_adapter(values):
    """僅改寫合法公開輸入文法；不計算答案、不更動原 B 權重／tokenizer／深度。"""
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("B 只收恰三個整數")
    if any(type(value) is not int or value not in range(7) for value in values):
        raise ValueError("B 三數值域固定 0–6")
    a, b, c = values
    return f"三數加法;x={a};y={b};z={c}"


def b_families():
    values = list(itertools.combinations_with_replacement(range(7), 3))
    random.Random(SPLIT_SEED + 3).shuffle(values)
    return {"auto": values[:30], "train": values[30:], "retention": values}


def _calculator(operation, a, b):
    word = {"add": "相加", "subtract": "相減", "multiply": "相乘"}[operation]
    return Request(f"請用計算器把 {a} 和 {b} {word}。")


def auto_panel():
    rows = []
    for index, family in enumerate(c_splits()["auto"]):
        facts = visible_c_facts(family, index)
        state = state_text(facts)
        query = family_questions(family, index)
        request = Request(f"服務決策：{state} Choose A action, check B information, grade C.")
        rows.append(
            {
                "family": "auto-C-" + fingerprint(family),
                "parent": ["C", list(family)],
                "request": request,
                "gold_route": "C",
                "payload": query,
                "gold": {kind: question.gold for kind, question in query.items() if kind != "choice_info"},
            }
        )
    for values in b_families()["auto"]:
        request = Request(f"計算三個整數的和：{values[0]}、{values[1]}、{values[2]}。")
        rows.append(
            {
                "family": "auto-B-" + fingerprint(values),
                "parent": ["old-B-sum84", list(values)],
                "request": request,
                "gold_route": "B",
                "payload": values,
                "gold": sum(values),
            }
        )
    for i in range(15):
        operation = ("add", "subtract", "multiply")[i % 3]
        a, b = i + 3, i + 17
        gold = a + b if operation == "add" else a - b if operation == "subtract" else a * b
        rows.append(
            {
                "family": f"auto-A-calculator-{operation}-{a}-{b}",
                "parent": ["old-A-calculator", operation, a, b],
                "request": _calculator(operation, a, b),
                "gold_route": "A",
                "payload": {"operation": operation, "a": a, "b": b},
                "gold": gold,
            }
        )
    media = (
        ("clothing", Request("辨識這張圖片裡的衣物類別。", image_present=True)),
        ("relation", Request("圖片裡兩個物件的相對位置是甚麼？", image_present=True)),
        ("ocr", Request("讀出圖片中的文字。", image_present=True)),
        ("voice", Request("回答這段語音裡的問題。", audio_present=True)),
        ("video-audio", Request("發聲時，灰階影片畫面是圓形還是方形？", audio_present=True, video_present=True)),
    )
    for name, request in media:
        rows.append(
            {
                "family": "auto-A-" + name,
                "parent": ["old-A" if name != "video-audio" else "new-A-final", name],
                "request": request,
                "gold_route": "A",
                "payload": {"requires_frozen_media_binding": name},
                "gold": None,
            }
        )
    unsupported = (
        "畫一張貓的圖片。",
        "合成自然真人歌聲。",
        "預報台北明天天氣。",
        "查詢即時股價。",
        "把影片裡的人臉命名。",
        "辨識灰階照片裡原本是紅色還是藍色。",
        "計算四個整數1、2、3、4的和。",
        "把兩個小數0.5和0.7相乘。",
        "用計算器將100和2相加。",
        "解方程式x平方加1等於0。",
        "將PDF翻譯為日文並寄信。",
        "在我的帳戶執行銀行匯款。",
        "播放任意自然語音長達十分鐘。",
        "判定醫療影像診斷。",
        "處理一百個盒子的狀態。",
        "替未知裝置D決定未定義的服務規則。",
        "計算三個整數7、8、9的和。",
        "為聲音程式生成5000Hz連續十秒音訊。",
        "讀取未提供的圖片。",
        "操作遠端瀏覽器購買車票。",
    )
    for index, text in enumerate(unsupported):
        rows.append(
            {
                "family": f"auto-unsupported-{index}",
                "parent": ["unsupported-public-capability", text],
                "request": Request(text),
                "gold_route": "none",
                "payload": None,
                "gold": "unsupported",
            }
        )
    keys = [fingerprint(row["request"].router_state()) for row in rows]
    if len(rows) != 100 or len(set(keys)) != 100:
        raise ValueError("Auto 必須有 100 個不同可見 router 家族")
    for row in rows:
        routing_question(row["request"], row["gold_route"])
    return rows


def training_routes():
    rows = []
    held = {fingerprint(row["request"].router_state()) for row in auto_panel()}
    held_calculator = {
        tuple(row["parent"][1:]) for row in auto_panel() if row["family"].startswith("auto-A-calculator")
    }
    for operation in ("add", "subtract", "multiply"):
        for a in range(20):
            b = 30 + a
            if (operation, a, b) not in held_calculator:
                rows.append(routing_question(_calculator(operation, a, b), "A"))
    for values in b_families()["train"]:
        rows.append(routing_question(Request(f"計算三個整數的和：{values[0]}、{values[1]}、{values[2]}。"), "B"))
    # 附件存在也是可見輸入；訓練有支援／不支援附件需求，才能檢查真 Auto 路由。
    for request in (
        Request("這份灰階衣物照片請辨識品類。", image_present=True),
        Request("請看兩格圖片，描述物件的左或右關係。", image_present=True),
        Request("請讀指定區域裡的中文字。", image_present=True),
        Request("語音問的是哪個有限客服問題？", audio_present=True),
        Request("影片和嗶聲請判斷同步時的圓或方。", video_present=True, audio_present=True),
    ):
        rows.append(routing_question(request, "A"))
    for index, family in enumerate(c_splits()["train"]):
        panel = ("complete", "removed", "missing")[index % 3]
        state = state_text(visible_c_facts(family, index, panel))
        rows.append(routing_question(Request(f"服務決策：{state} Choose A action, check B information, grade C."), "C"))
    # unsupported 的 train 概念與 Auto20 不同；全組措辭變體必須同側。
    for text in (
        "控制真實機器人移動。",
        "製作3D動畫電影。",
        "改寫作業系統核心。",
        "從衛星影像尋找油田。",
        "生成無限長聲音。",
        "精準預言明年選舉。",
    ):
        rows.append(routing_question(Request(text), "none"))
    for request in (
        Request("根據照片作生物DNA鑑定。", image_present=True),
        Request("讓這段聲音控制真實機器人。", audio_present=True),
        Request("根據整段電影重新製作3D動畫。", video_present=True, audio_present=True),
    ):
        rows.append(routing_question(request, "none"))
    if any(fingerprint(row.state) in held for row in rows):
        raise ValueError("Auto 可見輸入混入 router train")
    return rows
