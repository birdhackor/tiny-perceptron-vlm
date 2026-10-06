#!/usr/bin/env python3
"""Write original, finite Chinese conversations and real calculator traces.

No pretrained model, external transcript, or old capstone answer is used. The
short question families and answers below are authored in this file. Their
semantic scope is inspectable; human review is explicitly pending in the
manifest. Family assignment happens before conversation expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.selftrained.tools import (  # noqa: E402
    OPERATIONS,
    TOOLS_SCHEMA,
    CalculatorCall,
    ToolCallError,
    execute_tool_call,
    parse_tool_call,
    run_tool_loop,
    serialize_tool_call,
    serialize_tool_result,
)

VERSION = "selftrained-text-tools-v1"
AUTHOR = "project-authored-by-codex-phase5; human-review-pending"
BASE_SYSTEM = "你是有限中文助手，支援問候、格式、選項記憶與地址、App、卡片三類客服問題；資訊不足時先詢問。"
FORMATS = ("one_sentence", "two_points")
FORMAT_REQUESTS = {
    "one_sentence": ("接下來請用一句回答。", "之後只用一句話回覆。"),
    "two_points": ("接下來請用兩點回答。", "之後請分成兩點回覆。"),
}
FORMAT_ACK = {"one_sentence": "好，接下來用一句回答。", "two_points": "好，接下來用兩點回答。"}
ANSWERS = {
    "greeting": {
        "one_sentence": "你好，我可以協助教過的有限中文需求。",
        "two_points": "1. 你好。\n2. 我可以協助教過的有限中文需求。",
    },
    "scope": {
        "one_sentence": "我支援有限中文聊天、指定格式、選項記憶、三類客服與0到99整數的加減乘。",
        "two_points": "1. 我支援有限中文聊天、指定格式、選項記憶與三類客服。\n2. 計算器支援0到99整數的加減乘。",
    },
    "address": {
        "one_sentence": "請透過官方管道查詢地址更新方式。",
        "two_points": "1. 請使用官方管道。\n2. 詢問地址更新方式。",
    },
    "app_error": {
        "one_sentence": "先檢查網路並重新啟動App，仍失敗再詢問官方客服。",
        "two_points": "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。",
    },
    "card_issues": {
        "one_sentence": "請先確認卡片無法使用的情況，再詢問官方客服。",
        "two_points": "1. 請先確認卡片無法使用的情況。\n2. 再詢問官方客服。",
    },
    "clarification": {
        "one_sentence": "請說明是地址、App還是卡片的問題。",
        "two_points": "1. 請說明遇到的問題。\n2. 是地址、App還是卡片？",
    },
    "out_of_scope": {
        "one_sentence": "這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。",
        "two_points": "1. 這超出我的範圍。\n2. 我能協助有限聊天、地址、App、卡片或加減乘問題。",
    },
}

# Each pair is one original paraphrase family, not independently split rows.
# Indices 0..3 train, 4..5 validation, 6..7 test. Their full trees stay together.
QUESTION_FAMILIES = {
    "greeting": (
        ("你好，想和你打個招呼。", "哈囉，我來打招呼。"),
        ("早安，請回我一聲。", "早上好，和我問好吧。"),
        ("午安，今天先向你問好。", "下午好，請向我問好。"),
        ("晚安，見到你很開心。", "晚上好，我想先問候你。"),
        ("嗨，見面先說你好。", "嗨，請和我打聲招呼。"),
        ("我來了，請問候我。", "我到了，請回個問候。"),
        ("很高興見到你，請打招呼。", "又見面了，請向我問好。"),
        ("你好呀，回我一個問候吧。", "哈囉呀，回我一句招呼。"),
    ),
    "scope": (
        ("你支援哪些需求？", "你可以幫哪些忙？"),
        ("請說明你會處理的問題。", "請介紹你支援的問題。"),
        ("你的支援範圍有哪些？", "哪些事情在你的範圍內？"),
        ("我可以向你問什麼？", "可以找你處理什麼事？"),
        ("你教過的能力有哪些？", "你學過哪些有限需求？"),
        ("請列出能協助的事情。", "請列出可以處理的事情。"),
        ("想了解你支援什麼問題。", "想知道能請你協助什麼問題。"),
        ("先介紹你的有限能力吧。", "先說說你的有限支援範圍吧。"),
    ),
    "address": (
        ("搬家後，我想修改登記的地址。", "我搬家了，想改登記地址。"),
        ("住址變了，該如何更新資料？", "住址換了，該怎麼更新資料？"),
        ("我要查詢地址更新的方式。", "想查一下地址怎麼更新。"),
        ("新的地址要如何改到帳戶資料中？", "帳戶中的地址想換成新地址。"),
        ("登記住址需要更改，請給我方向。", "請給我更改登記住址的方向。"),
        ("現在的住址和原先登記的不一樣。", "原先登記的住址已不是現在住址。"),
        ("請問住址改動後要找誰確認？", "住址改動後，想確認更新的方法。"),
        ("我想處理地址資料的變更。", "地址資料變更可以怎樣處理？"),
    ),
    "app_error": (
        ("App打不開，該怎麼辦？", "我開不了App，該怎麼處理？"),
        ("App顯示錯誤，想請你給個方向。", "App出現錯誤，請給處理方向。"),
        ("使用App時一直失敗。", "我在App操作一直失敗。"),
        ("手機上的App停住了。", "我的手機App沒有反應。"),
        ("App無法正常啟動，該先做什麼？", "啟動App不成功，先檢查什麼？"),
        ("我遇到App故障，想知道下一步。", "App故障後可以先試什麼？"),
        ("這個App反覆出錯，請幫我。", "請協助我處理App反覆出錯。"),
        ("App操作不能完成，請給建議。", "在App裡操作不了，請給建議。"),
    ),
    "card_issues": (
        ("我的卡片無法使用，怎麼處理？", "卡片用不了，該怎麼辦？"),
        ("刷卡沒有成功，想請你給方向。", "我刷卡失敗，請給處理方向。"),
        ("這張卡片出了問題。", "我的卡片有使用問題。"),
        ("使用卡片時被拒絕了。", "卡片操作被拒絕，怎麼辦？"),
        ("卡片突然不能用，下一步是什麼？", "卡片不能用了，請給下一步。"),
        ("想確認卡片故障要如何處理。", "想問卡片故障的處理方向。"),
        ("我的卡片操作一直失敗，請幫忙。", "請協助處理卡片操作失敗。"),
        ("付款時卡片沒成功，請給建議。", "卡片付款不成功，請給建議。"),
    ),
    "clarification": (
        ("那個問題要怎麼處理？", "這個問題怎麼辦？"),
        ("它不能用，請幫忙。", "它失敗了，幫我看看。"),
        ("有個問題，能給我方向嗎？", "我有問題，能給個方向嗎？"),
        ("我要改資料，但還沒說是哪項。", "想更改資料，還沒決定哪項。"),
        ("剛才那個出錯了。", "剛才的東西出了問題。"),
        ("請幫我處理一下。", "協助我處理一下吧。"),
        ("我不知道要從哪裡開始處理。", "要開始處理，但我還沒說問題。"),
        ("能告訴我該怎麼做嗎？", "請告訴我可以怎麼做。"),
    ),
    "out_of_scope": (
        ("請預測明天天氣。", "明天的天氣會如何？"),
        ("請幫我寫一篇長小說。", "替我寫長篇故事。"),
        ("請告訴我股票明天價格。", "明天股價是多少？"),
        ("請介紹所有世界歷史。", "能教我整部世界歷史嗎？"),
        ("請幫我診斷身體不舒服。", "能替我作身體診斷嗎？"),
        ("請替我預測比賽的勝負。", "這場比賽誰會贏？"),
        ("我想問今天外面的天氣。", "今天外面天氣怎樣？"),
        ("請寫一整本小說給我。", "請給我完整的長篇小說。"),
    ),
    "preference": (
        ("我剛才選了什麼？", "記得我選哪一個嗎？"),
        ("請說出我的選項。", "請告訴我先前的選擇。"),
        ("剛才的選擇是哪個？", "我先前選的是哪個？"),
        ("把我選的東西再說一次。", "再說一次我的選項。"),
        ("請回想我留下的選擇。", "回覆我之前留下的選項。"),
        ("我之前決定選什麼？", "之前我決定了哪個選項？"),
        ("我剛才選的是哪個？", "請再說一次我的選擇。"),
        ("我說過要選的東西是什麼？", "請指出我說過要的選項。"),
    ),
    "format_change": (
        ("改成一句回答。", "請將剛才的回覆改為一句。"),
        ("改用一句話說。", "現在用一句話回覆。"),
        ("請只用一句重新回答。", "只留一句，重新說一次。"),
        ("把答案改寫成一句。", "請把回覆合成一句。"),
        ("同一個問題請用一句。", "同樣的問題改用一句回覆。"),
        ("剛才的問題只回答一句。", "將剛才問題的答案變成一句。"),
        ("保留原問題，回覆改成一句。", "原本的問題改成一句回答。"),
        ("我想讓剛才的答案只有一句。", "讓上個答案只用一句話。"),
    ),
}

# Additional base prompts are original situation families, written before any
# model fitting. A pair describes one base situation in two ways. It is never
# counted as two independent families. Scope remains the ten finite intents.
EXTRA_QUESTION_FAMILIES = {
    "greeting": {
        "train": (
            ("剛下課，我來向你問好。", "下課了，先和你打個招呼。"),
            ("我第一次使用這個助手，先說你好。", "第一次來這裡，請和我問好。"),
            ("剛起床，想聽一聲問候。", "才起床，請向我打聲招呼。"),
            ("午休時間到了，向你問好。", "午休時先說聲你好。"),
            ("我要休息了，先和你說晚安。", "睡前來說晚安，請回個問候。"),
            ("朋友介紹我來這裡，先打招呼。", "聽朋友介紹來了，先向你問好。"),
        ),
        "validation": (
            ("工作做完了，回來和你問好。", "下班後回到這裡，打個招呼。"),
            ("這是今天第一次見面，請問候我。", "今天第一次碰面，向我問好吧。"),
            ("週末了，來向你說你好。", "週末來了，請回我一個問候。"),
        ),
        "test": (
            ("等車的時候，想向你打招呼。", "我在等車，先和你問好。"),
            ("課前還有一點時間，來說你好。", "上課前先向你問好。"),
            ("旅行回來了，再次向你問好。", "結束旅行回來，先打聲招呼。"),
            ("剛開完會，想和你問好。", "會議結束了，來向你打招呼。"),
            ("我來試用這裡，第一句先說你好。", "第一次試用，先請你打招呼。"),
            ("晚餐吃完，來和你說聲你好。", "飯後來問好，請回個問候。"),
            ("好久沒開這個助手，先問候一下。", "隔了一段時間回來，先和你問好。"),
            ("今天心情平靜，來向你說早安。", "新的一天開始了，先互相問好。"),
        ),
    },
    "scope": {
        "train": (
            ("我是新使用者，想先知道這裡能做什麼。", "初次使用，請介紹能幫忙的事。"),
            ("還沒有具體問題，先說支援哪些事情。", "我還沒選問題，請先介紹範圍。"),
            ("朋友問你會什麼，請給一個介紹。", "我想向朋友介紹你的支援能力。"),
            ("準備開始對話前，請說明可用能力。", "開始問問題之前，先了解支援內容。"),
            ("我只想知道這個有限助手的功能。", "請告訴我有限助手可以處理哪些需求。"),
            ("請讓我知道有哪些類型可以問你。", "我想選問題類型，請說可問的種類。"),
        ),
        "validation": (
            ("在提出需求前，想先確認你有什麼能力。", "先確認能力，再向你提問題。"),
            ("這裡的客服與聊天支援到什麼範圍？", "請概括這裡聊天和客服的可用範圍。"),
            ("我要向同學介紹你，請簡介你的功能。", "同學想知道你能協助哪些事。"),
        ),
        "test": (
            ("我在挑選可問的問題，請給支援範圍。", "還在選問題，想知道哪些類型你會處理。"),
            ("準備用這個助手做練習，有哪些教過的能力？", "練習開始前，請介紹有限能力。"),
            ("只看名字還不了解你，請說明能做什麼。", "我還不了解這裡的功能，請給介紹。"),
            ("希望先看一份簡短的能力介紹。", "先向我概括你能協助的需求。"),
            ("用過一次後還想確認完整的支援範圍。", "再向我介紹一次這裡可處理的事情。"),
            ("我沒有要立刻辦事，先了解支援類型。", "暫時沒有需求，只問可支援什麼。"),
            ("向家人介紹之前，請說明你有哪些功能。", "家人想知道這個助手能處理哪些問題。"),
            ("請給我開始提問前的能力概覽。", "先說可問的範圍，讓我知道如何開始。"),
        ),
    },
    "address": {
        "train": (
            ("郵寄地址已經換了，要怎麼查更新方式？", "收信的地址改了，想查如何更新。"),
            ("剛租新房，帳戶上的地址要更改。", "我租了新住處，想更改帳戶地址。"),
            ("地址資料有錯字，想確認修改途徑。", "登記地址寫錯，該找哪個管道查修改方式？"),
            ("目前和家人同住，想更新我的住址。", "搬去家人住處後，地址要如何更新？"),
            ("工作搬到外地後，住址需要更新。", "因為工作換了居住地方，想修改住址。"),
            ("我已搬回原住處，想改帳戶地址。", "搬回去了，帳戶登記地址要更改。"),
        ),
        "validation": (
            ("更換租屋地點後，想確認地址更新方法。", "租約換了地方，該怎麼查住址修改？"),
            ("新家門牌和舊家不同，登記地址需要改。", "已經換了門牌地址，想更新登記資料。"),
            ("發現帳戶保留的地址已經過時。", "帳戶還是以前住址，請給更新方向。"),
        ),
        "test": (
            ("搬到另一座城市了，想更改登記住址。", "跨城市搬家後，登記地址怎麼更新？"),
            ("住處不變，但地址中的門牌號更正了。", "門牌地址有更正，想查資料更新方式。"),
            ("換了公寓房間，帳戶地址資料要修改。", "搬到另一間公寓，想更新帳戶住址。"),
            ("信還寄去以前的住處，想查地址修改方法。", "舊地址仍在資料裡，如何查變更方式？"),
            ("入學後住進宿舍，需要更改住址資料。", "現在住校，登記住址想換成宿舍。"),
            ("離開宿舍回家後，想更新登記地址。", "不住學校了，想查地址更新的辦法。"),
            ("地址資料缺了樓層，想確認補改的方式。", "登記住址沒寫完整，請給更新方向。"),
            ("換了長期居住地，想知道官方地址更改管道。", "長期住址改變後，該怎麼查更新方法？"),
        ),
    },
    "app_error": {
        "train": (
            ("App登入畫面一直卡住。", "登入時App停住不動，該先做什麼？"),
            ("App一開就關掉了。", "開啟App後它立刻退出。"),
            ("App頁面一直在載入。", "手機App不停載入，無法操作。"),
            ("App只有空白畫面，請給處理方向。", "打開App後看不到內容。"),
            ("按App的按鈕沒有反應。", "App按鈕點下去也不動。"),
            ("更新App後不能正常使用。", "剛更新的App操作失敗。"),
        ),
        "validation": (
            ("App在輸入資料時突然停止。", "填資料的時候App沒反應了。"),
            ("App錯誤訊息一直重複出現。", "每次開App都看到錯誤提示。"),
            ("手機App進不到主畫面。", "App不能進入首頁，請給方向。"),
        ),
        "test": (
            ("App查看帳戶時畫面凍住。", "我在App看帳戶資料時卡住了。"),
            ("App從背景切回來就失去反應。", "回到App後，它不再回應操作。"),
            ("App填完表格卻無法送出。", "表格填好了，App送出失敗。"),
            ("App顯示連線失敗，但我不知道先查什麼。", "手機App連不上，請給檢查方向。"),
            ("App開首頁時跳出錯誤。", "我打開App首頁就遇到錯誤提示。"),
            ("App操作到一半自行退出。", "進行操作時App突然關掉。"),
            ("App頁面按重新載入也沒有內容。", "重新載入後App仍是空白。"),
            ("App顯示一直等待，沒辦法繼續。", "一直在等待的App無法完成操作。"),
        ),
    },
    "card_issues": {
        "train": (
            ("在商店刷卡時，卡片被拒絕。", "店內用卡付款被拒，怎麼處理？"),
            ("機器讀不到我的卡片。", "卡片放上去後機器無法讀取。"),
            ("網路購物時卡片付款失敗。", "線上購物用卡沒有付成功。"),
            ("卡片感應沒有反應。", "使用卡片感應時沒成功。"),
            ("換了付款機器仍然不能用這張卡。", "試了另一台機器，卡片還是無法使用。"),
            ("原本可用的卡片今天刷不過。", "昨天能用的卡今天付款失敗。"),
        ),
        "validation": (
            ("店員說這張卡片交易沒有完成。", "用卡後店家表示交易失敗。"),
            ("購物結帳時卡片不能通過。", "結帳用卡失敗，該問誰？"),
            ("我的卡片在讀卡設備上沒有反應。", "讀卡設備無法辨識這張卡。"),
        ),
        "test": (
            ("搭乘交通付款時卡片讀取失敗。", "用卡付交通費時沒有成功讀取。"),
            ("自助結帳機不接受我的卡片。", "在自助結帳時卡片被拒了。"),
            ("卡片在不同商店都刷不過。", "幾家商店都無法使用這張卡。"),
            ("付款機顯示卡片無法處理。", "刷卡後機器顯示不能處理卡片。"),
            ("這張卡片剛拿到，使用時卻失敗。", "新拿到的卡片在付款時不能用。"),
            ("訂購商品時出現卡片交易錯誤。", "買商品用卡付款顯示交易錯誤。"),
            ("卡片插入機器後，交易沒有繼續。", "插卡後付款機停在失敗畫面。"),
            ("同一張卡片再次付款仍然不成功。", "卡片重新刷了一次還是失敗。"),
        ),
    },
    "clarification": {
        "train": (
            ("我收到一個提示，但沒說是什麼提示。", "有提示出現，內容還沒告訴你。"),
            ("剛才操作失敗，還沒有說是哪個操作。", "有個操作沒成功，類型還沒提供。"),
            ("我要更新一項東西，還沒說是哪項。", "想更新東西，具體項目稍後才說。"),
            ("朋友遇到問題，我還沒問清楚內容。", "朋友說出事了，但我不知道具體情況。"),
            ("有個東西不對，細節還沒提供。", "看起來不太對，還沒有具體資料。"),
            ("我想詢問一個問題，先讓你知道。", "我有事想問，具體題目還沒說。"),
        ),
        "validation": (
            ("我只知道失敗了，其他情況還不清楚。", "只知道沒成功，還不能說是哪類問題。"),
            ("之前提到的事情還沒說明內容。", "有件事情要處理，內容還沒補上。"),
            ("想找客服，但還沒說為什麼。", "我需要客服方向，需求還沒講清楚。"),
        ),
        "test": (
            ("家人說無法使用，但沒說用的是什麼。", "家人有東西不能用，具體物件還不清楚。"),
            ("畫面上有訊息，尚未告訴你訊息內容。", "看到一段提示，但內容還沒提供。"),
            ("想修改某項資料，尚未指出資料名稱。", "資料要修改，但我沒說是哪一項。"),
            ("這次又不成功了，之前的細節還沒說。", "再次失敗，問題細節還未提供。"),
            ("同學讓我問如何處理，卻沒講具體需求。", "同學請我來問問題，問題本身還沒給。"),
            ("我有客服需求，題目還沒有說明。", "要尋求客服協助，具體問題還沒寫。"),
            ("事情進行到一半停住，沒有其他資訊。", "有件事停住了，類型和内容都沒提供。"),
            ("還不確定是哪種問題，想先請你幫忙。", "不知道問題屬於哪類，細節仍不明。"),
        ),
    },
    "out_of_scope": {
        "train": (
            ("替我安排完整的海外旅行。", "請規劃一週出國行程。"),
            ("請解釋深奧的物理理論。", "教我整套高階物理。"),
            ("請替我寫一套網站程式。", "幫我做完整的網站程式碼。"),
            ("推薦我最近上映的電影。", "最近有哪些新電影可看？"),
            ("請提供一份法律案件分析。", "幫我分析複雜法律問題。"),
            ("請把整篇外語文章翻譯成中文。", "幫我翻譯一整篇外文。"),
        ),
        "validation": (
            ("請推薦晚餐食譜與作法。", "教我做一道晚餐料理。"),
            ("請告訴我今天的國際新聞。", "介紹今天各國最新消息。"),
            ("請回答大學化學考題。", "幫我解一整份化學試卷。"),
        ),
        "test": (
            ("教我如何演奏一首鋼琴曲。", "請提供完整的鋼琴彈奏教學。"),
            ("請為我的花園規劃植物配置。", "我想設計整個花園，請給方案。"),
            ("請解說一篇哲學論文。", "幫我分析整篇哲學文章。"),
            ("我想知道今晚球賽比分。", "告訴我最新的比賽結果。"),
            ("請替我比較所有汽車品牌。", "我想知道每個汽車品牌的比較。"),
            ("教我修理家裡的電器。", "家用電器壞了，請提供維修步驟。"),
            ("請查詢國際航班的即時狀況。", "幫我找今天飛機是否準時。"),
            ("我想學會整套高等數學。", "請教我完整的高等數學課程。"),
        ),
    },
    "preference": {
        "train": (
            ("先前提到要選的物件是哪個？", "請回想我指定的物件。"),
            ("照我的決定，留下哪個選項？", "我決定保留的選項是什麼？"),
            ("我想要的那一個叫什麼？", "說出我想留下的物件名稱。"),
            ("依我之前的選擇，現在是哪個？", "以之前的決定為準，我要哪個？"),
            ("請確認我自己選定的物件。", "確認一下我已經選定什麼。"),
            ("告訴我最後決定要的那一個。", "我的最後決定是哪個選項？"),
        ),
        "validation": (
            ("請提醒我剛才指定的選項。", "提醒我先前說要的物件。"),
            ("按我留下的決定，哪個是我要的？", "留下來的選擇是哪一個？"),
            ("我的選擇已說過，請再確認名稱。", "再確認我已經說過的選項名稱。"),
        ),
        "test": (
            ("我說要保留的物件是什麼？", "根據我說的話，應保留哪個？"),
            ("照剛才作的決定，請指出選項。", "我的決定作好了，請告訴我選了什麼。"),
            ("如果現在要拿，應拿我選的哪個？", "要拿我指定的東西，名稱是什麼？"),
            ("若只留下我的選項，會留下什麼？", "只保留我說要的物件，會是哪個？"),
            ("把先前選定的名稱回覆給我。", "我選定的東西叫什麼名字？"),
            ("已經選好了，請回顧我的決定。", "選擇完成了，再告訴我作了什麼決定。"),
            ("請跟著我的選擇指出物件。", "按照我的選擇，指出該保留的名稱。"),
            ("若要確認清單，應寫上我選的什麼？", "我的選項要記在清單，請說出名稱。"),
        ),
    },
    "format_change": {
        "train": (
            ("內容保留，請濃縮成一句。", "不換問題，把答案縮成一句話。"),
            ("我要更短的格式，只用一句。", "格式改短，請以一句回覆。"),
            ("把上一題的重點放在一句裡。", "用一句包含上一題的重點。"),
            ("不增加內容，請改成一句話。", "保持原來意思，格式變成一句。"),
            ("主題照舊，現在請用一句。", "接著同一主題，改用一句回答。"),
            ("我要轉述給朋友，請改為一句。", "準備轉述這個答案，請用一句話。"),
        ),
        "validation": (
            ("剛才回答要放進筆記，請用一句。", "我要記下上一題，請濃縮成一句。"),
            ("同樣的資訊，用一句再說一次。", "不改資訊，重新用一句回答。"),
            ("請把上個問題的回覆寫成一句。", "上題的答覆格式改為一句話。"),
        ),
        "test": (
            ("我只保留一句摘要，請改上個答案。", "請將上題答案整理成一句摘要。"),
            ("要把答覆寫在卡片上，請改成一句。", "卡片只放一句，請整理剛才的答覆。"),
            ("剛才的主題不變，請只寫一句。", "保留剛才主題，以一句重新回答。"),
            ("我想快速轉述上題，請給一句版本。", "上題要快速轉述，請改用一句。"),
            ("請用一句把原先兩個重點串起來。", "將剛才的重點合在一句話裡。"),
            ("我要保存上一題答案，請用一句格式。", "保存答覆時只要一句，請重寫上題。"),
            ("不要另外選題，原題改成一句回覆。", "仍然回答原來的題目，但只用一句。"),
            ("讀過剛才的答覆後，我想看一句版本。", "上個答案已讀過，請給一句的格式。"),
        ),
    },
}


def question_families(intent: str, split: str) -> tuple:
    original = tuple(
        family for index, family in enumerate(QUESTION_FAMILIES[intent]) if split_for_family(index) == split
    )
    return original + EXTRA_QUESTION_FAMILIES[intent][split]


TOOL_TEMPLATES = {
    "train": (
        ("算{a}加{b}是多少。", "算{a}減{b}是多少。", "算{a}乘{b}是多少。"),
        ("請幫我計算{a}+{b}。", "請幫我計算{a}-{b}。", "請幫我計算{a}×{b}。"),
        ("把{a}與{b}相加。", "從{a}扣掉{b}。", "把{a}與{b}相乘。"),
        ("{a}+{b}的結果呢？", "{a}-{b}的結果呢？", "{a}×{b}的結果呢？"),
    ),
    "validation": (
        ("我想知道{a}加上{b}的值。", "我想知道{a}減去{b}的值。", "我想知道{a}乘以{b}的值。"),
        ("用計算器算出{a}+{b}。", "用計算器算出{a}-{b}。", "用計算器算出{a}×{b}。"),
    ),
    "test": (
        ("請回報{a}和{b}的和。", "請回報{a}減{b}的差。", "請回報{a}與{b}的乘積。"),
        ("這次要計算{a}+{b}，請幫忙。", "這次要計算{a}-{b}，請幫忙。", "這次要計算{a}×{b}，請幫忙。"),
    ),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_for_family(index: int) -> str:
    return "train" if index < 4 else "validation" if index < 6 else "test"


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def record(group_id: str, split: str, task: str, messages: list[dict], **supervision) -> dict:
    topic = supervision.get("topic", supervision["intent"])
    semantics = {
        "greeting": ([], ["你好", "哈囉", "嗨"]),
        "scope": (["格式", "選項", "客服"], ["加減乘", "計算器"]),
        "address": (["官方", "地址"], ["更新", "更改"]),
        "app_error": (["網路", "App", "客服"], ["重新啟動", "重啟"]),
        "card_issues": (["卡片", "客服"], ["確認", "情況"]),
        "clarification": (["請"], ["地址", "App", "卡片"]),
        "out_of_scope": (["範圍"], ["超出", "支援", "只能"]),
        "preference": ([supervision.get("selected_option", "")], []),
        "calculation": ([], []),
    }
    required, alternatives = semantics[topic]
    if task == "tool_missing":
        required, alternatives = ["整數"], ["提供", "缺少"]
    elif task == "tool_concept":
        required, alternatives = ["兩個數"], ["合", "相加"]
    elif task == "tool_unsupported":
        required, alternatives = ["0", "99"], ["支援", "範圍"]
    elif task == "tool_unavailable":
        required, alternatives = ["計算器"], ["不可用", "無法使用"]
    supervision.update(semantic_all=required, semantic_any=alternatives)
    if "counterfactual_pair" in supervision:
        supervision["pair_id"] = supervision.pop("counterfactual_pair")
        supervision["control_type"] = "history_choice" if topic == "preference" else "history_topic"
    elif task == "text":
        supervision["pair_id"] = f"{group_id}:format:{digest(messages[-2])[:12]}"
        supervision["control_type"] = "history_format"
    elif task == "tool_reply":
        supervision["pair_id"] = f"{group_id}:{digest(supervision['expected_call'])[:12]}:{supervision['format']}"
        supervision["control_type"] = "tool_result"
    value = {"group_id": group_id, "split": split, "task": task, "messages": messages}
    value["supervision"] = supervision
    value["provenance"] = {
        "source": "project-original",
        "author": AUTHOR,
        "license": "MIT (repository original content)",
        "human_review": "pending",
        "external_teacher": None,
    }
    value["id"] = "txt-" + digest(value)[:24]
    return value


def style_history(style: str, variant: int = 0, *, tool: bool = False, enabled: bool = True) -> list[dict]:
    system = BASE_SYSTEM
    if tool:
        system += TOOLS_SCHEMA if enabled else "計算器不可用；不要心算代替工具，請說明計算器不可用。"
    return [
        message("system", system),
        message("user", FORMAT_REQUESTS[style][variant % 2]),
        message("assistant", FORMAT_ACK[style]),
    ]


def text_records() -> list[dict]:
    rows = []
    for intent in QUESTION_FAMILIES:
        families = [
            (split, index, family)
            for split in ("train", "validation", "test")
            for index, family in enumerate(question_families(intent, split))
        ]
        for split, family_index, questions in families:
            template_family = f"text:{intent}:{split}-original-family-{family_index:02d}"
            # Format-followup rows extend the customer-dialogue roots used by
            # the three direct customer tasks. Keep all shared descendants in
            # the same tree group while retaining each original template ID.
            group = (
                f"text:customer-tree:{split}-original-family-{family_index:02d}"
                if intent in ("address", "app_error", "card_issues", "format_change")
                else template_family
            )
            for question_index, question in enumerate(questions):
                if intent in ANSWERS:
                    for style in FORMATS:
                        rows.append(
                            record(
                                group,
                                split,
                                "text",
                                style_history(style, question_index)
                                + [message("user", question), message("assistant", ANSWERS[intent][style])],
                                intent=intent,
                                format=style,
                                depends_on_history=True,
                                template_family=template_family,
                                rubric="有限語意與指定格式；原創參考答覆不代表實際銀行流程。",
                            )
                        )
                elif intent == "preference":
                    # The heldout question uses a familiar option/format combination;
                    # both counterfactual choices are descendants of this same tree.
                    options = (
                        ("包", "短靴") if split == "test" else ("包", "褲子") if family_index % 2 else ("短靴", "褲子")
                    )
                    for style in FORMATS:
                        for option in options:
                            answer = (
                                f"你選了{option}。"
                                if style == "one_sentence"
                                else f"1. 你選了{option}。\n2. 我記住這個選項。"
                            )
                            for selection_kind in ("single_choice", "updated_choice"):
                                previous = next(value for value in options if value != option)
                                first = (
                                    f"我選{option}"
                                    if selection_kind == "single_choice"
                                    else f"我先選{previous}，後來改選{option}"
                                )
                                first += "，接下來用" + ("一句回答。" if style == "one_sentence" else "兩點回答。")
                                rows.append(
                                    record(
                                        group,
                                        split,
                                        "text",
                                        [
                                            message("system", BASE_SYSTEM),
                                            message("user", first),
                                            message("assistant", "好，我會記住最新選項。"),
                                            message("user", question),
                                            message("assistant", answer),
                                        ],
                                        intent=intent,
                                        format=style,
                                        selected_option=option,
                                        previous_option=previous if selection_kind == "updated_choice" else None,
                                        selection_kind=selection_kind,
                                        depends_on_history=True,
                                        template_family=template_family,
                                        counterfactual_pair=f"{group}:q{question_index}:{style}:{selection_kind}",
                                        generalization="known-words-new-option-combination"
                                        if split == "test" and family_index == 0
                                        else "unseen-authored-phrasing"
                                        if split == "test"
                                        else "history-choice",
                                    )
                                )
                else:
                    for topic in ("address", "app_error", "card_issues"):
                        for style in FORMATS:
                            # An explicit first question supplies the topic; no gold
                            # topic label or answer table is included in the prompt.
                            prior_question = question_families(topic, split)[family_index][question_index]
                            followup = (
                                question
                                if style == "one_sentence"
                                else question.replace("一句話", "兩點").replace("一句", "兩點")
                            )
                            other_style = "two_points" if style == "one_sentence" else "one_sentence"
                            rows.append(
                                record(
                                    group,
                                    split,
                                    "text",
                                    [
                                        message("system", BASE_SYSTEM),
                                        message("user", prior_question),
                                        message("assistant", ANSWERS[topic][other_style]),
                                        message("user", followup),
                                        message("assistant", ANSWERS[topic][style]),
                                    ],
                                    intent=intent,
                                    topic=topic,
                                    format=style,
                                    depends_on_history=True,
                                    template_family=template_family,
                                    counterfactual_pair=f"{group}:q{question_index}:{style}",
                                )
                            )
    return rows


def numeric_families() -> dict[str, list[tuple[int, int]]]:
    # Assign canonical operand families before producing operation/direction/
    # availability/reply descendants. (a,b) and (b,a) never cross splits.
    groups = {"train": [(0, n) for n in range(100)], "validation": [], "test": []}
    candidates = sorted(
        ((a, b) for a in range(1, 100) for b in range(a, 100)), key=lambda pair: digest([VERSION, pair])
    )
    groups["train"].extend(candidates[:80])
    groups["validation"] = candidates[80:104]
    groups["test"] = candidates[104:128]
    # Include the simple 1+2 regression case as a real tool trajectory.
    if (1, 2) not in {pair for pairs in groups.values() for pair in pairs}:
        groups["train"].append((1, 2))
    return groups


def result_answer(result: int, style: str) -> str:
    return f"結果是{result}。" if style == "one_sentence" else f"1. 計算器已完成計算。\n2. 結果是{result}。"


def tool_records() -> list[dict]:
    rows = []
    for split, pairs in numeric_families().items():
        for pair_index, (left, right) in enumerate(pairs):
            group = f"tools:operands:{left:02d}:{right:02d}"
            template_index = pair_index % len(TOOL_TEMPLATES[split])
            template_family = f"tools:{split}:expression-family-{template_index:02d}"
            for operation_index, operation in enumerate(OPERATIONS):
                for direction, (a, b) in enumerate(((left, right), (right, left))):
                    if left == right and direction:
                        continue
                    question = TOOL_TEMPLATES[split][template_index][operation_index].format(a=a, b=b)
                    call = serialize_tool_call(operation, a, b)
                    actual = execute_tool_call(call)
                    for style in FORMATS:
                        prompt = style_history(style, direction, tool=True) + [message("user", question)]
                        meta = {
                            "intent": "calculation",
                            "format": style,
                            "expected_call": json.loads(call),
                            "expected_result": actual["result"],
                            "template_family": template_family,
                            "operand_family": group,
                            "depends_on_history": True,
                        }
                        rows.append(record(group, split, "tool_call", prompt + [message("assistant", call)], **meta))
                        reply_prompt = prompt + [
                            message("assistant", call),
                            message("tool", serialize_tool_result(actual)),
                        ]
                        rows.append(
                            record(
                                group,
                                split,
                                "tool_reply",
                                reply_prompt + [message("assistant", result_answer(actual["result"], style))],
                                **meta,
                                replay_kind="actual_executor",
                            )
                        )
                        # Counterfactual values are never training examples or normal
                        # calculator correctness cases. They test reading tool role.
                        if split == "test":
                            replacement = actual["result"] + 1 if actual["result"] < 9801 else actual["result"] - 1
                            replaced = {"tool": "calculator", "ok": True, "result": replacement}
                            replay = prompt + [
                                message("assistant", call),
                                message("tool", serialize_tool_result(replaced)),
                            ]
                            altered_meta = {**meta, "expected_result": replacement}
                            rows.append(
                                record(
                                    group,
                                    split,
                                    "tool_reply",
                                    replay + [message("assistant", result_answer(replacement, style))],
                                    **altered_meta,
                                    replay_kind="counterfactual_test_only",
                                    actual_result=actual["result"],
                                )
                            )
                    if operation_index == 0 and direction == 0:
                        for style in FORMATS:
                            unavailable = style_history(style, tool=True, enabled=False) + [message("user", question)]
                            answer = (
                                "計算器目前不可用，請開啟後再計算。"
                                if style == "one_sentence"
                                else "1. 計算器目前不可用。\n2. 請開啟後再計算。"
                            )
                            rows.append(
                                record(
                                    group,
                                    split,
                                    "tool_unavailable",
                                    unavailable + [message("assistant", answer)],
                                    intent="calculation",
                                    format=style,
                                    expected_tool=False,
                                    template_family=template_family,
                                    operand_family=group,
                                )
                            )
    # Missing, non-calculation and outside-contract families are split before
    # expansion; all cases include a real two-turn style history.
    cases = {
        "tool_missing": (
            ("請把8加上另一個數。", "另一個數還沒提供，幫我加8。"),
            ("請計算一個數乘7。", "乘7之前我還沒提供另一個數。"),
            ("我想算12減一個數。", "12要減去的數還沒告訴你。"),
            ("請算總和，但我還沒提供兩個數。", "我還沒提供數字，想先請你算總和。"),
            ("加法只給了5，還差一個數。", "目前只有5，要作加法。"),
            ("乘法只有9，還差另一個參數。", "目前只知道9，要作乘法。"),
            ("要作減法，但少了第二個數。", "第二個數還沒給，請算減法。"),
            ("兩個數還没提供，請求計算。", "我想計算，數字稍後才給。"),
        ),
        "tool_concept": (
            ("請說明加法的意思，不用計算。", "加法代表什麼？先不要算數字。"),
            ("我想知道相加的意思。", "請解釋把數合起來的概念。"),
            ("請說明加法概念，不要舉數字計算。", "只說加法概念就好。"),
            ("不用算，請介紹加法。", "先介紹加法的概念。"),
            ("我問的是加法意思，沒有計算題。", "沒有數字題，只問加法意思。"),
            ("先教我什麼叫相加。", "教我把兩個數合起來的意思。"),
            ("這次不要算，只解釋相加。", "這次問相加概念，不用算。"),
            ("請用文字解釋加法。", "加法怎麼理解？只需文字說明。"),
        ),
        "tool_unsupported": (
            ("請算100加2。", "請計算101減1。"),
            ("請算1.5加2。", "請計算2.5乘3。"),
            ("請算8除2。", "請用除法算9除3。"),
            ("請計算負2加3。", "算負1乘4。"),
            ("請算200乘2。", "幫我算120加5。"),
            ("請求計算3.5減1。", "請算小數4.5加2。"),
            ("請計算12除3。", "這次想算10除2。"),
            ("請算負3乘2。", "幫我算負4加8。"),
        ),
    }
    short = {
        "tool_missing": "請提供完整的兩個整數與運算方式。",
        "tool_concept": "加法是把兩個數合起來。",
        "tool_unsupported": "計算器只支援0到99非負整數的加減乘，請改成範圍內的題目。",
    }
    points = {
        "tool_missing": "1. 計算參數還不完整。\n2. 請提供兩個整數與運算方式。",
        "tool_concept": "1. 加法是把兩個數合起來。\n2. 概念說明不需要計算器。",
        "tool_unsupported": "1. 這超出計算器範圍。\n2. 請提供0到99非負整數的加減乘題目。",
    }
    for task, families in cases.items():
        for index, questions in enumerate(families):
            split = split_for_family(index)
            group = f"tools:{task}:original-family-{index:02d}"
            for question_index, question in enumerate(questions):
                for style in FORMATS:
                    answer = short[task] if style == "one_sentence" else points[task]
                    rows.append(
                        record(
                            group,
                            split,
                            task,
                            style_history(style, question_index, tool=True)
                            + [message("user", question), message("assistant", answer)],
                            intent="calculation" if task != "tool_concept" else "scope",
                            format=style,
                            expected_tool=False,
                            template_family=group,
                            depends_on_history=True,
                        )
                    )
    return rows


def audit_records(rows: list[dict]) -> dict:
    seen_ids = set()
    group_splits = defaultdict(set)
    template_splits = defaultdict(set)
    operand_splits = defaultdict(set)
    prompt_splits = defaultdict(set)
    counts = Counter()
    text_families = defaultdict(set)
    text_counts = Counter()
    for row in rows:
        assert row["id"] not in seen_ids, row["id"]
        seen_ids.add(row["id"])
        assert row["messages"][-1]["role"] == "assistant"
        assert all(set(item) == {"role", "content"} for item in row["messages"])
        assert sum(item["role"] == "user" for item in row["messages"]) >= 2
        group_splits[row["group_id"]].add(row["split"])
        template_splits[row["supervision"]["template_family"]].add(row["split"])
        if "operand_family" in row["supervision"]:
            operand_splits[row["supervision"]["operand_family"]].add(row["split"])
        prompt_splits[digest(row["messages"][:-1])].add(row["split"])
        counts[(row["split"], row["task"])] += 1
        if row["task"] == "text":
            text_families[(row["split"], row["supervision"]["intent"])].add(row["supervision"]["template_family"])
            text_counts[(row["split"], row["supervision"]["intent"])] += 1
        if row["supervision"].get("replay_kind") == "counterfactual_test_only":
            assert row["split"] == "test"
    for registry in (group_splits, template_splits, operand_splits, prompt_splits):
        assert all(len(splits) == 1 for splits in registry.values()), "cross-split family or prompt leak"
    known = {
        character
        for row in rows
        if row["split"] == "train"
        for item in row["messages"]
        for character in item["content"]
    }
    combination_cases = [
        row for row in rows if row["supervision"].get("generalization") == "known-words-new-option-combination"
    ]
    unknown_combination_chars = {
        character
        for row in combination_cases
        for item in row["messages"][:-1]
        for character in item["content"]
        if character not in known
    }
    assert not unknown_combination_chars, unknown_combination_chars
    return {
        "record_count": len(rows),
        "unique_groups": len(group_splits),
        "unique_template_families": len(template_splits),
        "unique_operand_families": len(operand_splits),
        "exact_prompt_cross_split_collisions": 0,
        "counts_by_split_and_task": {
            split: {task: count for (part, task), count in sorted(counts.items()) if part == split}
            for split in ("train", "validation", "test")
        },
        "groups_by_split": {
            split: sum(splits == {split} for splits in group_splits.values())
            for split in ("train", "validation", "test")
        },
        "text_intents": sorted(set(row["supervision"]["intent"] for row in rows)),
        "text_families_by_split_and_intent": {
            split: {intent: len(groups) for (part, intent), groups in sorted(text_families.items()) if part == split}
            for split in ("train", "validation", "test")
        },
        "text_records_by_split_and_intent": {
            split: {intent: count for (part, intent), count in sorted(text_counts.items()) if part == split}
            for split in ("train", "validation", "test")
        },
        "known_word_combinations": {
            "test_records": len(combination_cases),
            "unknown_input_characters": sorted(unknown_combination_chars),
            "heldout": "包/短靴 ordered choice-update combinations; train combines each only with 褲子",
            "verification": "every input character occurs in train; finite familiar-option recombination, not free language",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "records": len(rows),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def audit_voice_overlap(rows: list[dict], audit_path: Path) -> tuple[list[dict], dict]:
    """Exclude an entire text family if a source test query matches lexically.

    Source transcripts are used for exclusion only, never for authoring, tokens,
    answer supervision, or inference. This bounded script-equivalence map makes
    the comparison conservative for traditional/simplified Chinese phrases in
    these finite tasks; it is not an ASR correction or full Chinese converter.
    """
    if not audit_path.exists():
        return rows, {"status": "pending", "reason": "voice source audit not yet available"}
    equivalences = str.maketrans(
        dict(
            zip(
                "請問麼該處裡檢點網啟錯誤開機數帳戶無變換資樣個幫張後關敗種說這還給聲來時電應認試壞辦碼錄讓復讀實區據語對話題選記櫃靴褲詢寫預測氣長體診斷動謝隻從與過將當發顯稱別沒",
                "请问么该处里检点网启错误开机数账户无变换资样个帮张后关败种说这还给声来时电应认试坏办码录让复读实区据语对话题选记柜靴裤询写预测气长体诊断动谢只从与过将当发显称别没",
                strict=True,
            )
        )
    )

    def normalized(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).translate(equivalences).lower()
        text = "".join(character for character in text if character.isalnum())
        return re.sub(r"^(?:喂你好|嗨你好|您好|你好|请问)|(?:谢谢你|谢谢)$", "", text)

    source = [json.loads(line) for line in audit_path.read_text().splitlines()]
    heldout = [
        (item["source_utterance_id"], normalized(item["transcription"]))
        for item in source
        if item.get("selected") and item.get("prepared_split") == "test"
    ]
    query_groups = defaultdict(set)
    for row in rows:
        if row["split"] in ("train", "validation"):
            for item in row["messages"][:-1]:
                if item["role"] == "user":
                    query_groups[normalized(item["content"])].add(row["group_id"])
    excluded_groups = set()
    collisions = []
    maximum_similarity = 0.0
    for text, groups in query_groups.items():
        for utterance_id, voice_text in heldout:
            similarity = SequenceMatcher(None, text, voice_text, autojunk=False).ratio()
            maximum_similarity = max(maximum_similarity, similarity)
            if text == voice_text or similarity >= 0.90:
                excluded_groups.update(groups)
                collisions.append(
                    {"source_utterance_id": utterance_id, "text_group_ids": sorted(groups), "similarity": similarity}
                )
    filtered = [row for row in rows if row["group_id"] not in excluded_groups]
    return filtered, {
        "status": "completed_lexical_exclusion",
        "voice_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "voice_test_utterances": len(heldout),
        "train_validation_unique_user_queries": len(query_groups),
        "normalization": "NFKC, finite traditional/simplified map, alphanumeric lower, remove greeting/thanks",
        "near_duplicate_threshold": 0.90,
        "maximum_similarity_before_exclusion": maximum_similarity,
        "excluded_groups": sorted(excluded_groups),
        "excluded_records": len(rows) - len(filtered),
        "collisions": collisions,
        "limitations": "lexical exclusion only; no native-speaker semantic-duplicate review or manually checked source transcript claim",
        "source_transcripts_used_for": "exclusion audit only; never tokenizer, labels, or model inputs",
    }


def validate_calculator_runtime() -> dict:
    """Small actual CPU protocol tests, separate from any neural accuracy claim."""
    accepted = [
        ("add", 0, 0, 0),
        ("add", 99, 99, 198),
        ("subtract", 0, 99, -99),
        ("subtract", 99, 0, 99),
        ("multiply", 99, 99, 9801),
        ("multiply", 0, 99, 0),
        ("add", 1, 2, 3),
    ]
    for operation, a, b, expected in accepted:
        call = serialize_tool_call(operation, a, b)
        assert parse_tool_call(call) == CalculatorCall(operation, a, b)
        assert execute_tool_call(call)["result"] == expected
    invalid = [
        "",
        "{}",
        "[]",
        "null",
        "3",
        "not JSON",
        '{"tool":"calculator","operation":"add","a":1}',
        '{"tool":"shell","operation":"add","a":1,"b":2}',
        '{"tool":"calculator","operation":"divide","a":1,"b":2}',
        '{"tool":"calculator","operation":"add","a":true,"b":2}',
        '{"tool":"calculator","operation":"add","a":1.0,"b":2}',
        '{"tool":"calculator","operation":"add","a":"1","b":2}',
        '{"tool":"calculator","operation":"add","a":-1,"b":2}',
        '{"tool":"calculator","operation":"add","a":100,"b":2}',
        '{"tool":"calculator","operation":"add","a":1,"b":2,"code":"print(1)"}',
        '{"tool":"calculator","operation":"add","a":1,"a":2,"b":2}',
        '{"tool":"calculator","operation":"add","a":NaN,"b":2}',
        '{"tool":"calculator","operation":"add","a":null,"b":2}',
        '{"tool":"calculator","operation":"add","a":[1],"b":2}',
        '```json\n{"tool":"calculator","operation":"add","a":1,"b":2}\n```',
        '{"tool":"calculator","operation":"add","a":1,"b":2} trailing',
        " " * 257,
    ]
    for bad in invalid:
        try:
            parse_tool_call(bad)
        except ToolCallError:
            pass
        else:
            raise AssertionError(f"parser accepted an invalid input: {bad!r}")
    prompt = [message("user", "請計算1加2。")]
    calls = []

    def simulated_generator(history):
        calls.append(history)
        if history[-1]["role"] == "tool":
            returned = json.loads(history[-1]["content"])
            return str(returned["result"]) if returned["ok"] else returned["error"]
        return serialize_tool_call("add", 1, 2)

    trace = run_tool_loop(simulated_generator, prompt)
    assert trace["executed"] and trace["final_output"] == "3" and len(calls) == 2
    assert [item["role"] for item in calls[-1]] == ["user", "assistant", "tool"]
    modal_prompt = [
        {
            "role": "user",
            "content": "請計算1加2。",
            "image": "images/neutral.png",
            "audio": "audio/neutral.wav",
            "roi": [0, 0, 16, 16],
            "image_layout": "two_cells",
            "supervision": {"expected_result": 777},
            "unknown_private_field": "not-input",
        }
    ]
    trace = run_tool_loop(simulated_generator, modal_prompt)
    expected_public = {
        key: value for key, value in modal_prompt[0].items() if key not in ("supervision", "unknown_private_field")
    }
    assert calls[-2][0] == calls[-1][0] == expected_public
    assert trace["executed"] and trace["final_output"] == "3"
    trace = run_tool_loop(simulated_generator, prompt, tools_enabled=False)
    assert not trace["executed"] and trace["status"] == "unavailable" and trace["final_output"] == "unavailable"
    trace = run_tool_loop(lambda history: "直接回答" if history[-1]["role"] != "tool" else "錯誤", prompt)
    assert trace["status"] == "no_tool" and trace["final_output"] == "直接回答"
    trace = run_tool_loop(lambda history: "{}" if history[-1]["role"] != "tool" else "缺參數", prompt)
    assert trace["status"] == "invalid_call" and trace["final_output"] == "缺參數" and not trace["executed"]
    # Valid dataclass construction must not bypass executor bounds.
    try:
        execute_tool_call(CalculatorCall("add", True, 1))
    except ToolCallError:
        pass
    else:
        raise AssertionError("dataclass bypassed operand validation")
    return {
        "passed": True,
        "accepted_boundary_cases": len(accepted),
        "rejected_schema_cases": len(invalid),
        "loop_cases": 5,
        "public_modal_metadata_cases": 1,
        "dataclass_bypass_cases": 1,
        "device": "CPU",
        "model_generation": "simulated callback; no neural accuracy claim",
    }


def prepare(output_dir: Path, voice_audit_path: Path | None = None) -> dict:
    runtime_validation = validate_calculator_runtime()
    rows = text_records() + tool_records()
    rows, cross_modal_audit = audit_voice_overlap(rows, voice_audit_path or output_dir / "voice-audit.jsonl")
    audit = audit_records(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        files[split] = write_jsonl(output_dir / f"text-tools-{split}.jsonl", selected)
    manifest = {
        "version": VERSION,
        "author": AUTHOR,
        "license": "MIT (original repository material)",
        "source": {
            "kind": "project-original",
            "path": "scripts/selftrained/prepare_text_tools.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "external_teacher": None,
            "human_review": "pending",
            "reviewer": None,
        },
        "scope": {
            "intents": 10,
            "chat": "finite authored Chinese intent/format/history conversations",
            "calculator": {
                "operations": list(OPERATIONS),
                "operand_min": 0,
                "operand_max": 99,
                "result_min": -99,
                "result_max": 9801,
                "hops": 1,
            },
        },
        "record_contract": {
            "input": "messages[:-1], role/content only; final assistant is target",
            "supervision": "loss/evaluation only; never model input",
            "provenance": "audit only; never model input",
        },
        "split_contract": {
            "text": "authored paraphrase family and all complete conversation-tree descendants",
            "tools": "canonical unordered operands assigned before operations/directions/availability/replies",
            "tool_templates": "expression families exclusive to each split",
            "heldout_limits": "text has ten authored test base-prompt families per intent and five validation families; family counts do not establish learned generalization",
            "cross_modal": "no source transcript authored into text data; original train/validation queries lexically checked against voice test audit",
        },
        "counterfactual": "test-only tool_reply replay; exclude from normal arithmetic correctness and all training",
        "review": {
            "status": "pending",
            "authoring": "original constrained templates and reference replies authored in script",
            "checks_completed": [
                "strict record structure",
                "ids unique",
                "group/template/operand split isolation",
                "exact prompt collision check",
                "real calculator result labels",
            ],
            "checks_not_claimed": [
                "root human review",
                "native speaker review",
                "bank-specific correctness",
                "unseen speaker validation",
                "trained-model accuracy",
            ],
        },
        "runtime_validation": runtime_validation,
        "cross_modal_audit": cross_modal_audit,
        "tools_source": {
            "path": "tiny_perceptron/selftrained/tools.py",
            "sha256": hashlib.sha256((ROOT / "tiny_perceptron/selftrained/tools.py").read_bytes()).hexdigest(),
        },
        "audit": audit,
        "files": files,
    }
    manifest_dir = output_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "text-tools.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/selftrained/data")
    parser.add_argument(
        "--voice-audit", type=Path, default=None, help="exclusion-only source voice audit; never training input"
    )
    args = parser.parse_args()
    manifest = prepare(args.output_dir.resolve(), args.voice_audit)
    print(
        json.dumps(
            {"version": manifest["version"], "audit": manifest["audit"], "files": manifest["files"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
