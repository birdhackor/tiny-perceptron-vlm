"""自行繪製、無外部圖像授權需求的SVG：每張具靜態標註與漸進提示。"""

import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "course/figures"
VISUALS = {}


def escape(value):
    return html.escape(str(value))


def label(x, y, text, size=18, color="#172b4d", anchor="middle"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(text)}</text>'


def box(x, y, w, h, title, subtitle="", color="#e8f1ff"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#8da8ca"/>'
        + label(x + w / 2, y + h / 2 - 4, title, 19)
        + label(x + w / 2, y + h / 2 + 23, subtitle, 14)
    )


def save(name, title, caption, body, height=330, animated=True):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    animated = animated and 'class="pulse' in body
    animation = (
        """<style>
    text{font-family:"Noto Sans CJK TC","Microsoft JhengHei","PingFang TC",sans-serif}
    rect.pulse,.pulse rect{animation:pulse 6s infinite}.step1,.step1 rect{animation-delay:0s}.step2,.step2 rect{animation-delay:1.5s}.step3,.step3 rect{animation-delay:3s}.step4,.step4 rect{animation-delay:4.5s}
    @keyframes pulse{0%,23%{stroke:#e87924;stroke-width:4}24%,100%{stroke:#8da8ca;stroke-width:1}}
    @media(prefers-reduced-motion:reduce){rect.pulse,.pulse rect{animation:none}}
    </style>"""
        if animated
        else '<style>text{font-family:"Noto Sans CJK TC","Microsoft JhengHei","PingFang TC",sans-serif}</style>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 {height}" role="img" aria-labelledby="title desc">
    <title id="title">{escape(title)}</title><desc id="desc">{escape(caption)}。{"橘色輪流提示閱讀步驟，" if animated else ""}所有內容靜態可見。</desc>
    {animation}<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#466b97"/></marker></defs>
    <rect width="800" height="{height}" fill="#fff"/>{label(400, 32, title, 24)}
    {body}{label(400, height - 16, caption, 15)}
    </svg>'''
    (DIRECTORY / f"{name}.svg").write_text(svg, encoding="utf-8")
    VISUALS[name] = {"title": title, "caption": caption, "animated": animated}


def flow(name, title, caption, nodes):
    body = ""
    width = 140
    y = 115
    gap = 17
    for i, (primary, secondary) in enumerate(nodes):
        x = 17 + i * (width + gap)
        body += f'<g class="pulse step{i % 4 + 1}">{box(x, y, width, 80, primary, secondary)}</g>'
        if i < len(nodes) - 1:
            body += f'<path d="M{x + width} {y + 40}L{x + width + gap - 3} {y + 40}" stroke="#466b97" stroke-width="2" marker-end="url(#arrow)"/>'
    save(name, title, caption, body)


def token_rows(name, title, caption, rows):
    body = ""
    for row, (row_label, values) in enumerate(rows):
        y = 72 + row * 62
        body += label(12, y + 32, row_label, 16, anchor="start")
        for column, value in enumerate(values):
            x = 125 + column * 88
            color = "#d7f5e6" if value not in ("—", "PAD", "×") else "#edf0f4"
            body += f'<g class="pulse step{column % 4 + 1}">{box(x, y, 80, 48, value, "", color)}</g>'
    save(name, title, caption, body, height=110 + len(rows) * 62)


def matrix(name, title, caption, allowed):
    body = label(400, 69, "欄：可以讀取的過去／目前位置", 17)
    for i in range(4):
        body += label(235, 110 + i * 42, f"查詢 {i}", 16)
        for j in range(4):
            ok = allowed(i, j)
            x = 300 + j * 78
            y = 82 + i * 42
            body += f'<rect class="pulse step{i + 1}" x="{x}" y="{y}" width="68" height="34" fill="{"#d7f5e6" if ok else "#edf0f4"}" stroke="#8da8ca"/>'
            body += label(x + 34, y + 23, "可看" if ok else "禁止", 15)
    save(name, title, caption, body)


def build():
    flow(
        "lookup",
        "像查字典：ID 找到一列向量",
        "ID 只是索引；向量的內容由訓練改變。",
        [
            ("字元「貓」", "原始文字"),
            ("ID = 2", "不是數量2"),
            ("表的第2列", "查表"),
            ("3個數字", "feature 向量"),
            ("後續模型", "混合與預測"),
        ],
    )
    token_rows(
        "shift",
        "下一字對齊：問題與答案錯開一格",
        "同一格 logits 預測下一個字；shift 只做一次。",
        [("原文", ["貓", "看", "狗", "。"]), ("輸入 X", ["貓", "看", "狗"]), ("目標 Y", ["看", "狗", "。"])],
    )
    flow(
        "split",
        "先分整份文件，再切小窗口",
        "近似改寫的同一家族也放在同一側。",
        [
            ("原始文件", "先標來源家族"),
            ("去重／分組", "不是切碎再分"),
            ("train", "用來更新"),
            ("validation", "用來選設定"),
            ("test", "最後評估"),
        ],
    )
    flow(
        "softmax",
        "把分數換成比例：像分一塊餅",
        "一起加同常數不改比例，分數差才重要。",
        [
            ("分數", "[1,2,−1]"),
            ("減最大值", "[−1,0,−3]"),
            ("取 exp", "正數"),
            ("除以總和", "總和=1"),
            ("機率", "[.260,.705,.035]"),
        ],
    )
    body = ""
    for i, (x, text) in enumerate(((130, "w=1，loss=4"), (360, "gradient=−4"), (590, "w=1.4，loss=2.56"))):
        body += f'<g class="pulse step{i + 1}">{box(x - 90, 115, 180, 95, text, "η=.1；沿反方向")}</g>'
        if i < 2:
            body += f'<path d="M{x + 90} 163L{x + 138} 163" stroke="#466b97" marker-end="url(#arrow)"/>'
    save("gradient", "像調旋鈕：沿梯度反方向走一小步", "這是 (w−3)² 的一步驗算，沒有模型訓練。", body)
    # 同一張圖加入可核對的曲線；不是隻有文字流程框。
    points = " ".join(f"{130 + w * 95:.1f},{285 - (w - 3) ** 2 * 19:.1f}" for w in [i / 20 for i in range(121)])
    curve = '<path d="M100 70V300H730" fill="none" stroke="#8da8ca" stroke-width="2"/>'
    curve += f'<polyline points="{points}" fill="none" stroke="#2571b5" stroke-width="3"/>'
    curve += label(68, 80, "代價", 17) + label(740, 304, "w", 17)
    for w in (0, 1, 2, 3, 4, 5, 6):
        curve += label(130 + w * 95, 324, str(w), 15)
    for w, color, text in ((1, "#e87924", "起點 w=1"), (1.4, "#167947", "一步後 w=1.4"), (3, "#2571b5", "最低點 w=3")):
        x, y = 130 + w * 95, 285 - (w - 3) ** 2 * 19
        curve += f'<circle cx="{x}" cy="{y}" r="7" fill="{color}"/>'
        curve += label(x - 15, y - 14, text, 16, color, anchor="end" if w < 1.3 else "start")
    curve += '<path d="M225 209L263 236" stroke="#e87924" stroke-width="3" marker-end="url(#arrow)"/>'
    curve += box(420, 82, 280, 80, "敏感度 = −4", "沿反方向：w ← w − .1×(−4)")
    save("gradient", "旋鈕往哪邊轉，猜錯的代價會下降？", "藍線是 (w−3)²；橘點到綠點只走一小步。", curve, height=375)
    flow(
        "backward",
        "一次更新的生命週期",
        "忘記清零會累加；累積梯度時才刻意保留。",
        [
            ("zero_grad", "清除舊梯度"),
            ("forward", "得到分數"),
            ("loss", "猜錯的代價"),
            ("backward", "算敏感度"),
            ("step", "改參數"),
        ],
    )
    flow(
        "context",
        "上下文像把幾張卡片一起攤開",
        "拼接順序保留位置；窗口固定長度。",
        [
            ("前3個ID", "[1,2,3]"),
            ("各查向量", "3張卡片"),
            ("按順序拼接", "[C×D]"),
            ("混合特徵", "Linear+彎曲"),
            ("猜下一字", "詞表分數"),
        ],
    )
    flow(
        "qkv",
        "問路譬喻：想找什麼、誰知道、拿回什麼",
        "Q/K 決定讀哪裡；V 是讀回的內容。",
        [
            ("Q：需求", "我要找哪種線索"),
            ("K：索引", "每張卡能提供什麼"),
            ("點積→比例", "相關性權重"),
            ("V：內容", "真正取回資料"),
            ("加權混合", "一個輸出向量"),
        ],
    )
    matrix("causal", "不能偷看明天的答案", "對角線可看；右上角未來位置禁止。", lambda i, j: j <= i)
    matrix(
        "packing",
        "兩份文件隔開：EOS 不等於牆",
        "例中0–1為文件A，2–3為文件B。",
        lambda i, j: j <= i and (i < 2) == (j < 2),
    )
    flow(
        "residual",
        "像原稿旁加修訂，而不是撕掉原稿",
        "y=x+f(x)，原資料和修訂都有路徑。",
        [
            ("原稿 x", "保留主路徑"),
            ("正規化", "先控制尺度"),
            ("新修訂 f(x)", "attention / FFN"),
            ("加回 x", "residual"),
            ("新版表示", "同樣shape"),
        ],
    )
    flow(
        "normalization",
        "每張卡片自己調尺度",
        "沿 feature 軸，不能跨時間偷看別的卡片。",
        [
            ("一個token", "[1,2,3]"),
            ("平均／尺度", "只沿feature"),
            ("LN或RMS", "兩種不同規則"),
            ("可學縮放", "gamma / beta"),
            ("同shape輸出", "下一個零件"),
        ],
    )
    flow(
        "checkpoint",
        "存檔像暫停遊戲：不只存角色外觀",
        "想接續 Adam，還要動量、步數與隨機狀態。",
        [
            ("模型權重", "能推論"),
            ("optimizer", "動量狀態"),
            ("步數／RNG", "接續抽樣"),
            ("tokenizer配置", "相同ID意思"),
            ("恢復後核對", "下一步一致"),
        ],
    )
    token_rows(
        "bpe",
        "常見卡片合併：不是先懂詞義",
        "每次合併一對常見相鄰符號。",
        [("原符號", ["a", "b", "a", "b"]), ("合併 ab", ["ab", "ab"]), ("壓縮後", ["兩個token"])],
    )
    token_rows(
        "masks",
        "兩種遮罩：看得見，不一定直接考它",
        "X 的 assistant 格預測 A；PAD 不當答案。",
        [
            ("X", ["BOS", "user", "Q", "EOS", "assist", "A"]),
            ("Y 監督", ["—", "—", "—", "—", "A", "EOS"]),
            ("輸入可見", ["有", "有", "有", "有", "有", "有"]),
        ],
    )
    token_rows(
        "padding",
        "補齊長度像空椅子：不是新客人",
        "attention 別讀 PAD，loss 也別考 PAD。",
        [
            ("短例 X", ["Q", "A", "EOS", "PAD", "PAD"]),
            ("有效key", ["有", "有", "有", "×", "×"]),
            ("有效答案", ["—", "A", "EOS", "—", "—"]),
        ],
    )
    flow(
        "style",
        "有個性也要可靠：三個分開的面向",
        "漂亮文筆不能替代正確答案或格式遵循。",
        [
            ("同一題目", "先固定內容"),
            ("風格條件", "短答／貼切比喻"),
            ("內容正確", "真值核對"),
            ("遵循限制", "格式／長度"),
            ("新題驗證", "不是背範例"),
        ],
    )
    flow(
        "lora",
        "原畫不重畫：加一張低rank修正圖層",
        "基模仍佔空間；只是微調參數變少。",
        [
            ("凍結 W", "原來Linear"),
            ("小 A", "in→rank"),
            ("小 B", "rank→out"),
            ("alpha/r 縮放", "形成 ΔW"),
            ("W+ΔW", "合併或旁路"),
        ],
    )
    flow(
        "honesty",
        "盒中球數：看得到纔有根據",
        "隱藏真值不能當成可見資料；缺資訊要指出缺什麼。",
        [
            ("問題", "盒裡幾顆球"),
            ("可見資料", "有數量？"),
            ("有：作答", "數量可核對"),
            ("缺：澄清", "不要猜隱藏真值"),
            ("新情境測試", "含錯誤前提"),
        ],
    )
    flow(
        "safety",
        "像分清「借自己的鑰匙」和「拿別人的鑰匙」",
        "主題相同也可有不同權限，不能只靠關鍵字拒絕。",
        [
            ("具體情境", "誰的祕密碼"),
            ("權限資訊", "允許／不允許"),
            ("相關處理", "完成或說明邊界"),
            ("可行替代", "針對原需求"),
            ("分項評估", "含無害完成率"),
        ],
    )
    flow(
        "calibration",
        "信心像音量：說得大聲不保證說對",
        "每箱保留樣本量；文字自述信心是另一回事。",
        [
            ("預測分數", "softmax 最大值"),
            ("獨立真值", "答對／答錯"),
            ("按信心分箱", "例如.8到1"),
            ("平均信心", "和正確率相比"),
            ("曲線判讀", "不要只報高信心"),
        ],
    )
    body = ""
    for i in range(4):
        for j in range(4):
            x, y = 95 + j * 45, 90 + i * 45
            body += f'<rect class="pulse step{(i + j) % 4 + 1}" x="{x}" y="{y}" width="42" height="42" fill="{"#b7deff" if (i + j) % 2 else "#d7f5e6"}" stroke="#fff"/>'
            body += label(x + 21, y + 27, i * 4 + j, 15)
    body += label(490, 140, "→ 依列展開成16張小卡", 20) + label(495, 185, "每張：3×4×4 = 48 個像素值", 17)
    save("patchify", "圖片切小方塊：先檢查能不能拼回來", "16×16 RGB，patch邊長4，得到16個patch。", body, height=330)
    token_rows(
        "modal_expand",
        "一個圖片佔位格，展開成三個向量",
        "展開 labels 同步加 ignored，再做下一字 shift。",
        [
            ("展開前", ["BOS", "image", "assist", "A", "EOS"]),
            ("展開後", ["BOS", "v0", "v1", "v2", "assist", "A", "EOS"]),
            ("監督目標", ["—", "—", "—", "—", "A", "EOS"]),
        ],
    )
    flow(
        "alignment",
        "轉接頭尺寸合了，還要學資料的意思",
        "先確認文字基模會用文字屬性回答，再學圖片接頭。",
        [
            ("圖像像素", "真正的圖片"),
            ("encoder", "可用視覺特徵"),
            ("projector", "Dv→D"),
            ("文字基模", "已有內容能力"),
            ("答案檢查", "正確+換圖敏感"),
        ],
    )
    flow(
        "contrastive",
        "像把名字卡與照片配對",
        "對角線是正配對；重複描述可能造成假負例。",
        [
            ("一批圖片", "圖encoder"),
            ("一批描述", "文字encoder"),
            ("cosine表", "每圖對每描述"),
            ("雙向CE", "image↔text"),
            ("保留組合", "檢索驗證"),
        ],
    )
    flow(
        "audio",
        "從聽到的起伏，找到每一小段的頻率",
        "frame→window→STFT→mel→log，每次只加一個步驟。",
        [
            ("waveform", "時間與振幅"),
            ("框+窗", "邊緣柔和"),
            ("STFT", "每框有哪些頻率"),
            ("mel加總", "低頻較密"),
            ("log", "壓縮能量跨度"),
        ],
    )
    body = ""
    points = " ".join(f"{65 + i},{160 - 45 * math.sin(2 * math.pi * i / 55)}" for i in range(640))
    body += f'<polyline points="{points}" fill="none" stroke="#3675b8" stroke-width="2"/>'
    for i in range(4):
        body += f'<rect class="pulse step{i + 1}" x="{80 + i * 130}" y="85" width="180" height="150" fill="none" stroke="#8da8ca"/>'
    save("frames", "像用放大鏡逐段看聲音：時間框可重疊", "框長決定分析範圍；hop 決定多久移動一次。", body)
    flow(
        "joint",
        "形狀在圖片，音高在聲音：兩把鑰匙一起用",
        "替換每個模態，答案只改對應部分；保留新組合。",
        [
            ("圖片線索", "circle／square"),
            ("聲音線索", "low／high"),
            ("兩個projector", "共同文字維度"),
            ("聯合回答", "shape,pitch"),
            ("逐項替換", "兩種都得有用"),
        ],
    )
    flow(
        "dpo",
        "試喫兩道菜：比起原廚師，更偏向好回答",
        "chosen 與 rejected 必須對同一個問題。",
        [
            ("同一prompt", "兩個候選"),
            ("policy分數", "答案logp加總"),
            ("reference分數", "固定SFT基準"),
            ("相對margin", "不是隻看長度"),
            ("DPO梯度", "改善相對偏好"),
        ],
    )
    body = '<circle cx="250" cy="165" r="90" fill="#edf5ff" stroke="#8da8ca"/>'
    for angle, color, text in ((0, "#3b82f6", "Q"), (45, "#d97706", "K")):
        x, y = 250 + 75 * math.cos(math.radians(angle)), 165 - 75 * math.sin(math.radians(angle))
        body += f'<line x1="250" y1="165" x2="{x}" y2="{y}" stroke="{color}" stroke-width="4"/>' + label(
            x + 12, y - 8, text, 18, color
        )
    body += (
        label(540, 130, "兩者一起旋轉：長度不變", 20)
        + label(540, 170, "相對角度仍45°，點積不變", 18)
        + label(540, 210, "位置像鐘面角度", 18)
    )
    save("rope", "成對feature像鐘面指針：用相對旋轉記位置", "單一pair的旋轉示意；不同pair有不同角速度。", body)
    body = ""
    # 一張訂單只走兩條實線；虛線表示這次沒有派工。
    for i in range(4):
        y = 65 + i * 74
        selected = i in (0, 2)
        color = "#d7f5e6" if selected else "#edf0f4"
        dash = "" if selected else 'stroke-dasharray="4 5"'
        body += f'<path d="M345 190L430 {y + 30}" stroke="#167947" fill="none" {dash} marker-end="url(#arrow)"/>'
        if selected:
            body += f'<path d="M575 {y + 30}L655 190" stroke="#167947" fill="none" marker-end="url(#arrow)"/>'
        subtitle = "本次不運算" if not selected else ("混合占60%" if i == 0 else "混合占40%")
        body += f'<g class="pulse step{i % 3 + 2}">' if selected else "<g>"
        body += box(430, y, 145, 60, "師傅 " + "ABCD"[i], subtitle, color) + "</g>"
    body += f'<g class="pulse step1">{box(15, 150, 135, 80, "一張字卡", "待加工的向量")}</g>'
    body += '<path d="M150 190L205 190" stroke="#466b97" marker-end="url(#arrow)"/>'
    body += f'<g class="pulse step2">{box(205, 150, 140, 80, "分單員", "選兩位師傅")}</g>'
    body += box(655, 150, 130, 80, "混合結果", "依比例相加")
    save(
        "moe", "四位師傅都在，這張訂單只請兩位動手", "例中只派給A與C；未選中師傅的權重仍佔儲存空間。", body, height=390
    )
    flow(
        "dispatch",
        "送出時分組，收回時依原訂單編號",
        "同token多份貢獻要相加，不能覆蓋。",
        [
            ("token 0/1/2", "原始順序"),
            ("索引分派", "expert各收子集"),
            ("expert計算", "可變小batch"),
            ("index_add", "依原row加回"),
            ("0/1/2輸出", "順序恢復"),
        ],
    )
    token_rows(
        "cache",
        "舊筆記不用重寫：只加新字的K/V",
        "cache保存每層筆記；不是提前知道下一字答案。",
        [
            ("舊cache", ["K0/V0", "K1/V1", "K2/V2"]),
            ("新token", ["只算3"]),
            ("append後", ["K0/V0", "K1/V1", "K2/V2", "K3/V3"]),
        ],
    )
    flow(
        "online_softmax",
        "邊切邊裝盤：不存整張注意力中間表",
        "數學結果等價；Python示例不是Flash GPU kernel。",
        [
            ("讀一小塊K/V", "固定block"),
            ("更新最大值m", "避免exp爆掉"),
            ("重縮舊分子母", "修正比例"),
            ("加新貢獻", "running sum"),
            ("最後相除", "完整加權平均"),
        ],
    )
    flow(
        "quantization",
        "換較粗刻度尺：省格子但有誤差",
        "縮小儲存與硬體加速分開驗證。",
        [
            ("浮點weight", "連續數值"),
            ("選scale", "一格多寬"),
            ("round+clamp", "整數刻度"),
            ("packing", "真正縮buffer"),
            ("反量化參考", "誤差/任務檢查"),
        ],
    )
    token_rows(
        "int4",
        "兩個小格裝一個byte：低四位與高四位",
        "本例signed加8：−3→0101，+4→1100，合成11000101。",
        [
            ("signed值", ["−3", "+4"]),
            ("加8編碼", ["0101", "1100"]),
            ("高位|低位", ["11000101"]),
            ("還原", ["−3", "+4"]),
        ],
    )
    flow(
        "distillation",
        "老師不只說第一名，也指出接近的候選",
        "學生選較小架構；老師可能教錯，仍要真值與行爲評估。",
        [
            ("同一prefix", "同一tokenizer"),
            ("teacher分佈", "凍結不更新"),
            ("student分佈", "較小架構"),
            ("CE+KL", "有效答案位置"),
            ("獨立檢查", "不只教師一致率"),
        ],
    )
    flow(
        "rag",
        "像先找小抄，再核對答案來源",
        "找到對文件與用對資料，是兩個不同分數。",
        [
            ("本地文件", "有source ID"),
            ("檢索", "詞匹配基準"),
            ("放進prompt", "外部新資料"),
            ("模型回答", "可核對引用"),
            ("證據查證", "不足就說明"),
        ],
    )
    flow(
        "tools",
        "工單先驗證，再交給真正工具做",
        "模型文字不等於已執行，result來自白名單純函式。",
        [
            ("模型請求", "name+arguments"),
            ("parse+validate", "名稱/欄位/型別"),
            ("執行純函式", "add/multiply"),
            ("回填實際結果", "不是預期答案"),
            ("done/上限", "明確停止"),
        ],
    )
    flow(
        "reasoning",
        "多試幾次，還要知道怎麼選",
        "候選包含正解，不代表最後答案已經選中。",
        [
            ("多個候選", "[3,3,4]"),
            ("coverage", "包含正解4"),
            ("多數決", "可能選3"),
            ("程式verifier", "窄任務可驗證"),
            ("成本表", "tokens+時間+正確"),
        ],
    )
    import json

    (DIRECTORY / "index.json").write_text(json.dumps(VISUALS, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(VISUALS)} 張自行繪製SVG")


if __name__ == "__main__":
    build()
