"""Package this reader's explicit bounded judgments, preserving every original byte."""
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
TASK = "/root/c_v3_reader_bridges"
IDS = ["9.18", "19.13", "19.16", "19.17", "19.19"]


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return ref(path)


freeze_path = ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v21.json"
assert sha(freeze_path) == "d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea"
freeze = json.loads(freeze_path.read_bytes())
old_freeze = json.loads((ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v19.json").read_bytes())
original_path = BASE / "reader-original.json"
original = json.loads(original_path.read_bytes())
records = json.loads((BASE / "reader-records.json").read_bytes())
old_estimates = json.loads((BASE / "reading-times-complete-v2.json").read_bytes())
preserved = {name: sha(BASE / name) for name in (
    "reader-original.json", "reader-original-intake.json", "reader-records.json",
    "reading-times-v1.json", "reading-times-complete-v2.json", "supplement-inspection-v2.json", "visual-v1.json",
)}
assert preserved["reader-original.json"] == "a981adcc32b00bfcc6da4bd21dddf80638b73035b13e87d475604c4216c4de78"
spec = importlib.util.spec_from_file_location("bridge_bounded_units", ROOT / "docs/course-revision-20261016/tools/reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
pages, texts, chunks, notes = {}, {}, {}, {}
for pid in IDS:
    page = next(row for row in freeze["pages"] if row["page_id"] == pid)
    previous = next(row for row in old_freeze["pages"] if row["page_id"] == pid)
    body = (ROOT / page["snapshot"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == page["source_sha256"]
    assert page["figures_sha256"] == previous["figures_sha256"] == {}
    pages[pid], texts[pid] = page, body.decode()
    chunks[pid] = reader.units(texts[pid])
    session = next(row for row in records["sessions"] if row["page_id"] == pid)
    assert len(chunks[pid]) == len(session["notes"])
    notes[pid] = []
    assert sha(ROOT / session["opening"]["path"]) == session["opening"]["sha256"]
    for artifact in session["notes"]:
        assert sha(ROOT / artifact["path"]) == artifact["sha256"]
        notes[pid].append(json.loads((ROOT / artifact["path"]).read_bytes()))

# These are reader judgments, written after directly seeing the new frozen hunks.
fixes = [
    {
        "id": "BR-19.17-01", "page_id": "19.17", "units": [4, 5],
        "quotes": [
            'print("指定 medium、硬上限 2 → 可跑預算", min(EFFORT["medium"], 2), "輪")',
            '上限 2 就只給 2 輪預算；函式庫實際推論遇到這種限制時會標成 `budget_limited`，不假裝跑滿。',
        ],
        "identity": "這一行只計算可給B的輪數預算，沒有執行B前向；實際推論狀態另外說明。",
        "mechanism": "medium要求3，min(3,2)=2；新標籤與相鄰『只給2輪預算』一致，只有真正推論才回budget_limited。",
        "need": "原建議要避免把純min打印當已跑B的執行證據；目前兩個位置都把預算示範和實際推論分開，正好補上該表述關係。",
        "usage": "可用此值預測受限預算，不能把示範計數當模型答案或CPU執行收據；requested仍3，actual預算2。",
        "example_scope": "數字與分支未變，不增加新算法或成本結論；只改善本來已可達的理解，原選讀改善分級保留。",
        "variation": "若cap改4，medium可跑預算是3；這個min程式仍不因此執行B。",
        "verdict": "pass", "status": "resolved", "new_required_issue": False,
    },
    {
        "id": "BR-19.17-02", "page_id": "19.17", "units": [7],
        "quotes": ['在本輪完整決策題上，這樣的平均輪數會超過3，不符合本次的決策目標。'],
        "identity": "平均所屬對象明指本輪完整決策題，不是短程式打印的1–6事件與加法各一例。",
        "mechanism": "界線4改2仍使3/4事件選5輪；平均超3是完整決策資料上的報告，沒有聲稱短程式本身計了該平均。",
        "need": "原建議要分清示例等權平均與決策資料平均；新增對象使同一限制的適用範圍在練習句可直接定位。",
        "usage": "可手推改分支；若要核整批平均仍須相應決策資料，不能拿含壓力例的打印清單替代。",
        "example_scope": "未增加題數或最優性宣稱；仍不以示範輸出證完整資料均值，原可選的表述缺口已處理。",
        "variation": "只改打印range時，不等於改完整決策題分布，不能因此重定該平均的分母。",
        "verdict": "pass", "status": "resolved", "new_required_issue": False,
    },
    {
        "id": "BR-19.19-01", "page_id": "19.19", "units": [11],
        "quotes": ['RNG（Random Number Generator，隨機數生成器）的狀態'],
        "identity": "RNG現在有完整英文名和中文，交付項明確是隨機數生成器的狀態，可對到前段隨機狀態。",
        "mechanism": "推論參數檔不含完整續訓狀態；新字句標出所需RNG狀態，沒有只列一個未解縮寫。",
        "need": "原可選建議只為讓高中文數背景讀者辨清這項續訓配套；新增名稱與狀態字樣已完成，不需再展隨機API。",
        "usage": "完整續訓包另需optimizer、RNG狀態、下一步位置、資料生成版本；仍不能只拿model.pt稱無縫續訓。",
        "example_scope": "這是名詞和配套角色補充，未提供也未宣稱執行過恢復狀態測試。",
        "variation": "檔案副檔名改了而仍沒RNG狀態，仍不能說完整續訓材料已齊備。",
        "verdict": "pass", "status": "resolved", "new_required_issue": False,
    },
]
for item in fixes:
    for quote in item["quotes"]:
        assert quote in texts[item["page_id"]]
    item["observed_at"] = now()
    item["source"] = ref(ROOT / pages[item["page_id"]]["snapshot"])
    item["original_issue"] = next(issue for page in original["pages"] for issue in page["issues"] if issue["id"] == item["id"])
    item["original_classification_preserved"] = True

observations_path = BASE / "bounded-recheck-v21-observations.json"
observations_ref = new(observations_path, {
    "schema_version": 1, "reviewer_task": TASK, "report_kind": "bounded_revised_source_observations",
    "created_at": now(), "source_freeze": ref(freeze_path), "original_report": ref(original_path),
    "scope": "直接讀source21的19.17/19.19三處修文及相鄰上下文；以原freeze19→21逐byte比較確認其餘段落不變。未讀作者receipt作答案，未開模型/final/外部技術答案。",
    "not_a_new_first_read": True, "post_route_exposure_disclosure": original["post_route_exposure_disclosure"],
    "fixes": fixes, "necessary_issues": [], "new_optional_issues": [],
})

# Explicit, per-concept dispositions. The evidence assembler below only binds hashes.
# Tuple: unit:concept, disposition, reason, later/current supporting units.
decisions = {
"9.18": [
("0:0","resolved","原u0先建立三份資料角色，原u2程式/u4數字已接上T、tau的分工；source21該三單位逐byte未變，數字待後段已完成。",[2,4]),
("0:1","nonblocking","接受只讀預測、真值只作事後計錯的原理解保留；三條門與三種失敗原句未變，沒有把技術執行未做升成文字缺口。",[0,2]),
("0:2","nonblocking","coverage/risk兩分母與零接受None的理解原已足夠；現在來源同一內容，不把未實跑改成實跑。",[0,4]),
("0:3","resolved","原本待數值反例，原u2/u4已展示X/Y跨題次序翻轉而同題argmax不變；這些仍是當前凍結正文。",[2,4]),
("1:0","nonblocking","NLL是正解負ln平均而非錯題率的角色、準則與用途已教；當前不變，仍未額外查外部校準證據。",[1,2]),
("2:0","resolved","當時手推尚待輸出，原u4的兩組X/Y值已核到；以後段實讀解原待展示，不倒寫早期紀錄。",[4]),
("2:1","nonblocking","以校準資料選T的代價機制可追，未執行程式的界線仍保留；本次只核文字來源不變。",[1,2,4]),
("2:2","nonblocking","take讀info_pred而errors讀gold的必要界線原已可解釋；當前未變。",[0,2]),
("2:3","resolved","原待最終數字在u4固定tau=.9及3/6零錯已讀到；其分母、可行性與不調final仍成立。",[4,5]),
("3:0","resolved","原待效果輸出由u4各組與全部final完成；合併題目不是平均risk的理解保留。",[4]),
("4:0","resolved","原u4尚未讀頁末，u5/u6後續已把固定規則失敗與不保新材料可靠收束；來源不變。",[5,6]),
("5:0","nonblocking","家族/來源/模板與固定final的分組範圍原已清楚，當前內容不變；未另查生成器。",[5]),
("5:1","nonblocking","區間基本角色已足夠；具體估算法與數值仍未驗證，當前頁未要求實算區間才可理解風險限制，故保留未驗證而非假稱補會。",[5,6]),
("5:2","nonblocking","z/T0乘T0還原的規則由本頁softmax可直接推得；API實際取得仍未測，不影響這項代數界線。",[2,5]),
("5:3","nonblocking","表內另一實測的分母與固定參數能閱讀，原紀錄/模型未核仍保留，不拿文本可讀代技術核實。",[5,6]),
("6:0","nonblocking","事後診斷曲線不能冒充固定部署規則的界線原已足夠；當前未改。",[6]),
("6:1","nonblocking","全頁固定規則、真值計分、人工/實測限制原已收束；來源/圖同19，保留文字pass，不稱重做首讀或執行。",[0,4,5,6]),
],
"19.13": [
("0:0","nonblocking","MoE有限客服用途與舊成績界線為導覽承諾，原已足夠；未另查訓練測試，不藉此宣稱全能。",[0]),
("0:1","nonblocking","B獨立、共享區塊多輪、有限答案類別的導覽角色原已清楚；此頁不需重教19.15全部架構。",[0]),
("0:2","nonblocking","C可獨立依狀態/問題/候選給機率的契約原已足夠；9.11背景依原路線，不冒充本次回讀。",[0]),
("1:0","resolved","原導覽待C集中度與Auto細則，實際原路線19.16u7及19.17u1/u5已讀到，不能回改當時待教但可在此收束。",[("19.16",7),("19.17",1),("19.17",5)]),
("1:1","nonblocking","舊權重成績不能替新契約背書的版本路線原已明確；新版技術/資格仍非此頁已驗證項。",[1]),
("2:0","nonblocking","三成品按工作選擇的整體導覽原已完成；source/fig完全同19，保留原理解。",[0,1,2]),
],
"19.16": [
("0:0","resolved","後段已給Choice具體管線與測試限制，u8/u9同材料兩問題/授權否完成必要欄位的實際對照；不把手工標籤當舊C已答對。",[2,8,9]),
("0:1","resolved","原完整數值流程待後段，由u2編碼回填、u4/u5概率與u6/u7前向完成；當前各單位未變。",[2,4,5,6,7]),
("0:2","nonblocking","權重/語義/規準/三T綁定的角色原已足夠；訓練數字及實作核查未做仍保留。",[0]),
("1:0","resolved","原待倒序數值在u5按候選名逐項給出，u7又說逐列獨立的最短理由；此為後文釐清，不改u1。",[5,7]),
("1:1","resolved","原待tensor實例在u2/u4提供144字表、四列長度與4×34形狀，固定語義/字符對應可追。",[2,4,5]),
("2:0","nonblocking","CLS/SEP/PAD與四候選各一列原可追，不需新增圖；當前文字/程式該單位完全未變。",[2,5]),
("2:1","resolved","原黑箱前向未知由u6原始碼與u7逐列解釋補足；原回填操作仍成立，未開model.py替作者補。",[6,7]),
("2:2","nonblocking","隨機/成品、assert/任務能力的證據界線原已清楚；執行仍未做，不被改成已驗證。",[1,2,5]),
("3:0","resolved","原預告的精確前向在u6/u7實際出現，CLS讀一分及兩個獨立Block已教；原未知保留時點。",[6,7]),
("4:0","resolved","u5給倒序概率，u6/u7給內部前向，完成u4尚待部分；沒有把第一組單獨當置換證明。",[5,6,7]),
("5:0","nonblocking","字元PAD和候選-inf兩層補位的作用已明說；多題執行未做，當前例全部四格填滿且範圍已限定。",[2,5]),
("5:1","nonblocking","合法候選與按語義對齊的換序核對原已足夠；完整訓練標籤/批次沒有被宣稱已教完。",[5]),
("6:0","nonblocking","當前所需CLS條件評分可由正文前向與u7解釋；不要求此處再證Block全內部，未查外部原碼的限制保留。",[6,7]),
("7:0","nonblocking","C兩個各自參數Block與B循環的差別、候選置換理由原已清楚，當前不變。",[7]),
("7:1","resolved","當時外層門檻/預算待後頁，原19.17u1明示本版不另加概率門檻、u2/u5給預算；不把集中度當答對率。",[("19.17",1),("19.17",2),("19.17",5)]),
("7:2","nonblocking","雙重驗收與跨措辭CE惡化的閱讀理由成立；原實測log仍未技術核對，本次不補做。",[7]),
("8:0","nonblocking","weather→none卻高集中boxes的具體錯例已提供，足以教集中不等正確；未開保存原檔的範圍不變。",[8]),
("8:1","nonblocking","同材料兩問題及必要欄位已使資訊充分依問題而變可見，當前不變。",[8,9]),
("8:2","deferred","單一正解契約本頁已教；新C-v2聚合算法只是後文路標，當前操作不依賴且未讀19.23，保留合理後教，不宣稱已會新版。",[8]),
("9:0","nonblocking","本頁編碼/前向/回填、指標界線與充分性例已完成原問題；source/fig同19，保留原理解而非新blind。",[2,6,7,8,9]),
],
"19.17": [
("0:0","resolved","本頁後段u1前置門、u2/u5固定表與cap已完成開頭待教的manual/Auto分工；不混成神經模型能力。",[1,2,5]),
("0:1","nonblocking","新增boxes/sum需相應訓練/驗證而非臨時加名字的角色原已明確；當前未變。",[0]),
("0:2","nonblocking","swap缺A使查B無法定，欄位契約與不由B猜標籤原已足夠；沒有把客服編號規則通用化。",[0]),
("1:0","resolved","後段u5手動1/3/5與u2/u4Auto表已接上；手動優先僅改預算仍受兩門，原待教已完成。",[2,4,5]),
("1:1","nonblocking","要求/可跑預算/受限狀態當前寫清；本次仍未執行完整服務，不把文本規範當實作遵守證據。",[5]),
("2:0","resolved","u4/u5給值、u6給45組決策搜索與平均≤3的獨立目標，完成具體用法及選法理由；不宣稱事件數必等於最優難度。",[4,5,6]),
("3:0","resolved","原尚待的policy值與代理定位由u4/u5/u6完成；自然文仍可解巢狀條件，原始早判不改。",[4,5,6]),
("4:0","resolved","source21直接改為『可跑預算』且相鄰文只給2輪預算；EFFORT與選法原後文已教，純min不再被字面叫實跑。",[4,5,6]),
("5:0","resolved","BR-19.17-01的可選標籤歧義由兩處預算字句一併處理；requested3/actual預算2與真正推論budget_limited的界線一致。",[4,5]),
("5:1","resolved","原待策略效果在u6報2926/2998、固定medium2927及84輪/1題代價，且不稱最優；當前合法追蹤>=1的限定未變。",[5,6]),
("6:0","nonblocking","政策選擇目標與實際品質/工作取捨本來已清楚，並未因小表述修文改數字；實測原log仍未核。",[6]),
("6:1","nonblocking","C開銷、B輪數和重跑兩次成本的範圍原已足夠；計時實作未驗證保留。",[6]),
("6:2","deferred","各預算成功率須另標籤的基本角色已教；學習式Auto是明列後續里程碑，本版仍規則，不要求現在完成其實作/效果。",[6]),
("7:0","deferred","新C/B適配與附件可見性是19.24預告，當前只用原C+B；不靠未讀新版理解這一版，保留未驗新版。",[7]),
("7:1","resolved","兩個原可選皆在source21直接對上，平均明指完整決策題、min改稱可跑預算；頁末主線與原數字保留，無新必要負擔。",[4,5,6,7]),
],
"19.19": [
("0:0","resolved","原待清單/回報實例由u1和u4/u5提供；外層manifest指紋不同於權重指紋的對象已可定位。",[1,4,5]),
("0:1","nonblocking","eval不固定抽樣、下載不等離線載入的界線已教；本讀者仍未做命令或無網測試，不改成通過執行。",[0,3]),
("1:0","nonblocking","release兩product同commit不同prefix/hash的配對原已足夠；沒有實際下載的限制保留。",[1]),
("2:0","resolved","原待具體命令在u3完整列出，後u7/u8/u9有對應CPU收據欄位；來源路線已接完。",[3,7,8,9]),
("2:1","nonblocking","推論參數與optimizer/random完整續訓狀態的差別原已教，當前RNG補名再改善定位；沒有重訓/續訓實測。",[2,11]),
("3:0","resolved","原待三輸出由u7答2/3輪、u8info no/T.25、u9兩C判斷及B3輪完成；這仍只是閱讀作者收據，不是本讀者執行。",[7,8,9,10]),
("4:0","resolved","u5具體C下載JSON對上u1 c，完成當時待實例；files_verified5仍是檔案數非答題數。",[5]),
("5:0","nonblocking","C三身份欄與release的閱讀核對已足夠；未自行核遠端或重算包內hash，仍不冒充下載驗證。",[1,4,5]),
("6:0","resolved","原待作者CPU執行欄位由緊接三JSON完成；後u10直接解讀，原『尚未看』時點保留。",[7,8,9,10]),
("7:0","nonblocking","answer2/loops3與swap可核；stress_length精確內規仍未在此學到，但它不是版本清單或本例答案判斷所需，不假稱該未知已教。",[7,10]),
("8:0","nonblocking","info選no與T.25/缺ID的真值對應原已可讀，未實跑仍保留；高信心不保泛化的前頁限制仍適用。",[8,10]),
("9:0","nonblocking","Auto的兩判斷/requested/實際B輪數可對上既有流程；未實跑與不能外推cap可靠的限制保留。",[9,10]),
("10:0","nonblocking","答案恰好2不能證版本的必要關係原已明說；身份清單與任務成功不互相代答，當前未改。",[0,10]),
("10:1","nonblocking","三問題旗標與本機HTTP基本角色是導覽，不需在下載示範前學完整JSON服務協定；服務實際操作仍未做。",[10]),
("11:0","resolved","source21直接補RNG完整英文/中文及『的狀態』，可對到前段隨機狀態；原名稱建議已處理，並不宣稱恢復執行已驗。",[2,11]),
("11:1","resolved","唯一原可選RNG補充已完成，整頁清單鏈/本機收據/換權重重驗的主線維持；沒有新增必要缺口。",[0,1,5,10,11]),
],
}


def unit_evidence(pid, index):
    raw = chunks[pid][index].encode()
    session = next(row for row in records["sessions"] if row["page_id"] == pid)
    prior = notes[pid][index]
    return {
        "source_snapshot": ref(ROOT / pages[pid]["snapshot"]),
        "current_unit_index": index,
        "current_unit_sha256": hashlib.sha256(raw).hexdigest(),
        "same_as_original_read_unit": hashlib.sha256(raw).hexdigest() == prior["unit_sha256"],
        "original_read_note": session["notes"][index],
        "original_saved_at": prior["saved_at"],
        "changed_text_observations": [
            {"report": observations_ref, "json_pointer": "/fixes/" + str(i), "quotes": item["quotes"]}
            for i, item in enumerate(fixes) if item["page_id"] == pid and index in item["units"]
        ],
    }


basis_pages = []
for pid in IDS:
    prior = next(page for page in original["pages"] if page["page_id"] == pid)
    issues = [item for item in fixes if item["page_id"] == pid]
    changed = [i for i, chunk in enumerate(chunks[pid]) if hashlib.sha256(chunk.encode()).hexdigest() != notes[pid][i]["unit_sha256"]]
    assert changed == ({"19.17": [4, 5, 7], "19.19": [11]}.get(pid, []))
    questions = json.loads(json.dumps(prior["four_questions"]))
    if pid == "19.17":
        questions["mechanism"]["answer"] += " 修後min示範只計『可跑預算』；真正推論遇限制才標budget_limited。"
        questions["example_scope"]["answer"] = "原交換缺初值、跨事件界線、medium/cap2與2998題比較各保原效力。source21把min明稱可跑預算，並把平均明指本輪完整決策題；兩個原可選表述已處理，不把短程式當B執行或完整資料平均實測。"
    if pid == "19.19":
        questions["identity"]["answer"] += " 修後RNG已給Random Number Generator與隨機數生成器的狀態，可對到前段隨機狀態。"
        questions["example_scope"]["answer"] = "兩product同commit不同清單、下載5檔與作者CPU三輸出保原用途；版本身份不由答案2代替。RNG名稱補充已完成，續訓配套的角色更可辨，但未宣稱本讀者執行下載/離線/服務/恢復訓練。"
    for answer in questions.values():
        answer["basis_kind"] = "原實讀+來源不變比對" if not changed else "原實讀+有界source21修文核對"
        answer["current_evidence"] = [unit_evidence(pid, i) for i in changed] if changed else [unit_evidence(pid, len(chunks[pid]) - 1)]
    basis_pages.append({
        "page_id": pid, "lesson_id": pid, "reviewer_task": TASK,
        "source": pages[pid]["source"] + "#" + pid,
        "source_sha256": pages[pid]["source_sha256"], "figures_sha256": {}, "figure_sha256": {},
        "source_snapshot": ref(ROOT / pages[pid]["snapshot"]), "verdict": "pass",
        "review_kind": "retained_original_understanding_after_exact_source_and_figure_identity_check" if not changed else "bounded_revised_source_reading",
        "not_a_new_first_read": True,
        "changed_original_units": changed,
        "unit_identity_checks": [unit_evidence(pid, i) for i in range(len(chunks[pid]))],
        "four_questions": questions,
        "issue_resolutions": issues,
        "required_issues": [], "new_optional_issues": [],
        "visual_status": "source21_preview_not_yet_rechecked; earlier visual-v1 remains immutable",
        "execution_status": "not_run",
    })

basis_path = BASE / "basis-v21.json"
basis_ref = new(basis_path, {
    "schema_version": 1, "report_kind": "reader_bounded_basis", "reviewer_task": TASK,
    "review_stage": "reader", "reviewer_context": "same_reader_bounded_recheck_with_disclosed_post_route_exposure",
    "created_at": now(), "source_freeze": ref(freeze_path), "original_report": ref(original_path),
    "observations": observations_ref, "pages": basis_pages,
    "post_route_exposure_disclosure": original["post_route_exposure_disclosure"],
    "scope": "9.18/19.13/19.16 source/fig同19，保留原理解；19.17及19.19只重讀三處修文與相鄰上下文，其餘單位以原實讀紀錄及精確unit SHA同一性承接。無新blind/技術驗證/模型或final執行。",
})

estimate_rows = []
for pid in IDS:
    old = next(row for row in old_estimates["pages"] if row["page_id"] == pid)
    assert "<details" not in texts[pid].lower()
    reason = old["reason"] + (" source21此頁文字與圖完全同19，保留原實讀估計，不稱重新首讀。" if pid not in ("19.17", "19.19") else " source21只有已實際核對的表述/名稱補充，不新增操作、圖或選讀；完整snapshot的理解負擔仍在此區間。")
    estimate_rows.append({
        "page_id": pid, "minutes_min": old["minutes_min"], "minutes_max": old["minutes_max"],
        "reviewer_task": TASK, "reason": reason, "source_sha256": pages[pid]["source_sha256"],
        "figures_sha256": {}, "scope": "complete_canonical_inventory_snapshot", "supplement_status": "none",
    })
estimate_path = BASE / "reading-times-complete-v21.json"
estimate_ref = new(estimate_path, {
    "schema_version": 1, "estimate_kind": "ai_estimate", "created_at": now(), "reviewer_task": TASK,
    "source_freeze": ref(freeze_path), "basis_report": basis_ref, "previous_complete_estimate": ref(BASE / "reading-times-complete-v2.json"),
    "method": "五頁完整snapshot：原正文已實讀、無折疊選讀且無引用圖，source21有界修文已直接核對。含讀文理解程式/輸出與短暫思考，不含安裝下載執行/前置補讀；非字數公式或真人計時。", "pages": estimate_rows,
})

current_pages = []
for i, pid in enumerate(IDS):
    b = basis_pages[i]
    current_pages.append({
        "page_id": pid, "lesson_id": pid, "reviewer_task": TASK,
        "source": b["source"], "source_sha256": b["source_sha256"], "source_snapshot": b["source_snapshot"],
        "figures_sha256": {}, "figure_sha256": {}, "verdict": "pass", "basis_report": basis_ref,
        "basis_json_pointer": "/pages/" + str(i), "four_questions": b["four_questions"],
        "reader_summary": "保留原正文理解；source21三處可選修文均已直接核對並處理。" if pid in ("19.17", "19.19") else "source/fig與原實讀同一，保留原頁末理解，未聲稱新首讀。",
        "checks": {
            "background_and_links": {"status": "pass", "details": "沿原路線先備；本次不開其他頁/外部解答。"},
            "terms": {"status": "pass", "details": "原必要概念角色保留；RNG補名已直接核對，原三可選已處理。"},
            "examples": {"status": "pass", "details": b["four_questions"]["example_scope"]["answer"]},
            "program_explanation": {"status": "not_applicable" if pid == "19.13" else "pass", "details": "保留原逐段程式/輸出理解；本次未執行，預算示範與真推論另有明確語句。"},
            "exercise": {"status": "pass", "details": "保留原小變化及理由；19.17平均對象已在本輪完整決策題明指。"},
        },
        "ai_estimate": next(row for row in estimate_rows if row["page_id"] == pid),
        "complete_estimate_report": estimate_ref,
        "issues": [], "required_issue_ids": [], "open_optional_issue_ids": [],
        "resolved_original_issue_ids": [item["id"] for item in fixes if item["page_id"] == pid],
        "missing_visuals": "原判未缺必要圖；此頁當前仍無引用圖且相關文字/例子關係保留。",
        "visual_checks": {"status": "pending_source21_preview_check", "previous_supplement": ref(BASE / "visual-v1.json"), "details": "尚未收到重建預覽通知，不把舊圖標成新source21實頁檢查。"},
        "not_a_new_first_read": True, "execution_status": "not_run",
    })
current_path = BASE / "current-v21.json"
current_ref = new(current_path, {
    "schema_version": 1, "report_kind": "reader_current_judgment", "reviewer_task": TASK,
    "review_stage": "reader", "created_at": now(), "confirmed_at": now(),
    "source_freeze": ref(freeze_path), "basis_report": basis_ref, "original_report": ref(original_path),
    "post_route_exposure_disclosure": original["post_route_exposure_disclosure"],
    "pages": current_pages, "scope": "文字理解的有界修後判定；視覺重建通知尚待，不宣稱新blind或技術/執行通過。",
})

closure_pages = []
for pid in IDS:
    session = next(row for row in records["sessions"] if row["page_id"] == pid)
    explicit = {key: (status, reason, support) for key, status, reason, support in decisions[pid]}
    expected = {(i, j) for i, note in enumerate(notes[pid]) for j, check in enumerate(note["concept_checks"]) if check["verdict"] != "已足夠"}
    assert set(tuple(map(int, key.split(":"))) for key in explicit) == expected
    resolutions = []
    for (i, j) in sorted(expected):
        check = notes[pid][i]["concept_checks"][j]
        status, reason, support = explicit[f"{i}:{j}"]
        evidence = [unit_evidence(pid, i)]
        for support_unit in support:
            support_pid, unit = support_unit if isinstance(support_unit, tuple) else (pid, support_unit)
            evidence.append(unit_evidence(support_pid, unit))
        resolutions.append({
            "note": session["notes"][i], "concept_index": j, "concept": check["concept"],
            "original_verdict": check["verdict"], "original_saved_at": notes[pid][i]["saved_at"],
            "original_source_quote": check["source_quote"],
            "original_sufficiency_and_scope": {key: check[key] for key in ("identity", "mechanism", "need", "usage", "example_scope")},
            "final_status": status, "reason": reason, "evidence": evidence,
            "preserve_early_judgment": True,
            "disposition_scope": "後文原實讀/同一性承接或三處source21有界修文；不將未驗證計算/技術/新版演算法改為已驗證。",
        })
    closure_pages.append({
        "page_id": pid, "lesson_id": pid, "reviewer_task": TASK,
        "source_sha256": pages[pid]["source_sha256"], "figures_sha256": {}, "verdict": "pass",
        "checkpoint_resolutions": resolutions,
        "original_issue_resolutions": [item for item in fixes if item["page_id"] == pid],
        "necessary_issues": [], "not_a_new_first_read": True,
    })
closure_path = BASE / "checkpoint-closure-v21.json"
closure_ref = new(closure_path, {
    "schema_version": 1, "report_kind": "reader_checkpoint_closure", "reviewer_task": TASK,
    "created_at": now(), "basis_report": current_ref, "reading_basis_report": basis_ref,
    "source_freeze": ref(freeze_path), "original_report": ref(original_path), "pages": closure_pages,
    "post_route_exposure_disclosure": original["post_route_exposure_disclosure"],
    "summary": {"concepts": sum(len(page["checkpoint_resolutions"]) for page in closure_pages), "required_open": 0, "original_optional_resolved": 3},
})

# Validate structural coverage without inventing the parent's seals or intake times.
for pid in IDS:
    page = next(row for row in closure_pages if row["page_id"] == pid)
    original_map = {(session_ref["sha256"], j): check["verdict"]
        for session in records["sessions"] if session["page_id"] == pid
        for session_ref in session["notes"]
        for j, check in enumerate(json.loads((ROOT / session_ref["path"]).read_bytes())["concept_checks"])
        if check["verdict"] != "已足夠"}
    result_map = {(item["note"]["sha256"], item["concept_index"]): item["original_verdict"] for item in page["checkpoint_resolutions"]}
    assert result_map == original_map
    assert all(item["reason"] and item["evidence"] and item["final_status"] in {"resolved", "nonblocking", "deferred"} for item in page["checkpoint_resolutions"])
assert json.loads(closure_path.read_bytes())["basis_report"] == current_ref
assert json.loads(current_path.read_bytes())["created_at"] >= json.loads(basis_path.read_bytes())["created_at"]
assert json.loads(closure_path.read_bytes())["created_at"] >= json.loads(current_path.read_bytes())["confirmed_at"]
from scripts.reading_time import load_estimates
assert set(load_estimates(estimate_path, {"pages": list(pages.values())}, require_complete=True)) == set(IDS)
assert all(sha(BASE / name) == digest for name, digest in preserved.items())

intake_path = BASE / "bounded-v21-intake.json"
intake_ref = new(intake_path, {
    "schema_version": 1, "reviewer_task": TASK, "created_at": now(), "source_freeze": ref(freeze_path),
    "observations": observations_ref, "basis": basis_ref, "current": current_ref, "checkpoint_closure": closure_ref,
    "complete_estimates": estimate_ref,
    "pages": [{
        "page_id": pid, "json_pointer": "/pages/" + str(i),
        "source_sha256": pages[pid]["source_sha256"], "figures_sha256": {}, "verdict": "pass",
        "original_report": ref(original_path), "original_json_pointer": "/pages/" + str(i),
        "basis": {"report": basis_ref, "json_pointer": "/pages/" + str(i)},
        "report": current_ref, "report_json_pointer": "/pages/" + str(i),
        "checkpoint_closure": {"report": closure_ref, "json_pointer": "/pages/" + str(i)},
        "sessions": [next(row for row in records["sessions"] if row["page_id"] == pid)],
    } for i, pid in enumerate(IDS)],
    "validation": {"exact_original_nonplain_concept_coverage": 74, "source_and_figure_hashes": "pass", "creation_order_basis_current_closure": "pass", "complete_inventory_estimate_schema": "pass", "originals_and_40_notes_unchanged": True},
    "actual_collection": "尚待主代理實際解析並保存真收件；此檔不捏造received_at、review wave或seal。",
    "parent_to_fill_after_actual_intake": ["wave", "collection_role", "received_at"],
    "visual_status": "Awaiting source21 rebuilt preview notification; visual-v1 untouched.",
})
print(json.dumps({"observations": observations_ref, "basis": basis_ref, "current": current_ref, "checkpoint_closure": closure_ref, "estimates": estimate_ref, "intake": intake_ref, "validation": {"concepts": 74, "required_open": 0, "optional_resolved": 3, "originals_unchanged": True}}, ensure_ascii=False, indent=2))
