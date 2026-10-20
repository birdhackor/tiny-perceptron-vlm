import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

TASK = '/root/rv1019_reader_ch08'
OUT = Path('docs/course-review-20261019/reviews/rv1019_reader_ch08')
SESSIONS = Path('docs/course-revision-20261016/reader-sessions')
rows = [
    ('chapter-08','06e3ed439ae2459aaae080ccbbf74523',[(0,0)],'本章從明確要求出發，區分內容、範圍與格式責任，再看提示、示範與少量參數修正。導覽已交代LoRA與自動裁判可選，不需先完成訓練流程才能學要求。','無；本頁沒有折疊選讀。',2,4,'短導覽，需理解兩個要求的對照與主線、選讀路徑。',['LoRA與自動裁判的機制尚未知，預期在後面的延伸頁介紹，導覽不需要先懂。']),
    ('8.1','a9e5407f10f24b92a4b2a048be9a4769',[(0,0),(1,0),(2,0),(2,1)],'將有趣拆成數字正確與比喻對上分組。三份手寫回答、人工標籤及海的改句讓兩項可分判，程式只列印標籤。','選讀補充固定積木句是窄風格指標，不能量機智或洞察；正文已足以理解rubric。',8,12,'讀三例與真假輸出，改句重判標籤，再思考新題才能驗學會。',['如何訓練參數與驗證新題尚未操作，預期8.3承接；通用貼切比喻不是本頁成果。']),
    ('8.2','cdfb19bb8e6b48c8a9962d605871932b',[(0,0),(1,0),(2,0),(2,1)],'prompt把要求放進當次前文，可能影響下一個token而不更新參數。程式只組裝提示，真正比較須固定模型與選字規則，再看回答分項。','選讀補上實驗的三種已教style條件，強調不是任意新風格名字都會懂；沒有補正文必要缺口。',8,12,'讀兩份提示輸出，分清輸入改動與訓練，再預測只改第二要求的結果。',['任意新要求是否有效尚未測；本頁只教比較設計，8.4給已教條件的實際切換例。']),
    ('8.3','37d06a740ca546239c7c39f993cc2c94',[(0,0),(2,0),(2,1),(3,0)],'以不同目標回答作SFT示範，先檢查byte與EOS形成的有效目標量。短答2、比喻44及改句17可追算；實驗寫法變而答案仍錯，不能推通用能力。完整訓練與對話標籤對位仍依指名的7.1。','選讀增加450步、每批16及16,591對318,991有效目標，並說底座原先七題也全錯，不能說文風訓練破壞原有加法；未外開完整紀錄。',16,24,'需追對話資料、byte與EOS計數，解讀失敗案例、訓練量差異並做改句計數練習。',['完整SFT損失、梯度與對話輸入標籤對位需回查7.1，未讀故未驗證；本頁數目標與判比較限制可跟上。']),
    ('8.4','e307d1cf13d74910a7552654a6481a36',[(0,0),(1,0),(2,0),(3,0)],'讓同題跨兩種要求配目標，才能表達同模型讀條件切換寫法。四筆資料與三種實際回答有對應，格式切換成功與算術全錯分開。','選讀增加同49題三風格、1,000次更新與每條件7/7風格、0/7數值；另交代EOS皆有及JSON整數檢查。沒有把新風格理解混入已教條件切換。',14,20,'追兩層迴圈四筆資料與三列結果，做刪風格對照練習，並讀窄風格評分限制。',['能否理解沒教過的style名稱或創作新比喻尚未知，屬不同能力，不是本頁切換結論。']),
    ('8.5','b39022de23664537819a06a364185fcc',[(0,0),(2,0),(2,1)],'將JSON可解析、欄位合規與數值正確分開，明定內容欄須格式先過。三份手寫輸出與多note練習足以辨認不同失敗；預設解析器及值型別限制已明說。','選讀以模型七題JSON皆可解析且型別合規、數值全錯補強反例；正文已完成解析和正確性區分。',12,18,'需讀try/except、型別與鍵集合，再對三行輸出與加欄位練習逐項核對。',['完整嚴格JSON與值型別檢查未實作，正文明示需按任務另加，不影響這三例的解讀。']),
    ('8.6','c56af5a763fc48dd8dbb8cfcac2c9c76',[(0,0),(1,0),(2,0),(2,1)],'將澄清限制在必要日期缺少時才問，有日期則確認。人工規則與真模型保留日期的失敗分清，測試需同時有缺與齊全例，日期變體分組。','選讀說明24日期分19訓練、2驗證、3測試，缺日期輸入以?隱去分組日期；6題3對其實是追問全對、確認全錯，不是會安排會議。',14,20,'兩分支與模型失敗要分讀，再紙上列日期時間四組、設只問缺項的回答。',['任意自然措辭理解與實際安排會議未測；範圍改成日期時間只做規則擴充練習。']),
    ('8.7','e52b0f2f6681437bb8f68b4ed8ef4c9f',[(0,0),(1,0),(2,0),(2,1)],'固定回答與分項，只改風格權重可翻轉排名；低平均代價也可能由長固定句的易猜位置拉低。圖、公式與七題零對例讓總分取捨和分母組成可分別理解。','選讀列代價合計、有效目標與平均對應的紀錄欄位及較精細值，強調有助理解須實際核對關係；沒有改變正文結論。',16,24,'手算兩權重及平手，核對圖，再追有效目標加總與平均的兩個分母。',['實際人類風格偏好與跨題材有助理解的評分未測，不可由此手算或固定句推論。']),
    ('8.8','4e08dd7b77e147fb82114415657aae72',[(0,0),(0,1),(1,0),(3,0),(5,0),(6,0)],'LoRA保留凍結原路，加A/B窄通道修正；形狀、低秩限制、56可學數、初始相同與第一次梯度都有相鄰解釋。零B保原輸出、非零A允許B起步的理由完整，省可調數不等於品質或總模型同倍縮小。','選讀補完整模型11處插入、9,504可調數約6.7%、同底座同資料同有效目標的全調對照，保留6/7風格與亂碼失敗；Query/Value/FFN只作插入位置背景且附前文回查。',40,55,'需讀低秩矩陣與兩路圖、兩段程式、梯度起步因果，做rank4計數，再理解完整模型選讀對照。',['最適rank、插入位置選擇與更廣泛任務品質未知，本頁未聲稱解決；合併與縮放分別待8.9、8.13。']),
    ('8.9','af90bb9bbf7a4c3d99829e4d11469e0c',[(0,0),(2,0),(2,1)],'adapter是依賴底座的整套A/B與設定，手設兩套數字示範備份還原。只看B平均不夠，完整equal可抓未還原A；合併W+ΔW後不再重套同修正。','選讀給真正短答→比喻→短答分數差0、合併差約0.00000668且容差0.0001，限制在同一前文，不擴成所有題風格品質。',14,22,'追兩套狀態的clone/copy與兩個equal，做刪A還原反例，再讀合併與容差。',['實際adapter檔案載入命令未操作，正文已列需保存設定，選讀指T.5；本頁只驗記憶體矩陣還原。']),
    ('8.10','61312ff823c742879ef3ae80e8f9a5f9',[(0,0),(2,0)],'自動裁判先拿人工可解釋正反例核對。四例一致3/4、分歧索引1能回讀原文，但一致率不等於好回答率，也不解釋分歧原因。','無；本頁沒有折疊選讀。',12,18,'逐份讀四回答關係，對布林平均與索引輸出，修改標籤並設計長文正反例。',['任何真實裁判是否可靠尚未測，給定標籤只能教校準方法；位置與長度偏差待接下兩頁。']),
    ('8.11','e8d7837879bc4e99ac8d70c5d130f766',[(0,0),(2,0)],'固定回答身分再交換顯示位置，選第一項會先正後錯，固定選correct則兩次同內容。測位置偏差要保留題目、規則、順序和選中名稱。','無；本頁沒有折疊選讀。',10,15,'追字典身分與反轉清單，讀三行輸出，再改選擇函式預測對照。',['真實裁判的位置偏差程度未測；身分一致也不能單獨保證判斷正確，本例另有正答判準。']),
    ('8.12','37b4ade2178b4c1c8f49c784dfe2cf69',[(0,0),(2,0),(2,1)],'先控制事實只重複增長，以len假裁判示範偏長；再對照真正增加分組解釋的長答。不同句子數只能看這個重複例，不能當通用資訊量。','無；本頁沒有折疊選讀。',12,18,'讀長度5與50、split去末空項和set去重，再改2次並對照有用長版。',['如何通用自動判資訊收益未教，本頁明說粗計數不行，需回到內容判準。']),
    ('8.13','c641c235490547309f5b2c65ff1fd60f',[(0,0),(2,0),(2,1)],'同名alpha/rank要核對實際公式；固定A/B與rank只改alpha時修正倍增，總權重不倍增。改rank連矩陣內容形狀也變，不能只由倍率推修正或品質。','選讀增加底座數字指紋的核對意義，並明說沒有訓練其他rank或另一公式，無從挑最佳rank。',12,18,'核算單格與兩公式，讀浮點差及容差，再做alpha6三倍預測。',['哪種縮放公式或rank效果較佳未測，這裡只教設定相容與固定數字的機制。']),
    ('8.14','6dcf531b16b34b7fa22332878844a5ef',[(0,0),(1,0),(2,0),(4,0),(5,0)],'從完整可讀告示取指定中文兩行，不需照片辨識。圖與四候選輸出呈現包含首句不足，應同查完整範圍與交付方式；只改第一行要求會改變合格候選。','選讀列InstructGPT與FLAN作分項評估的來源與界線；未開論文或來源筆記。正文例子已足夠，不需選讀才分得開內容、範圍和交付。',20,30,'讀四列表與選取圖，追四候選兩種檢查，再更改要求重定目標、說明結果翻轉。',['這不是照片辨識或一般開放式內容評分；分項JSON檢查待8.16。']),
    ('8.15','c7aa16d86efd4e79962d08e2cad45fe0',[(0,0),(2,0),(3,0),(4,0),(4,1)],'同告示配全部、中文與中文JSON三種要求，輸入需含請求且目標真正執行它。解碼有效目標與EOS能查資料，render_chat不會抓語意錯配；新措辭與新材料測試須分清。','選讀補SFT與指令資料的來源及泛化界線，EOS與首答案位置分別指7.3、7.4、7.9；未開範圍外內容，不把來源當首讀已知。',22,32,'核兩份對話資料53/26與EOS、看配對圖和JSON，再做英文27與錯配目標的練習。',['實際模型能否在新材料或新措辭依要求完成未知；本頁只整理資料，沒有訓練效果。']),
    ('8.16','8c4f00d8e14c42978a4965edd902b8a0',[(2,0),(2,1),(3,0),(6,0),(9,0),(9,1),(10,0)],'內容、範圍、裸JSON格式與正常EOS各有責任，同題須全部通過。六例教分項，八反例展示check字串代理的誤判，後文明說工具邊界；正常停因擷取依7.9未驗證，概念和紙上結果可理解。','選讀介紹IFEval逐項與同題全過的差別，指出官方JSON規則可容外框、不等於本頁要求且不驗EOS，另指生成設定的EOS/強制EOS/上限區分；未讀外部文件。',40,55,'需追四卡圖、六例、check三段程式與全部輸出、八例誤判及Unicode轉義，再做語言目標切換與全且分母練習。',['實際生成紀錄如何取得EOS與停因需7.9，尚未驗證；完整通用裁判與開放式建議檢查不是本頁有限字串規則可提供。']),
]

issues = {
 '8.3': [dict(id='8.3-1', unit_index=0, quote='做法是[7.1](07.md#7.1)的監督式微調（SFT）：用人寫的目標回答教模型，在給定提問後提高目標文字的機率。', problem='我未讀過7.1；完整SFT中輸入與答案標籤的對位，以及目標機率如何形成損失並更新參數，需回查該節。本頁有明確回查位置，局部目標計數已可理解。', impact='無法獨立驗收完整訓練機制，並非本頁缺少所有概念介紹；照順序讀的學生有路回查。', severity='增加理解負擔', type='先備依賴', necessary=False, suggestion='保留7.1回查，於依賴審閱核對該節是否足以提供對話標籤與SFT更新鏈；本輪不要求重講。')],
 '8.16': [dict(id='8.16-1', unit_index=5, quote='真正評估時，它要從生成紀錄讀出模型是選了EOS，還是被上限截停（[7.9](07.md#7.9)）。', problem='我能說明正常EOS與上限停的差別，但未讀7.9，尚不能驗證實際生成紀錄如何取得並區分停因；頁面已明確指向7.9。', impact='概念分項與紙上check可讀懂，真模型正常結束的操作查核仍待先備覆核。', severity='增加理解負擔', type='先備依賴', necessary=False, suggestion='於依賴審閱核對7.9的EOS及停止原因記錄是否足夠，保留現有回查；本輪不要求完整重述。')],
}
figures = {
 '8.7': ('/Users/yclin/project/tiny-perceptron-vlm/docs/course-revision-20261016/frozen-figures/a35b5a8aaffc9cf44c644a25b45d7a5a6cf67159ddb1f2c0a27d8d5cd4a31e39.svg','a35b5a8aaffc9cf44c644a25b45d7a5a6cf67159ddb1f2c0a27d8d5cd4a31e39','藍框短答4為風格1正確1，橘框錯答5為3與0；下兩框權重0.2顯1.2>0.6，權重2顯3<6，原正確分不變，對應固定回答只換尺。'),
 '8.8': ('/Users/yclin/project/tiny-perceptron-vlm/docs/course-revision-20261016/frozen-figures/4349f1c9a64ae732be807f4ca9ec7cb084123465daec984a608d8c37ff35ba81.svg','4349f1c9a64ae732be807f4ca9ec7cb084123465daec984a608d8c37ff35ba81','頂部同一16數輸入分流到左W16→12、192權重凍結，及右A16→2接B2→12、32+24=56可學。兩路在黃相加框合流成12數；框內是資料寬度不是權重存放形狀。'),
 '8.14': ('/Users/yclin/project/tiny-perceptron-vlm/docs/course-publish-20261008/freeze/current/figures/course/figures/rewrite-08-new-8.14-selected-lines.svg','d55bd46951f2f6ea24ea2df0e3e682de67a066f4365198046e3b49ef7b5c210c','綠框列1今日休館與列3明日開放經右側綠線指向下方同序兩行；灰框英文列2/4仍可見但不連輸出。黃要求框只抄中文一行一項，圖中列號不進回答。'),
 '8.15': ('/Users/yclin/project/tiny-perceptron-vlm/docs/course-publish-20261008/freeze/current/figures/course/figures/rewrite-08-new-8.15-request-target.svg','da70f52bc2bd24453a5514077e9e761b5754785d4e92bfb3b800302d45ec5675','頂灰框同一四行材料，兩箭頭分別配藍色全部照抄目標四行與綠色只中文目標兩行。各框請求與目標分隔標清，箭頭是資料配對而非模型內運算。'),
 '8.16': ('/Users/yclin/project/tiny-perceptron-vlm/docs/course-revision-20261016/frozen-figures/4b364f4149729e847dba63e0eb47e847b30e9ca794a4e42f08cc8d8055cc2655.svg','4b364f4149729e847dba63e0eb47e847b30e9ca794a4e42f08cc8d8055cc2655','黃框完整要求下，四藍卡分內容原文順序、範圍兩句且不多、格式裸JSON單text字串清單、正常結束主動EOS未截停。右側綠線合流至底部四項全過，表示同題全且而非四部件。'),
}

def sha(path):
    raw = subprocess.check_output(['shasum','-a','256',str(path)], text=True)
    return raw.split()[0]

def evidence(path):
    return dict(path=str(path), sha256=sha(path))

pages=[]
notes_total=0
for page, session, picks, summary, optional, low, high, reason, unknowns in rows:
    directory = SESSIONS/session
    opening_path = directory/'opening.json'
    opening=json.loads(opening_path.read_text())
    note_paths=sorted(directory.glob('note-*.json'))
    notes=[json.loads(p.read_text()) for p in note_paths]
    assert all(n['reviewer_task']==TASK for n in notes)
    assert [n['unit_index'] for n in notes]==list(range(len(notes)))
    checks=[]
    for index, number in picks:
        item=copy.deepcopy(notes[index]['concept_checks'][number])
        v=item['verdict']
        mv = '未驗證' if v == '未驗證' else '已足夠'
        item['mechanism']=dict(answer=item['mechanism'],verdict=mv)
        item['need']=dict(answer=item['need'],verdict='已足夠')
        checks.append(item)
    current_issues=issues.get(page,[])
    viewed=[]
    if page in figures:
        svg,digest,observation=figures[page]
        assert sha(svg)==digest
        viewed=[dict(svg=svg,png=f'outputs/course-review-20261019/figure-renders-03/{digest}.png',observation=observation)]
    pages.append(dict(page_id=page,source_sha256=opening['source_sha256'],figures_sha256=opening['figures_sha256'],verdict='需補說明' if any(i['necessary'] for i in current_issues) else 'pass',reader_summary=summary,checks=checks,issues=current_issues,unknowns=unknowns,missing_visuals='無；已展示的文字材料、輸出與原位置凍結圖足以追蹤本頁必要關係。未驗證網站桌面、手機的實際排版、折疊與閱讀路徑。' if viewed else '無；本頁主要是文字、清單或數值比較，完整例子與輸出已足以追蹤必要關係，不需額外圖。未驗證網站桌面、手機的實際排版與閱讀路徑。',figures_viewed=viewed,optional_reading_observation=optional,ai_estimate=dict(minutes_min=low,minutes_max=high,reason='AI估計，非真人計時；'+reason+'；包含正文、已有選讀、程式輸出與紙上練習，不含安裝或完整訓練。'),session=session,session_evidence=dict(opening=evidence(opening_path),notes=[evidence(p) for p in note_paths])))
    notes_total+=len(notes)
report=dict(schema_version=1,reviewer_task=TASK,created_at=datetime.now(timezone.utc).isoformat(),report_kind='original first-read judgments after sequential reading',source_freeze=evidence(Path('docs/course-review-20261019/freeze-03.json')),reading_plan=evidence(Path('docs/course-review-20261019/reader-plan-ch08.json')),context_reading='先以supplement完整讀first-steps及指定兩張PNG。實際用到W.2字典、迴圈、清單索引切片、f字串、set/key=len、try/except；W.3軸、形狀、tolist/item；W.4機率與負log代價；W.5矩陣乘法、Linear權重轉置、偏移、copy_/no_grad；W.6參數、梯度、backward與真正更新的差別。暖身兩圖分別核對跨軸平均與w=1到1.4代價4到2.56。教材內容僅經指定reader與executed_outputs工具取得，未執行教材程式；全部正文後才補選讀。',method_incidents=[],pages=pages)
path=OUT/'reader-original-01.json'
with path.open('x') as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps(dict(reviewer_task=TASK,pages=len(pages),notes=notes_total,necessary_issues=sum(i['necessary'] for p in pages for i in p['issues']),prerequisite_issues=sum(len(p['issues']) for p in pages),report=str(path),sha256=sha(path)),ensure_ascii=False))
