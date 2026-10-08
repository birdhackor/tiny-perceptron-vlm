import json,pathlib,hashlib,re,datetime
B=pathlib.Path('/workspace/work/tutorial-audit-20261008')
M=json.loads((B/'manifest.json').read_text()); P={p['page_id']:p for p in M['inventory']['pages']}
p=B/'work/continuity-modalities/actual-page-notes.json'
notes=json.loads(p.read_text())
need_quotes={
'chapter-10':'先把像素、圖塊與位置對應成數字，再接入文字序列。',
'10.1':'希望程式讀出中央的 `[1,0,0]` 與左上的 `[0,0,0]`，還要說清楚這些數字屬於哪一張圖、哪個位置。',
'10.2':'沿用上一節的 16×16 黑底紅色方塊圖，要排成一串可依序處理的材料。',
'10.3':'它們的用途要由訓練目標建立。',
'10.4':'問「左邊是什麼」時，需要把位置也留下。',
'10.5':'本節只檢查這條影響能否傳回圖片入口。',
'10.6':'圖片入口每位置給 8 個數字，文字核心每位置需要 12 個。',
'10.7':'除了移輸入，也要移對應目標，讓助手位置仍預測答案第一個字。',
'10.8':'沒有附圖時，希望兩者的候選分數逐項相同；這是介面接通前後的計算核對，不是問模型是否會寫文章。',
'10.9':'先除以自身長度，讓方向比較不受倍率支配，叫歸一化。',
'10.10':'希望每張圖的正確欄分數比另一欄高。',
'10.11':'雙向平均表達兩種查詢都重要，但不保證每次效果勝過單向。',
'chapter-11':'圖片接到文字介面之後，還要確認它能改變正確答案。',
'11.1':'文字描述「紅色圓形」時能回答紅，換成紅圓圖片時也應回答紅，再換藍圓圖片時應改答藍。',
'11.2':'只想教圖片接頭，就先把它的名字與數量列出，再確認優化器收錄同一批參數。',
'11.3':'如果圖片特徵能分相關屬性，文字核心也會用這些詞，可以先只調接頭，讓視覺線索接到文字行為。',
'11.4':'圖片沒變，問題決定要取哪項資訊；這才是圖片問答，而不只是每次念出完整描述。',
'11.5':'先數每種方案允許改幾個數字，再用相同資料與預算比較答案；參數多不代表新題一定更好。',
'11.6':'如果第一方案多花一整段訓練，最後較好可能只是多用了資料。',
'11.7':'把兩類並排，才知道學習共享文字權重的代價。',
'11.8':'先算這個固定猜測基準，才知道高分是否真的需要圖片。',
'11.9':'先看前處理後的圖，再判斷是哪一段出問題。',
'11.10':'更細的材料可能保住小物件，但也佔更多上下文。',
'11.11':'稀少需求的使用者卻一次也沒成功。',
'11.12':'先把這些真正可見的筆畫與標籤配好，再談辨識器。',
'11.13':'查驗編號要用完整字串，所以要把逐位正確與整串成功分開。',
'11.14':'兩圖的紅藍總量相同，但「紅在藍哪邊」的答案不同。',
'11.15':'更大文字模型可能憑語境猜字，卻無法可靠從相同模糊數字還原不同原筆畫。',
'11.16':'如果先念第二行，字的種類一樣，訊息順序仍變了。',
'11.17':'物件名稱沒有變，但第二題還需要位置。',
'11.18':'這條路把區域選取交給已知的框，讓我們先專心教少量字形和字序。',
'chapter-12':'本章先用合成單音看清資料流與控制；後半段分開聽寫與需求回答，最後讓真人短句的有限意圖和打字共用同一段對話。',
'12.1':'希望用一串數字表示聲音隨時間的變化，並分清一個數字與一個完整週期。',
'12.2':'這個「每秒多少點」叫取樣率，是讀懂時間軸的必要資料。',
'12.3':'短框方便表示局部頻率，但框太短時很難分清接近的頻率；框太長又把較長時間混在一起。',
'12.4':'頻率計算也會把這個接點跳變算進去，帶出原本不是單音本身的額外頻率成分，讓能量分散。',
'12.5':'希望每個短時間框都能指出「哪些頻率較強」，並核對最強的格子是否對應 440 Hz。',
'12.6':'頻譜每框有 201 個頻率格，想用 16 個較寬區段表示能量。',
'12.7':'若直接用同一亮度尺度，前三格可能幾乎都看不見。',
'12.8':'依先後排成 11 條特徵，供後面的模型逐位置讀取；這一步不把整段平均成一個數字。',
'12.9':'先檢查聲音入口能否接到文字模型，讓答案誤差沿這條路徑傳回。',
'12.10':'有分數不等於有使用聲音。',
'12.11':'若問題是「哪段音高較高」，這兩個相同字串就不足以區分。',
'12.12':'單看任一路都不足以確定完整答案。',
'12.13':'若回答需要先後，入口與後續模型要保留時間位置，並接受相關任務的教學材料。',
'12.14':'若採先聽寫的路線，希望兩種入口都把文字放進同一段對話，再由同一個聊天核心根據這個條件推薦餐點。',
'12.15':'這裡想教的是聽懂幾種需求後回應，不要求把任何人說的每個字都完整抄下來。',
'12.16':'換輸入方式，不應把原先的格式要求或談到的主題清掉。',
'chapter-13':'對同一問題，你可能知道哪份回答更合適，卻很難只寫一份唯一標準答案。',
'13.1':'保存原因讓後來的人知道，A勝出是因限制，不是所有比喻都不好。',
'13.2':'若rejected連提問都換了，分數差也可能來自題目難度，已不是本次同題比較。',
'13.3':'很多小數連乘會很小；取自然log後，乘法變加法。',
'13.4':'它相對原差距改善了`0−(−1)=1`，但兩候選現在只是同分，尚未保證較佳勝出。',
'13.5':'因此相對更偏向較佳答案，代價就降低。',
'13.6':'只改beta，我們要看同一位置附近的代價與梯度，而不是比較重新訓練後的模型。',
'13.7':'模型可能學到增加文字量，而非增加有用解釋。',
'13.8':'希望得到的是準確而尊重的修正。',
'13.9':'因為新舊任務使用同一組語言模型權重，更新一項可能影響其他項。',
'13.10':'它叫**獎勵模型（reward model）**，讀問題與回答，輸出估計這份回答有多合適的分數。',
'13.11':'希望選擇規則更常作出比預期好的選擇，較少作出比預期差的選擇，才把評分轉成更新方向。',
'13.12':'重用這批紀錄時，要保留作答當時的機率，才能知道已經改了多少。',
'13.13':'當它已從1升到1.3時，能否暫停同一筆舊評語的額外鼓勵？',
'13.14':'本配方另對policy加入偏離reference的代價，叫**KL散度**（Kullback–Leibler divergence）。',
'13.15':'這樣縮小任務，是為了能追清PPO中每份數值的去向。',
'13.16':'實測要用不依賴reward分數的任務檢查。',
'13.17':'也可以直接用偏好對調整policy，省去獨立評分員，這條路叫**DPO（直接偏好最佳化）**。'
}
visual360={'10.1':['rewrite-10-pixels'],'10.2':['rewrite-10-patch-order'],'10.7':['rewrite-10-image-targets'],'12.5':['p7-12-frequency-match'],'13.13':['rewrite-13-clip-cases'],'13.14':['rewrite-13-model-roles']}
seen_pages={'11.17':['11.17-390-context-0.png','11.17-390-context-1.png','11.17-1280-context-0.png'],'12.5':['12.5-390-context-0.png','12.5-390-context-1.png','12.5-1280-context-0.png'],'13.14':['13.14-390-context-0.png','13.14-1280-context-0.png']}
for x in notes:
 pid=x['page_id']; raw=pathlib.Path(P[pid]['snapshot']).read_text()
 if pid=='13.1': x['quoted_basis'][0]=x['quoted_basis'][0].replace('偏好','**偏好**',1)
 if pid=='13.4': x['checks']['operation']=x['checks']['operation'].replace('-.4/-.3','−4/−3')
 q=need_quotes[pid]; assert q in raw,(pid,q)
 x['quoted_basis'].append(q)
 x['source_sha256']=P[pid]['source_sha256'];x['source_path']=P[pid]['snapshot'];x['source_position']=P[pid]['selector']
 x['quoted_basis']=[{'quote':q,'source':pid,'line':raw[:raw.index(q)].count('\n')+1} for q in x['quoted_basis']]
 x['checks']['mechanism']['basis']=x['quoted_basis'][0]
 x['checks']['need']['basis']=x['quoted_basis'][1]
 if pid in visual360:
  x['visual']+='；另已實看360：'+','.join(visual360[pid])+'，关键数字、符号、箭头及角色仍可辨认；不据此声称实际页全通过'
 if pid in seen_pages:
  x['page_capture_observation']={'captures':seen_pages[pid],'judgment':'已在390手機和1280桌面局部图文看必要关系。11.17单物件/关系、12.5四点与谱轴、13.14五角色字体可读并邻正文；滚动上下边缘截字只属当前视窗，未见该处关系被遮挡。','limitations':'既有新站预览省略runtime outputs/reading annotations；仅这些位置，未操作折叠、跳转或重跑页面。'}
 if pid=='10.3': x['later_main_clarification']={'read':'2.1、2.3','answer':'已实讀文字embedding按ID选可调特征列、Linear沿末軸共享，早期待补现可用；原unknown观察保留。'}
 if pid=='10.5':x['later_main_clarification']={'read':'1.8、1.10、1.11、1.12','answer':'已實讀交叉熵正答机率、链式求導、梯度符號与step/重新计算；早期待补现可用。'}
 if pid=='10.7':x['later_main_clarification']={'read':'3.6','answer':'已實讀因果许可i/j≤i與查询×来源方表，六格新序列用途由已读关系支持。'}
actual=['7.1','7.3','7.4','7.5','5.17','3.6','1.8','1.10','1.11','1.12','2.1','2.3','5.3','5.5','4.3','4.4','7.18']
obj={'reviewer':'/root/continuity_modalities','role':'continuity','group':'modalities','saved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'criteria_sha256':hashlib.sha256((B/'freeze/criteria/SKILL.md').read_bytes()).hexdigest(),'route':'完整66页主文銜接審，details尚未读取；全文銜接而非逐段盲讀','actual_prerequisites':[{'page_id':pid,'source_sha256':P[pid]['source_sha256'],'scope':'通过continuity_source modalities main实读全文；details未读；引用正文表/文字/程式关系，不宣称已看该前文图'} for pid in actual],'pages':notes,'issues':[],'main_judgment':'在本轮已读实际先备和正文范围内，没有发现需要后文或details才能补足的必要概念/设计/规则/限定銜接缺口；各页保留机制与用途及具体边界，未知结果/执行/未看页面不算通過。','coverage':{'required_pages':M['groups']['modalities']['pages'],'main_actually_read':66,'not_blind_incremental':True,'details_read_before_seal':0},'unverified_scope':['原始实测与数据/论文事实未作技术独立查证','未运行教材程序或训练','未验全组实际网页与交互，仅3页局部新站captures','前文图未看，仅依所读正文与表格核先备']}
assert len(notes)==66
out=B/'reports/continuity-modalities-main-notes.json'
assert not out.exists(),out
out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print('saved',str(out),'sha256',hashlib.sha256(out.read_bytes()).hexdigest(),'pages',len(notes))
