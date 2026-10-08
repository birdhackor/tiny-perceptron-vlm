from pathlib import Path
import json,re,datetime,hashlib
B=Path('/workspace/work/tutorial-audit-20261008');M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}
E={
'14.1':('一次小模型比較既沒有證明故事寫得好，也沒有測試超過128位置的外推。','補原文定位與實測條件，參數不同/byte分母/短訓外推界線；不是主文旋轉理由必要補洞。數值未驗原始實報。'),
'14.2':('這輪沒有測到速度收益','補RMS實測微差/省320参/速度未收益與單種子；主文簡化不等同規則的理由已足夠。'),
'14.3':('正式modern比較沒有QK normalization組','只是實測不存在的範圍限定，不新教當下必要規則。'),
'14.4':('不能把更低的代價換成「已能寫好故事」','補ReLU²同param真短訓數字及失敗，主文非線性需要未依賴此效果排名。'),
'14.5':('並非重現論文的等參數比較','補原GLU及本次中間h不縮的比較範圍；公平參數控制正文已教。'),
'14.6':('輸出分布已不同，所以不能用這次短訓斷言共享本身傷害品質','補共享組初值loss差與訓練代價，不把共享機制/省參數當品質證據。'),
'14.7':('本節十六張卡是紙上例子，沒有訓練或測試模型。','權威來源與有效長度論文定位；主文三長度界線自足。未開來源筆記/原論文。'),
'14.8':('沒有保證任意模型只改一個設定便能保持品質','補PI訓/驗論文節定位，主文刻度與適應用途都已教。'),
'14.9':('本節60度與15度是人為挑選的示意轉速','補原文多頻與示意數字界線，不改主文通過。'),
'14.10':('另一個改動可以寫成`新分數 = c × 原分數`','補γ過渡與c算式、分數尺度仍只是操作效果；未補擴长為何需要另一尺度改動，CA-14.10保留段落級need負擔。'),
'15.1':('不是上述12個權重的玩具程式','補真Dense基準size與訓練結果，未混玩具權重實測。'),
'15.2':('不能證明八份已各有專長','補完整八expert參數數，不冒按領域專長。'),
'15.3':('後者數的是進入前兩名的次數，不是原softmax值相加','補實際連續比例與top2份額差；主文與15.8已足夠區分。'),
'15.4':('分母應是有效輸入數×k','補多層、PAD、top1/2統計分母實際範圍；正文k每token與份額规则已足。'),
'15.5':('本教材的top-2再正規化是另一個明示的選擇','原Switch形式定位與top1/2约定不同，主文15.7已解任務梯度需要。'),
'15.6':('沒有因為統計忽略PAD就自動跳過它們的工作','補Python dispatch非稀疏GPU核心及PAD運算/統計區分，16.5主文再教PAD工作。'),
'15.7':('沒有證明離散的expert索引能直接求導或路由已最優','補非零router實測梯度/原長度與aux乘數，主文已教連續係數學習路径。'),
'15.8':('這個熵彙整整份驗證集，不保證每一批、每一類文字都均勻','負載熵新的選讀統計有完整formula、0份額處理/均勻normalizer與加權批平均；主線用bincount不依赖熵。'),
'15.9':('兩種分母不可混用','補top2 f分母、兩層aux和及四組單種子結果，task heldout不含aux／batch_aux非全局重乘清楚。'),
'15.10':('這個代理卻刻意把兩張完整表的參數全列入','完整active_proxy不是實讀權重/FLOPs，主文15.13已說同一限制。'),
'15.11':('開始前配置不同，所以不是各模型單獨部署的記憶體比賽','補L4完整update計時和allocator口徑，主文成本理由不依賴AdamW內部全算法。'),
'15.12':('本次七組訓練沒有啟用固定expert容量、丟棄或重送','公式來源/未啟用政策範圍，非主文capacity分析缺少執行。'),
'15.13':('代理也不是相同FLOPs','補7組近似匹配剩差/不同初始化/有效目標、固定教學teacher未挑最優及重複續寫。主文公平資源問題自足。'),
'16.1':('不能把兩種時間混在同一欄','補底層aten列分類及中位數/mean/GPU同步；當下主文只找一種矩陣運算已足，不誤把選讀全names當必修。'),
'16.2':('不是服務TTFT','補MHA25位置含BOS/12固定不EOS停、時間含prefill，不代decode吞吐或語言品質。'),
'16.3':('不能只看解碼文字一樣就宣布序列相同','補兩模型各兩路原始ID比較與10^-5容差；數值非bit，主文有完整cache需要。'),
'16.4':('我們沒有重現它的轉換配方','補真MHA/MQA轉換初始化與cache省4倍，不保品質因果或所有GQA後端可用；主要head對應已在正文。'),
'16.5':('兩種裝填結果接近，不等於改後的學習任務仍保持問答品質。','補semantic packing對齊/改助手續寫任務後問答退步，不把無PAD例當省PAD成績；主文已明确程式只核mask。'),
'16.6':('沒有逐bit相同的承諾','補7/13真目標統計與40update精度/時間記憶代價，原階段規則不用選讀補。'),
'16.7':('不能預設等於200次更新','補AMP成功/跳過與FP32主權重、有限/NaN/inf界線，具體計時/峰值起點非全中間表省；主文GradScaler理由已教。'),
'16.8':('本次是節省中間資料記憶體的memory-efficient實作，沒有觀察到CUDA Flash','補完整模型vsFlash核的兩條證據與低精度容差，上游梯度比較角色自足；不是主文接口必要補洞。'),
'16.9':('不能只把新增32.75與2.032相比','Flash後端條件與FP16/bf16核心範圍及同65基線，補圖真正看了灰基線/彩增量/總峰值；未開原始實報。'),
'16.10':('也沒有dropout或跨裝置函式的核驗','補整block重算區與真40update數值/品質對齊，峰增量不是部署大小；主文提出dropout條件合理而未據此聲稱已驗。'),
'16.11':('不是零次或忘記算','補Inductor真圖真kernel但更慢、回本null的意思及子程序總成本；不需補主文編譯need。'),
'16.12':('並未重現該模型的設定或成績','Mistral來源与淘汰核心需要，主文window3定義自足。'),
'16.13':('不代表所有名稱含 chunk 的實作','原Flash/Transformers規則定位，簡化third表不普遍命名；主文同位3三法自足。'),
'16.14':('只比較兩邊第一名，不能代替這個程序','抽樣接受/校正分布為進階選讀，與正文greedy範圍相稱；MTP訓練和推測可分開角色足。'),
'17.1':('不能照抄這些bytes','補私有檔/randomstate/metadata與公開export大小另量，量化115200係數/其他/bias/scale加總詳細；原主文已區分數值與文件。'),
'17.7':('並不代表每個數字都只有四位元','補retainedfloat与biasbuffer實際口徑，主文weight-onlyFP32輸入輸出齊。'),
'17.8':('不是先用參數個數除以二推估的理想值','補13Linear各buffers與retainedparams合計，主文真pack同字數需求已教。'),
'17.9':('不能把那一題進步歸功於量化','補同120更新後FP32各直接PTQ與family切分，品質非量化extra訓練差。'),
'17.10':('本工具採更直白的逐層完整反量化，也沒有實作GPTQ的權重校正','补同L4示范后端时长、allocated/reserved/driver范围，普通完整还原不代表GPTQ；未验数字。'),
'17.11':('不能把那些GPU分數當作本節方法的成績','weight-only PTQ/QAT未做activation，補證據範圍；主文已有同限定。'),
'17.14':('普通FP32載入不會自動啟用fake quant','補同350update,39348targets与各QAT/PTQpacked對比、前向對齐非答对、float主權重檔非计算fake规则；主文QATneed完整。'),
'17.15':('不能替代JSON、日期澄清或安全拒絕的品質檢查','量化實報only合成屬性範圍及操作入口；主文behavior要求不以已有实测冒全部验。'),
'18.1':('沒有採用該論文的reverse KL或policy-gradient方法','原论文信号分类与teacher来源fingerprint；MiniLLM说明方法范围，正文18.8已定义forward/reverse方向，不必须选读全算法。'),
'18.2':('這些縮小先由架構決定','補actual64教师/16,32student，same宽三signals同架构起点；no偷偷接头，结构与signal仍分开。'),
'18.3':('這不是額外知識帶來的提升','翻译题答案仅核已正文关系；实际属性hard同真导致sameweights／style五错误仍留失败对照，EOS原ID不虚补，合理扩展而非必要规则错置。'),
'18.4':('沒有把標準答案換成教師自由續寫的前文','练习q-p核对及实际softcache已知标准前文，明示硬生成与soft264整列不同；主文设计理由已足。'),
'18.5':('沒有跑多個溫度挑最好成績','补高T同两方与T²梯度相对尺度、固定T2/alpha.5非最优及greedy最终生成；约定为何T²可进阶不影响已明确接口用法。'),
'18.6':('沒有拿其不同前文逐行硬配這份白盒快取','补sameByte264角色/EOS及softknown前文，与hard自由生成分离；正文对齐条件自足。'),
'18.7':('這比只看梯度None多核對了實際結果','实际3teacherfingerprints unchanged与讯号cache成本/版本，正文工具已teach基础，不替能力验证。'),
'18.9':('本輪沒有搜尋alpha或T','补固定0.5CE+0.5T²KL同标准前文、CE真值不teacher错代，未按test调整；主文契约明确。'),
'18.10':('不能稱為完整GSM8K benchmark','推理题无需速度回本已main可推；共同真值NLL/单seed任务和域外窗口选择不完整benchmark，人工表python只衍指标。'),
'18.11':('沒有測一般誠實或安全拒絕能力','补条件风格同算术多写非独立题，JSONint不是bool和原ID日期的有限模板；未冒一般安全。'),
'18.12':('能隔離教師訊號的是兩個讀相同資料的學生','actualMoEteacher读409而students32，same学生之间CE/KL可比较不把teacher差归signal；故事仍复词失败。'),
'18.13':('與玩具例的15和3不同','补真实角色问题/模态展开后25-28vs13-16与36-46vs24-34，same答案ID核；音调encoder先见非整路未见，合成非自然任务。toy机制自足。'),
'18.14':('不能把私有原檔當成最終發布大小','补fouractual版本最后PTQ no update，privatefile另metadata，完整反量化非低bitkernel／隔离RAM；未改变main品質成本判断。')
}
rows=[]
for pid in M['groups']['architecture']['pages']:
 raw=Path(P[pid]['snapshot']).read_text(); ds=re.findall(r'<details\b[^>]*>.*?</details>',raw,re.S)
 if ds:
  assert pid in E,pid
  quote,j=E[pid]; assert quote in '\n'.join(ds),(pid,quote)
  rows.append({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'details_count':len(ds),'actual_read':'全部折疊已實讀（在71頁main封存後）','quoted_basis':quote,'supplement_judgment':j,'essential_main_explanation_misplaced': '尺度need仍未補' if pid=='14.10' else '無已識別必要正文說明僅由選讀提供','unverified':'連結論文、原始結果JSON、設定/訓練實作未開；本輪不計數字或效果原始驗證'})
 else:
  assert pid not in E,pid
  rows.append({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'details_count':0,'actual_read':'helper確認本頁無折疊','supplement_judgment':'不需要補讀折疊'})
(B/'work/continuity-architecture/actual-extras-notes.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main_report_unchanged':True,'pages':rows},ensure_ascii=False,indent=2)+'\n')
print('actual supplements saved',len(rows),'with details',sum(r['details_count']>0 for r in rows))
