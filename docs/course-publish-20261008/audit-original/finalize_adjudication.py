import collections
import datetime
import hashlib
import json
from pathlib import Path

B = Path('/workspace/work/tutorial-audit-20261008')

def read(p):
    return json.loads((B / p).read_text())

def sha(p):
    return hashlib.sha256((B / p).read_bytes()).hexdigest()

# Explicit author judgments, including the cost of changing otherwise adequate text.
# Full source quotations and differing original judgments remain in the immutable
# reports embedded below; this registry does not replace those reports.
reasons = {
    'X-F-4.4-design-purpose': '採用局部用途澄清。2.3的多套配方、2.4的非線性及4.2的同形殘差已足以重建操作，所以不是缺整個FFN概念；仍宜把擴張的加工分工和示範採GELU的約定直接說清。兩句即可，不宣稱四倍寬度或GELU必優。',
    'X-F-glossary-acronym-expansion': '保留查找入口。现有中文作用和完整章節連結已完成詞表任務；在真正首次介紹處補少數缺名比給整張手機表加長英文更有收益。詞表未列全名不列為全書概念缺漏。',
    'X-F-2.5-chart-mobile': '採用可選圖版調整。正文精確表和趨勢說明已完整，窄螢幕補圖字小只影響舒適度；維持同一數據及限制，不新增訓練或搬入主文。',
    'X-F-W1-floating-menu': '延後UI潤飾。實際普通捲動180px可取得完整uv/Git段，推翻不可讀的外推；局部蓋字仍是真實干擾，但縮小按鈕會影響導航，宜集中評估。',
    'X-F-entry-overview': '保留入口。課程入口和2/3/4章轉折已教限制與新增工作；再加總演進段落容易重複，缺史年或模型譜系不是本教材問題。',
    'cross-learning-5.5-motive': '採用一小段選用目的。乘法回拉、m/v分離、零與None及驗證選強度都已教；缺口是為何在任務代價外另加小權重偏好。應明示為可關閉且需驗證的候選設計，避免必然泛化改善的結論。',
    'cross-learning-5.6-warmup': '採用可選一兩句起步位移說明。1.12已使小步需要可正常推論，故不升必要；不得新增早期梯度必定不穩的未證前提。',
    'cross-learning-6.5-nll-name': '採用首次短括註。數學、中文動作和用途俱已教；補Negative Log-Likelihood只改善對外名稱對照。',
    'cross-learning-7.13-json-name': '採用7.13一次短括註，合併19.7和B.1的同名建議。格式、解析與正確性區別已教；不在每個後續使用處重複長名稱。',
    'cross-learning-5.4-adam-name': '採用首次短括註Adaptive Moment Estimation。機制已完成，名稱有查論文和工具的收益，不列必要算法補課。',
    'cross-learning-7.5-duplicate': '採用刪重。保留程式前retain_grad首次說明即可，程式後直接接logits與Q梯度區別；減少閱讀負擔而非擴充。',
    'cross-learning-7.5-link-scope': '採用修正文案或刪輔助連結。1.11未教承諾的葉與中間節點區別，但本頁已自足，屬可選查找精度。',
    'C-M-resample-action': '採用補最低動作。把含糊的控制改為降取樣前減弱/濾除新界線以上成分，連到已教混疊；限定升取樣的不同情況，不要求濾波器設計。',
    'C-M-mel-adoption': '採用一兩句實際表示選擇。加權和、非均勻配置及不可逆都已教；需把後續實際採用的合差、低密高疏接回作者目的。允許據實標示教學約定，不要求新性能實驗。',
    'C-M-contrastive-scope': '採用可選短實驗身份。六種RGB×形狀和留出位置可降低讀者回找成本，但當頁運算已足夠；限制範圍比加長完整實驗卡更合適。',
    'C-M-projector-scope': '保留。正文已分出通路與失敗結果，完整實驗可由折疊定位；此處重貼卡片沒有新的必要理解收益。',
    'C-M-vqa-shared-scope': '採用一次共用短摘要及連結，合併11.5/11.6。標出已知顏色形狀、新位置、真訓練及凍結階段，避免把小例誤讀為任意照片能力。',
    'C-M-padding-wording': '採用一詞精確化：補長用的PAD位置。正常答案空白仍計分，現文已有補長限定，故只是可選去歧義。',
    'cross-architecture-yarn-scale-purpose': '採用局部第二步需要。分頻保近遠與尺度改集中度均已教；延窗後為何另外選這項尺度仍需連到作者實際比較/驗收目的。不可把可調集中度本身當獨立需要或保證較正確。',
    'cross-architecture-top2-choice-purpose': '採用局部選擇理由。兩合併规则和top-1梯度保留理由已足；正文承諾選用原因，應說清選中者相對分工的含義或示範約定，不虛構品質優勢。',
    'cross-architecture-14.1-name': '採用短中文對名旋轉位置嵌入，連到已學embedding；作用已教，屬查找潤飾。',
    'cross-architecture-14.2-name': '採用短中文對名均方根正規化，區分RMS量與整層名稱；無新算法。',
    'cross-architecture-14.3-name': '採用L2正規化短對名，與同段實際向量動作相連；不是缺用途。',
    'cross-architecture-14.4-mlp-name': '保留。2.5已完整介紹MLP，正常前文承接足夠。',
    'cross-architecture-14.4-relu-name': '保留。2.4已完整介紹ReLU，重複英文不增理解。',
    'cross-architecture-15.3-name': '保留直接的連續權重混合說明；再添軟路由術語無足夠本輪收益。',
    'cross-architecture-15.4-name': '保留只留比例最大k位的直接規則；新增譯名不比現文清楚。',
    'cross-architecture-15.6-names': '保留。角色已清楚，統一括號外觀不是概念修補。',
    'cross-architecture-15.8-scope-wording': '採用固定排名造成工作集中的精確措辭；本例未展示完整學習反馈形成過程，無需新實驗。',
    'cross-architecture-15.10-name': '保留。5.9/16.1已有FP32完整名稱與位元意義。',
    'cross-architecture-16.2-names': '延後全書術語統一。整段提示/新token的動作已清楚，解碼新譯名可能與ID解碼混淆。',
    'cross-architecture-16.4-name': '採用MQA多查詢注意力短對名，和同組GQA中文配對；保留多Q的原圖及作用。',
    'cross-architecture-16.7-float-names': '採用實際API別名float16/bfloat16對照，減少填dtype時猜測；不補Brain命名史。',
    'cross-architecture-16.7-gradscaler-name': '採用梯度縮放器短對名；當頁尺度管理已教清，不加入新背景。',
    'cross-architecture-17.3-name': '採用mean absolute error與abs/mean的名稱對照；不可把MAE誤作任務品質判準。',
    'cross-architecture-17.14-name': '採用straight-through estimator短對名，保留近似非round真導數的限制。',
    'cross-architecture-18.1-name': '保留。7.1已有SFT完整介紹。',
    'cross-architecture-18.8-name': '保留使用已知前文的直接中文；另添教師強制易與蒸餾教師混淆，不要求統一名。',
    'X-INT-NAMES': '保留。7.1、11.12、12.11已有SFT/OCR/ASR完整介紹，單一跳讀路徑沒讀過不等於全書漏教。',
    'X-INT-JSON': '與7.13的首次名稱建議合併，不在19.7重複括註。',
    'X-INT-CONTROLS': '採用19.12短表。19.3/19.6明確承諾配對控制和分布檢查，既存原始結果可補閉合；區分兩成員都正確與只改回答、pair與image分母。无需新訓練。',
    'X-INT-NFKC': '採用一次標準名稱短對照。現有全半形、臺台反例及預先宣告規則已足以理解計分，屬可選查找。',
    'X-INT-CTC': '保留相鄰選讀。主文已教32欄/整串標籤、免逐欄對齊、獨立頭與core字串分別計分；主文推論不依賴collapse/path細節。未要求讀者實作CTC時移整套算法會新增控制符號與解碼路線負擔。',
    'X-INT-U1': '原unknown保留在初稿，後續限定CPU觀測已核同公開bytes的head錯而生成答對。當頁可保留，不上升為因果優勢、全卷通過或新留出證據。',
    'CXE-01': '保留A.1。SFT已在7.1介紹，本頁更新與查找差別自足。',
    'CXE-02': '採用RAG第一次完整英名，外部檢索有收益；現有中文機制已足。',
    'CXE-03': '併入7.13一次JSON全名，B.1保留實際格式角色。',
    'CXE-04': '採用B.6短位元組說明及6.1定位，照顧附錄跳讀；全書6.1/6.8已教，不作必要概念漏教。',
    'CXE-05': '保留C.5。16.2已教KV角色，本页重述不增等待範圍理解。',
    'CXE-06': '採用一次RLVR完整英名，三份同源建議合併；角色與更新已教。',
    'CXE-07': '採用把固定兩分數比較與抽樣方差用途分清；保留參考非正解，完整抽樣机制由13.13定位，不宣稱小例已證降低方差。',
    'CXE-08': '保留T.3。MLP已有2.5完整介紹。',
    'CXE-09': '保留T.7名稱。DPO已有7.18/13.4完整介紹。',
    'CXE-10': '採用輔助連結13.1改13.4，使固定參考角色的承諾可找到；本地文字自足故非必要。',
    'CXE-11': '保留T.10的KL短名稱及18.8定位，前文已有完整定義。',
    'CXE-12': '採用所選完整教師路線的必要前置定位。標出三model.pt、dataset.json與必須讀的固定配方折疊；不將所有訓練歷史搬入主文，避免把本機小練習checkpoint誤當完整教師。',
    'CXE-13': '採用公開safetensors推論權重的精確名，避免safe形容混淆；不加該格式只能存模型張量的錯誤限制。',
    'CXE-14': '採用環境表簡短ASR語音辨識/OCR讀字或正確首次章節定位，照顧直接選環境者；全書已有完整定義。',
    'CXE-15': '保留Git LFS產品名與pointer/實體/官方入口，英文拆字不增目前操作收益。',
    'CXE-16': '採用但只算獨立課綱設計參考路線的必要定位。循序教材6.5已完整教BPB，不列核心漏教；表內加共同原文byte分母及6.5連結即可完成此參考頁自己的比較理由。',
    'CXE-17': '保留training的PPO入口。7.18/13.13已介紹，折疊不需重複。',
    'CXE-18': '保留SDPA名稱與16.8後端定位；已有完整名稱，介面不等实际Flash的限制清楚。',
    'CXE-19': '保留QAT與17.14定位，前文已教完整名和用途。',
    'CXE-20': '採用證據記錄勘誤而非重跑模型。36/56 generation metadata引用後續變動messages，原input_ids逐ID仍正確；未來存當時deepcopy，保留舊JSON/SHA，附重建快照與頁面勘誤。',
    'CXE-21': '採用C.7必要安裝連結修正。T.1無Git LFS安裝入口且fetcher遇pointer確需CLI；改真安裝來源即可，未要求改章節。',
}

necessary = {
    'X-F-4.4-design-purpose': ('N01', '4.4', '核心正文', 'FFN擴張及示範GELU的選用目的'),
    'cross-learning-5.5-motive': ('N02', '5.5', '核心正文', 'AdamW額外小權重偏好的採用目的'),
    'C-M-resample-action': ('N03', '12.2', '核心正文', '降取樣前減弱/濾除超界成分的具體動作'),
    'C-M-mel-adoption': ('N04', '12.6', '核心正文', 'mel頻帶合併及非均勻配置的選用目的'),
    'cross-architecture-yarn-scale-purpose': ('N05', '14.10', '核心正文', 'YaRN尺度調整的獨立需要'),
    'cross-architecture-top2-choice-purpose': ('N06', '15.5/15.7', '核心正文', 'top-2選用重新正規化的理由'),
    'X-INT-CONTROLS': ('N07', '19.12', '核心正文', '已承諾的配對控制與答案分布結果摘要'),
    'CXE-12': ('N08', 'T.10', '操作與參考入口', '完整教師checkpoint與必要配方定位'),
    'CXE-21': ('N09', 'C.7選讀重做路線', '操作與參考入口', 'Git LFS安裝來源錯指T.1'),
    'CXE-16': ('N10', 'curriculum獨立設計參考', '操作與參考入口', 'BPB共同單位及已有6.5說明的定位'),
    'CXE-20': ('N11', 'B.3/B.4/T及tools trace', '證據記錄', '歷史逐步對話metadata快照勘誤'),
}
keeps = {
    'X-F-glossary-acronym-expansion', 'X-F-entry-overview', 'C-M-projector-scope',
    'cross-architecture-14.4-mlp-name', 'cross-architecture-14.4-relu-name',
    'cross-architecture-15.3-name', 'cross-architecture-15.4-name',
    'cross-architecture-15.6-names', 'cross-architecture-15.10-name',
    'cross-architecture-18.1-name', 'cross-architecture-18.8-name',
    'X-INT-NAMES', 'X-INT-CTC', 'X-INT-U1', 'CXE-01', 'CXE-05',
    'CXE-08', 'CXE-09', 'CXE-11', 'CXE-15', 'CXE-17', 'CXE-18', 'CXE-19',
}
defer = {'X-F-W1-floating-menu', 'cross-architecture-16.2-names'}
completed = {
    'cross-architecture-14.4-mlp-name', 'cross-architecture-14.4-relu-name',
    'cross-architecture-15.6-names', 'cross-architecture-15.10-name',
    'cross-architecture-18.1-name', 'cross-architecture-18.8-name', 'X-INT-NAMES',
    'CXE-01', 'CXE-05', 'CXE-08', 'CXE-09', 'CXE-11', 'CXE-15', 'CXE-17', 'CXE-18', 'CXE-19',
}

records = []
cross_refs = []
for group in ['foundations', 'learning', 'modalities', 'architecture', 'integration', 'extensions']:
    p = f'reports/cross-{group}.json'
    d = read(p)
    cross_refs.append({'path': p, 'sha256': sha(p)})
    for case in d.get('issues', d.get('issues_dispositions', [])):
        cid = case.get('id', case.get('cross_id'))
        assert cid in reasons, cid
        grade = ('necessary' if cid in necessary else
                 'resolved_unknown' if cid == 'X-INT-U1' else
                 'not_a_current_gap' if cid in completed else 'optional')
        disposition = 'retain' if cid in keeps else 'defer' if cid in defer else 'adopt'
        records.append({
            'id': cid, 'group': group, 'origin_ids': list(case['origin_ids']),
            'root_grade': grade, 'root_disposition': disposition,
            'root_reason': reasons[cid],
            'necessary_item': dict(zip(['id', 'location', 'category', 'problem'], necessary[cid])) if cid in necessary else None,
            'source_cross_report': {'path': p, 'sha256': sha(p)},
            'immutable_cross_assessment': case,
        })

# A single change at the genuine first introduction resolves all three JSON
# routes. The different original dispositions remain visible in merge evidence.
primary = next(x for x in records if x['id'] == 'cross-learning-7.13-json-name')
primary['merged_related_cross_assessments'] = []
for cid in ['X-INT-JSON', 'CXE-03']:
    other = next(x for x in records if x['id'] == cid)
    primary['origin_ids'].extend(other['origin_ids'])
    primary['merged_related_cross_assessments'].append(other)
    records.remove(other)

history = read('synthesis/historical-comparison.json')
records.extend([
    {'id': 'HIST-9.6-variable', 'group': 'historical_prompted_followup',
     'origin_ids': ['HIST-9.6-variable'], 'root_grade': 'optional', 'root_disposition': 'adopt',
     'root_reason': '主文及輸出已明說不拒答不等於完成，故非概念錯誤。把completion_rate的三處變數名改non_refusal_rate，讓程式名吻合已教的計分邊界。',
     'provenance': 'Only discovered/rechecked after all cross seals and historical comparison; not an independent fresh-reader discovery.',
     'source_evidence': history['relations'][6]},
    {'id': 'HIST-9.8-mobile-wrap', 'group': 'historical_prompted_followup',
     'origin_ids': ['HIST-9.8-mobile-wrap'], 'root_grade': 'optional', 'root_disposition': 'defer',
     'root_reason': '實看390px已開第二張折疊表，color/for/answer有字中斷行。主文任務仍清楚；跟手機表格樣式一起評估按單詞/分隔符換行或表內橫捲，不因選讀表顯示潤飾重寫內容。',
     'provenance': 'Historical prompt after cross seals; first misplaced screenshot retained and explicitly excluded from target verification.',
     'source_evidence': history['relations'][7], 'visual_evidence': history['historical_visual_followup']},
])

originals = read('synthesis/all-three-round-candidates.json')['candidates']
expected = {x['id'] for x in originals} | {'F-2.5-supplemental-chart-mobile', 'ROOT-entry-overview', 'HIST-9.6-variable', 'HIST-9.8-mobile-wrap'}
coverage = collections.Counter(y for x in records for y in x['origin_ids'])
assert set(coverage) == expected, (sorted(expected-set(coverage)), sorted(set(coverage)-expected))
assert all(n == 1 for n in coverage.values()), coverage
assert len(records) == 65
assert sum(x['root_grade'] == 'necessary' for x in records) == 11
assert sum(x['root_grade'] == 'optional' and x['root_disposition'] == 'adopt' for x in records) == 28
assert sum(x['root_disposition'] == 'defer' for x in records) == 3

result = {
    'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'review_only': True, 'curriculum_changes_performed': False,
    'source_scope': {'canonical_pages': 325, 'lessons': 287, 'intro_pages': 23, 'guides': 7, 'references': 7, 'home': 1},
    'method': 'Three globally ordered source rounds, six reviewers per round, immutable pre-peer source judgments; cross reconciliation only after all source seals, historical comparison only after all cross seals. Grouped routes and recorded prerequisites, not one human reading every page sequentially.',
    'counting': 'Count distinct repair relations, not reviewer mentions; merge shared JSON name repair at7.13. 72 initial raw origins plus2 root-prompted and2 historical-prompted followups =76 origins,65 final relations.',
    'counts': {
        'raw_origins': len(expected), 'final_relations': len(records),
        'necessary_repairs': 11, 'necessary_categories': dict(collections.Counter(x['necessary_item']['category'] for x in records if x.get('necessary_item'))),
        'optional_adopted': 28, 'deferred_optional': 3, 'retained': 23,
        'whole_chapter_rewrites': 0, 'course_order_rewrites': 0,
    },
    'recommendation': '保留全書架構及既有小例，集中完成11項必要局部修補，再選擇處理有具體收益的28項潤飾。沒有觀察到需要整章或全書重寫的問題。',
    'limits': ['AI reviewer judgments do not replace observed human novice learning.', 'No full training, all-kernel rerun or publishing acceptance performed.', 'Image/interaction checks are scoped; not every visual element received human-like exhaustive inspection, and the external20.4 thumbnail was not fetched.', 'Current necessary items are diagnosed, not already fixed.', 'Old active acceptance remains on its old criteria; updated criteria drift was not bypassed.'],
    'source_cross_reports': cross_refs, 'origin_records': originals, 'decisions': records,
}
(B/'synthesis/root-adjudication.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts': result['counts'], 'necessary': sorted([x['necessary_item'] for x in records if x.get('necessary_item')],key=lambda x:x['id'])},ensure_ascii=False,indent=2))
