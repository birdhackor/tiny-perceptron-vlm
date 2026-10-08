import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/workspace/work/tutorial-audit-20261008')
R = B / 'reports'
W = B / 'work/continuity-integration'
S = B / 'freeze/sources'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(name):
    return json.loads((R / name).read_text())

def q(page, exact):
    p = S / (page + '.md')
    body = p.read_text()
    assert exact in body, (page, exact)
    return {'page_id': page, 'source_sha256': sha(p), 'line': body[:body.index(exact)].count('\n') + 1, 'quote': exact}

own = read('continuity-integration-initial.json')
ownseal = read('continuity-integration-seal.json')
mainseal = json.loads((W / 'main-notes-seal.json').read_text())
assert sha(R / 'continuity-integration-initial.json') == ownseal['sha256']
assert sha(R / 'continuity-integration-main-notes.json') == ownseal['main_notes_sha256'] == mainseal['sha256']
technicalseal = read('technical-integration-initial.seal.json')
addendumseal = read('technical-integration-u1-addendum.seal.json')
assert sha(R / 'technical-integration-initial.json') == technicalseal['report_sha256'] == addendumseal['initial_sha256_unchanged']
assert sha(R / 'technical-integration-u1-addendum.json') == addendumseal['addendum_sha256']
assert sha(R / 'technical-integration-initial.seal.json') == addendumseal['initial_seal_sha256_unchanged']
addendum = read('technical-integration-u1-addendum.json')
assert sha(addendum['new_execution']['raw_artifact']) == addendum['new_execution']['raw_sha256']

inputs = [
    'reader-integration-main.json', 'reader-integration.json',
    'technical-integration-initial.json', 'technical-integration-u1-addendum.json',
    'continuity-integration-main-notes.json', 'continuity-integration-initial.json',
]
input_records = [{'path': str(R / n), 'sha256': sha(R / n)} for n in inputs]
reader_main = read(inputs[0]); reader_final = read(inputs[1]); technical = read(inputs[2])
for a, b in zip(reader_main['pages'], reader_final['pages']):
    assert a['page_id'] == b['page_id']
    for k in a:
        assert a[k] == b[k], (a['page_id'], k)

# Extract only raw numbers and contracts. No historical audit judgment is used.
raw = {}
for v, expected in [('moe', '41c25f32fa57000ea845a5411e8799559da60b6d91993d0f923f309706049207'),
                    ('dense', '3136da84ca76ec686f374db426042faa6f7d171f7576ee1f54e9ea33aa8ae420')]:
    p = Path('/workspace/tiny-perceptron-vlm/docs/selftrained/results/public-raw') / v / 'test/metrics.json'
    assert sha(p) == expected
    d = json.loads(p.read_text())
    raw[v] = {'path': str(p), 'sha256': sha(p), 'checkpoint_sha256': d['checkpoint_sha256'],
              'controls': d['controls'], 'class_counts': {str(i): d['stratified_final_reply'][f'vision_clothing/query_class={i}']['count'] for i in range(3)}}
vision = Path('/workspace/tiny-perceptron-vlm/outputs/selftrained/data/vision-test.jsonl')
assert sha(vision) == '540441d5b7a49c2317d5f4432859451311a66959d92af16b4909b34665238b4c'
rows = [json.loads(s) for s in vision.read_text().splitlines()]
from collections import Counter
gold = Counter(r['messages'][-1]['content'] for r in rows if r['task'] == 'vision_clothing')
assert gold == {'這是褲子。': 120, '這是包。': 120, '這是短靴。': 120}
control_extract = {'kind': 'post-mutual read-only raw-source extraction; no new model execution', 'metrics': raw,
    'gold_distribution': {'path': str(vision), 'sha256': sha(vision), 'scope': 'JSON parse and exact saved gold string count for 360 vision_clothing records; no visual or semantic relabeling', 'counts': dict(gold),
        'derived_constant_bag_baseline': {'numerator': 120, 'denominator': 360, 'rate': 1/3, 'meaning': 'Always answering the one-class gold string can match only its 120 of 360 records; this is arithmetic from saved targets, not a new inference run.'}},
    'implementation_contracts': [
        {'path': str(B/'freeze/implementation/scripts/selftrained/evaluate.py'), 'sha256': sha(B/'freeze/implementation/scripts/selftrained/evaluate.py'), 'read_lines': '555–647', 'meaning': 'Pairs grouped by control_type/pair_id; all_correct checks semantic and no explicitly failed format for every member; output_changed compares normalized strings when gold strings differ. Change alone is not correctness.'},
        {'path': str(B/'freeze/implementation/scripts/selftrained/prepare_vision_ocr.py'), 'sha256': sha(B/'freeze/implementation/scripts/selftrained/prepare_vision_ocr.py'), 'read_lines': '190–289,326–376,600–718; FASHION_CLASSES line36', 'meaning': 'Swap keeps position question and source objects, swaps actual pixel placement; ROI keeps same rendered image and changes public frame, with distinct targets.'},
        {'path': str(B/'freeze/implementation/scripts/selftrained/prepare_text_tools.py'), 'sha256': sha(B/'freeze/implementation/scripts/selftrained/prepare_text_tools.py'), 'read_lines': '420–482', 'meaning': 'History and tool-return control grouping is explicit. Full member traces were not reread.'}],
    'limits': ['Only stored aggregate observations checked; no inference/training/rerun.', 'No independent re-scoring of paired semantic outputs.', 'Public test folder inspected contains aggregate/receipts, not per-member generated outputs; if a literal two-answer example is desired, retrieve matching raw generated rows plus saved golds and model/condition hashes. Do not manufacture examples.']}
extract_path = W / 'cross-control-source-extract.json'
extract_path.write_text(json.dumps(control_extract, ensure_ascii=False, indent=2) + '\n')

issues = []
issues.append({
    'id': 'X-INT-NAMES', 'origin_ids': ['INT-01'], 'origins': [{'report': 'reader-integration-main.json', 'id': 'INT-01', 'original_severity': 'optional', 'discovery': 'reader main pre-seal; actual jump route had not read the earlier first introductions'}],
    'pages': ['19.4'],
    'strongest_source_evidence': [q('7.1', '用正確助理示範繼續調整已有模型，叫**監督式微調**（Supervised Fine-Tuning，SFT）。'), q('11.12', '從圖片讀出字元稱**光學字元辨識**（Optical Character Recognition，OCR）。'), q('12.11', '自動語音辨識（Automatic Speech Recognition，ASR）主要把語音轉成文字。')],
    'actual_prerequisites': 'Cross phase fully read 7.1,11.12,12.11 including body and details. These are explicit earlier course introductions; merely being absent from the integration jump route is not a full-book omission. Integration 19.4 also supplies the corresponding stage/input roles.',
    'original_promise': 'Understand the actual weight chain and stage jobs; no new introduction of these three algorithms is promised at 19.4.',
    'completed': 'Names, Chinese meanings and basic roles exist in earlier main text; present stage table ties each to the current product.',
    'remaining_gap_or_counterevidence': 'The reported name-only gap is true for that particular reader route, but countered for sequential course reading by the three explicit main introductions. No missing mechanism follows from it.',
    'mechanism': {'judgment': 'sufficient for stage reading', 'basis': 'Supervised examples adjust next-token preferences; OCR image-to-characters; ASR audio-to-text; 19.4 is not deriving these internals.'},
    'need': {'judgment': 'sufficient', 'basis': 'Stage roles support tracking what is trained/frozen and whether the same core is carried forward; English expansion is not needed for that reasoning.'},
    'understanding_or_operation_benefit': 'Repeating full names may help isolated jump reading find terminology, but adds no missing relation for sequential readers.',
    'minimal_repair': 'None required. If jump-reading support is a specific later editorial goal, use a small reference to the established introductions rather than repeat a tutorial.',
    'added_burden': 'Three repeated long parentheses clutter a compact stage guide.',
    'reviewed_necessity': 'not a full-book defect; optional route aid', 'disposition': '保留原文', 'repair_scale': 'none; hypothetical route aid paragraph',
})
issues.append({
    'id': 'X-INT-JSON', 'origin_ids': ['INT-02'], 'origins': [{'report': 'reader-integration-main.json', 'id': 'INT-02', 'original_severity': 'optional', 'discovery': 'reader main pre-seal'}],
    'pages': ['19.7'], 'strongest_source_evidence': [q('7.13', 'JSON把欄位名稱與值保存為文字。'), q('19.7', '若模型把26抄成25，請求仍可能合法，工具也會正確計算25×16，整個任務卻已偏離原題。')],
    'actual_prerequisites': 'Cross fully read 7.13 and8.1; 7.13 main teaches field/value text, parsing and legal-format versus correct-answer distinction. Full-source search found no JavaScript Object Notation expansion in the frozen course; this search is not claimed as every unrelated page reread.',
    'original_promise': 'Explain tool request, execution, genuine return and final reply, including legal request versus correct parameter.',
    'completed': 'Concrete request fields and two generated stages explain the complete tool role; old main establishes JSON meaning.',
    'remaining_gap_or_counterevidence': 'Only conventional full-name mapping is absent. It does not explain the request mechanism and should not be sold as its repair.',
    'mechanism': {'judgment': 'sufficient', 'basis': 'Fixed tool/operation/integer fields are parsed, checked, executed, then fed back to the same core.'},
    'need': {'judgment': 'sufficient', 'basis': 'Program requires a bounded machine-readable request; actual format and role are visible without knowing the name origin.'},
    'understanding_or_operation_benefit': 'Helps locate the standard format by name without adding another algorithm.',
    'minimal_repair': 'Optional short expansion at the genuine earlier main introduction7.13: JSON（JavaScript Object Notation，一種以文字保存資料的格式）. Do not repeatedly expand it at19.7.',
    'added_burden': 'One parenthesis; no new prerequisite or JSON grammar lesson.', 'reviewed_necessity': 'optional', 'disposition': '採用（可選名稱補充）', 'repair_scale': 'paragraph',
})
issues.append({
    'id': 'X-INT-CONTROLS', 'origin_ids': ['INT-03'], 'origins': [{'report': 'reader-integration-main.json', 'id': 'INT-03', 'original_severity': 'burden', 'discovery': 'reader main pre-seal; remained after assigned details'}, {'report': 'continuity-integration-initial.json', 'id': None, 'original_severity': 'not reported', 'discovery': 'continuity acknowledged this source relation only after peer exposure during cross-review; not a new independent discovery'}],
    'pages': ['19.3', '19.6', '19.12'],
    'strongest_source_evidence': [q('19.3', '零家族交集排除了指定來源的跨份重複，不能單獨證明模型使用了材料。這些檢查會在感知和最終驗收時與模型回答一起看。'), q('19.6', '統一用[19.12](19.md#19.12)的留出題及成對控制核對。'), q('19.12', '新問法與未見組合的結果支持上述具體範圍，不能換成一般中文、任意照片或任意真人語音的宣稱。')],
    'actual_prerequisites': '19.3/main establishes family separation, label balance and paired dependency checks;19.6/main commits final paired verification;19.7/main explains separate altered-return control and normal genuine tool scoring. These were actually read before own initial seal;19.3/19.6/19.12 were targeted reread now. Necessary interpretation does not require experimental design beyond holding a question/image/history condition fixed and checking the corresponding changed target.',
    'original_promise': 'Do not infer material use from separation alone; review answer distribution and paired controls alongside perception and final answers, specifically at19.12.',
    'completed': '19.12 clearly reports finite heldout-task results, real chains, independent OCR head versus generated answer, original gates, overall failure and generalization limits. These remain strong valid conclusions.',
    'remaining_gap_or_counterevidence': 'Main and its tool-boundary details omit the promised balance baseline and paired observations. External raw links preserve recoverable evidence but require the reader to assemble condition/result mapping. Raw-source recovery confirms this is a presentation closure gap, not missing experimentation or false task scores.',
    'mechanism': {'judgment': 'needs local result-to-condition relation', 'basis': 'To see dependency, distinguish all members being correct from strings merely changing; task-level correct counts alone do not reveal the paired relationship.'},
    'need': {'judgment': 'necessary and supported by original source', 'basis': '19.3 expressly rejects zero-overlap as proof and promises these controls;19.6 names19.12 as verification. This need is not invented from available experiment outputs.'},
    'raw_evidence_checked': {'extract': str(extract_path), 'sha256': sha(extract_path), 'source_status': 'Stored observations exist; no rerun required.',
        'balance': 'Saved vision-test targets are120 pants,120 bags,120 boots: always answering bag can match120/360=1/3, not357/360 or356/360. This is saved-label arithmetic, not a new generated run.',
        'swap': 'Question fixed, same source objects swapped in actual pixel slots: all members correct MoE707/720, Dense706/720 pairs; normalized output changed on gold change715/720 and714/720. 720 counts pairs, not original images.',
        'roi': 'Same rendered image, changed public reading frame: all members correct86/150 and110/150 pairs; changed output150/150 for both. Changing every time does not mean reading every frame correctly.',
        'history_format': 'Saved grouping controls historical format: all members correct103/140 and113/140; changed138/140 and140/140. Full member conditions not reread, so bind a literal example to raw rows before publishing its exact narrative.'},
    'understanding_or_operation_benefit': 'Closes the original material/history-use check while preserving imperfect results and keeping paired counts separate from independent material counts and all-task acceptance.',
    'minimal_repair': 'Add one compact19.12 main table/paragraph: the one-class120/360 baseline, swap and same-image different-frame controls with what is fixed/changed, expected target difference, both-member correctness counts and changed-output counts; include one pertinent history-control summary if retaining the continuation/format verification promise. Link the matching existing raw metrics and briefly state that these finite controls do not prove a universally superior bridge or full acceptance.',
    'minimum_material_if_literal_example_requested': 'Retrieve saved pair_id member prompts, image/ROI/history conditions, gold strings, final generated outputs and weight/split hashes from the matching test trace. Aggregates already support a short summary; do not fabricate per-member outputs from them.',
    'added_burden': 'A short definition plus3–4 rows; no method chapter, fresh data or additional training. Avoid dumping all controls or claiming mere output change as accuracy.',
    'reviewed_necessity': 'required', 'severity': 'burden', 'disposition': '採用', 'repair_scale': 'paragraph (one compact local result block in19.12)',
})
issues.append({
    'id': 'X-INT-NFKC', 'origin_ids': ['INT-04', 'continuity-integration-NFKC-name'], 'origins': [{'report': 'reader-integration-main.json', 'id': 'INT-04', 'original_severity': 'optional', 'discovery': 'reader main pre-seal'}, {'report': 'continuity-integration-initial.json', 'id': 'continuity-integration-NFKC-name', 'original_severity': 'optional', 'discovery': 'continuity main before main-notes seal'}],
    'pages': ['20.10'], 'strongest_source_evidence': [q('20.10', 'NFKC可把全形Ａ換成A，並不把臺改成台。'), q('20.10', '核對前先約定哪些整理允許。')],
    'actual_prerequisites': 'Current page actually explains CER/exact, reference character unit, allowed normalization and raw-result preservation. Earlier11.16 was read by technical; own11.18/basic OCR prerequisites support transcription boundaries. No full Unicode course is required.',
    'original_promise': 'Measure original characters separately from meaning and use predeclared comparison rules.',
    'completed': 'Actual transformed and unchanged character examples explain the local rule and limit; CER and strict transcribing rationale work.',
    'remaining_gap_or_counterevidence': 'Missing Unicode standard-name expansion is a small lookup aid, not an unexplained scoring mechanism.',
    'mechanism': {'judgment': 'sufficient for current comparison', 'basis': 'Example explicitly gives fullwidth conversion and no臺/台 conversion; other whitespace/punctuation/simplification rules are independently declared.'},
    'need': {'judgment': 'sufficient', 'basis': 'Prior agreement plus raw preservation prevent post-hoc scoring changes that conceal transcription errors.'},
    'understanding_or_operation_benefit': 'Identifies the specific Unicode normalization standard used in policy or implementation.',
    'minimal_repair': 'At20.10 first occurrence add「Unicode兼容正規化（Normalization Form KC，NFKC）」; keep the existing example and boundary.',
    'added_burden': 'One short parenthesis; no list of Unicode transformations.', 'reviewed_necessity': 'optional', 'disposition': '採用（可選名稱補充）', 'repair_scale': 'paragraph', 'dedup_basis': 'Both reports identify exactly the missing full-name/Chinese category mapping, with the same nonblocking effect.',
})
issues.append({
    'id': 'X-INT-CTC', 'origin_ids': ['continuity-integration-CTC-main'], 'origins': [{'report': 'continuity-integration-initial.json', 'id': 'continuity-integration-CTC-main', 'original_severity': 'burden', 'discovery': 'after main-notes seal during independent assigned details read; initial main interface pass preserved'}],
    'pages': ['19.6', '19.12'],
    'strongest_source_evidence': [q('19.6', '訓練只給整串標準文字，不要求人工逐欄標出哪裡屬於哪個字。'), q('19.6', '這是入口的學習方法；接頭仍把特徵與整組字形分數交給共同語言核心，最後回答由核心生成。'), q('19.6', '合併相鄰重複後是 `大、〈空白〉、小`，再移除空白符號，得到 `大小`。'), q('19.12', '這裡的324／324是那個字串與標準答案完整相同的題數。')],
    'actual_prerequisites': '11.14/11.18 main actually read before seal: ordered visual features and complete-character generation;19.6 current main identifies32 fields versus whole-string labels and separate core generation. Assigned19.6 details subsequently read, including collapse, blank-separated genuine repeats, compatible-path training and independent greedy decode limit.',
    'original_promise': 'Explain how limited OCR material is learned at the input and then presented to the same core. Main names CTC as the input-learning method and states the whole-label interface; the dedicated details explicitly teach its internal alignment/collapse method.',
    'completed': 'The need is genuinely stated: many ordered image fields, a short known whole-string label, no per-field annotation. Input learning and final core generation are distinctly named and connected.19.12 explicitly defines which independent string output the324/324 belongs to.',
    'remaining_gap_or_counterevidence': 'Main does not contain the collapse/path relation. That absence is real, but current main neither asks the reader to derive a CTC alignment, compute its loss, nor infer equality between head strings and core strings. Distinguishing those outputs needs their identity and scoring rule, both supplied. The missing relation is taught immediately in a labeled optional method explanation; it is not silently counted as known for main reasoning.',
    'mechanism': {'judgment': 'sufficient at the main interface scope; internal CTC mechanism taught in optional route', 'basis': 'This specifically answers the claimed minimal missing relation: without collapse the reader cannot explain CTC internals, but can explain the stated encoder-label interface and understand why head and core final strings are different outputs. No main conclusion relies on the unprovided collapse/path property.'},
    'need': {'judgment': 'sufficient', 'basis': 'The current short whole-label/many-fields mismatch and avoided per-field annotation are explicitly supplied, independent of knowing CTC name or produced score.'},
    'understanding_or_operation_benefit': 'Moving two collapse/path sentences could make the local input-training mechanism more concrete for readers who want that route; main integration and final-score reading already succeed.',
    'minimal_repair': 'No required move. Retain dedicated adjacent details; if the curriculum later makes explaining/implementing CTC a main objective, then move the two-rule大小 example and compatible-path training relation to main at first use.',
    'added_burden': 'A required main move introduces a new blank/control-symbol convention and a second decoding route that present main operations do not use.',
    'reviewed_necessity': 'optional internal-method expansion, not required for the actual main promise', 'disposition': '保留原文', 'repair_scale': 'none; potential optional move paragraph',
    'change_from_initial': 'Own independent burden conclusion is reassessed after peer exposure and closer promise/dependency matching. Original initial/main bytes and discovery timing remain unchanged. This is not a vote and not a claim that CTC internals were already main-taught.',
})
issues.append({
    'id': 'X-INT-U1', 'origin_ids': ['U1'], 'origins': [{'report': 'technical-integration-initial.json', 'id': 'U1', 'original_severity': 'unverified', 'discovery': 'technical initial pre-seal'}, {'report': 'technical-integration-u1-addendum.json', 'id': 'U1', 'original_severity': 'supported_by_new_scoped_execution', 'discovery': 'authorized own-source post-seal addendum before cross exposure'}],
    'pages': ['19.6'], 'strongest_source_evidence': [q('19.6', '這個公開語音例的回答正確，獨立分類頭卻沒有選對類別，兩件事不能混稱通過。')],
    'actual_prerequisites': 'Same19.6 main explicitly separates the head score vector, projected features and generated final answer.18.1 and12.15 main were actually read in own initial: full candidate proportions and direct audio-feature route. No claim of causal soft-bridge superiority is needed.',
    'original_promise': 'This published example is a counterexample to equating argmax-head success with final-response success.',
    'completed': 'Technical addendum binds matching public modelSHA, recordingSHA and saved gold to a later scoped CPU observation: reference1; argmax2; generated answer matches saved public answer and EOS. I checked the raw artifactSHA and read its actual fields, without executing it.',
    'remaining_gap_or_counterevidence': 'Initial unknown remains an accurate historical status; its specific evidence gap now has a later recorded observation. The addendum does not explain which features caused the correct answer and is not a heldout/fulltest success.',
    'mechanism': {'judgment': 'sufficient for distinguishing outputs; causal superiority unverified and unclaimed', 'basis': 'Head chooses2 while the full score vector still includes reference1 at0.459061861 versus2 at0.490890861, and core answer is a different observed product. This does not identify why it was right.'},
    'need': {'judgment': 'sufficient', 'basis': 'Independent scoring is necessary because outputs answer different jobs; the paired head/final observation supports that limited distinction.'},
    'raw_evidence_checked': {'path': addendum['new_execution']['raw_artifact'], 'sha256': addendum['new_execution']['raw_sha256'], 'head_argmax': 2, 'reference_id': 1, 'head_probabilities': addendum['new_execution']['head']['probabilities'], 'answer': addendum['new_execution']['answer'], 'eos': True, 'dependency': 'Technical reviewer saved scoped execution; not my independent run.'},
    'understanding_or_operation_benefit': 'Supports retaining the example and the two separate scoring categories without replacing uncertainty with presumed success.',
    'minimal_repair': 'No curriculum change; archive the addendum as later technical evidence. Do not attach causal or general-ability claims.', 'added_burden': 'None in main.',
    'reviewed_necessity': 'specific initial uncertainty resolved by later scoped evidence; no curricular defect', 'disposition': '保留原文', 'repair_scale': 'none',
    'limits': addendum['limits'],
})

strong = []
extra_pass_bases = {
    '19.2': ('四expert全存与每位置两FFN属于容量/运算差别，不包办架构胜负。原句明列训练目标与选定历程不同。', '需要是定位实际自行训练成品及比较边界，不是证明MoE必胜。'),
    '19.4': ('父权重保存一条链；init重建状态，resume保存状态与原目标，总steps规则明确。', '同一核心接新材料且记录可追，而非把新目标误称精确续跑。'),
    '19.6': ('保留特征与整组分数经可调投影，既区别top类贴句，也区别头与最终生成。U1补件只补特定观察。', '共同核心还需读文字/历史条件；保留候选信息是明确作用，不靠单例证明soft优于hard。'),
    '19.7': ('26抄成25仍可合法真算，明确反例体现请求正确与可执行不同；真回填与手工字符串例分开。', '原工作是完整用户计算，不能以外部工具无错代替模型请求/读回可靠。'),
    '20.6': ('3×4原W12参数而rank2新增14；正文自己限制小例，只教独立修正与一次step，rank1才7。', '训练少量修正和保留固定底座的路径可懂；更新证据和用途改善仍分开。'),
    '20.8': ('两条语音路线各不能退步，正确稿4→2已单独违反；输出预算/EOS又独立说明。', '本版要保留语音客服而非盲取后步；不需完整aggregate权重才能判已失败的独立门槛。'),
    'natural-v4-data': ('代码版本/资料commit/manifest内容各绑定不同对象，完整指纹配对；不切本机代码已明说。', '拿匹配资料而不误切程序或由同文件名猜相同数据；命令/list/verify角色明确。'),
    'natural-v4-training': ('rolling completed_steps决定剩余更新，固定候选路径各自对应；验证三配置不是叠加LoRA。', '在真实保存状态基础上继续且比较明确候选，避免用rolling替最终选版。'),
}
for p in own['strong_pass_rechecks']:
    mech, need = extra_pass_bases[p['page_id']]
    strong.append({'page_id': p['page_id'], 'original_claim': p['claim'], 'quote': p['basis'], 'mechanism': {'judgment': 'sufficient', 'scope_basis': mech}, 'need': {'judgment': 'sufficient', 'scope_basis': need}, 'cross_result': 'retained scoped strong pass', 'limits': 'Existing source-reading and declared independent technical checks are used at their actual scope; no new run or blanket technical certification.'})
strong.append({'page_id': '19.12', 'original_claim': '有限验收与全部能力失败的诚实收束', 'quote': q('19.12', '高分任務不能抵銷另一項未達標。'),
    'mechanism': {'judgment': 'sufficient for task-level/chained scoring', 'scope_basis': 'Provided return does not include request; first wrong response fails continuation chain; CTC independent324 versus core252/281 identifies distinct outputs.'},
    'need': {'judgment': 'sufficient for gates but original dependency-check closure needs repair', 'scope_basis': 'Original gates and bounded conclusions are correct; X-INT-CONTROLS identifies only the unclosed paired/balance promise.'},
    'cross_result': 'partially retained; task/gate pass not a blanket page pass', 'counterexample_to_overexpansion': 'No whole chapter rewrite or full algorithm/experiment expansion follows from a compact missing control summary.'})

origin_ids = [o for i in issues for o in i['origin_ids']]
assert len(origin_ids) == 7 and len(set(origin_ids)) == 7
report = {
    'schema_version': 1, 'reviewer': '/root/continuity_integration', 'role': 'continuity cross-review', 'group': 'integration',
    'kind': 'post-mutual adjudication; not a second independent discovery', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'instructions': {'path': str(B/'cross-review-instructions.md'), 'sha256': sha(B/'cross-review-instructions.md')},
    'input_reports': input_records,
    'seal_checks': {'own_initial_match': True, 'own_main_notes_match': True, 'technical_initial_match': True, 'technical_addendum_match': True, 'technical_initial_seal_unchanged': True, 'reader_final_preserves_all_main_page_fields': True},
    'reading_scope': {'reports': 'All reported issues/origin IDs and final supplement dispositions, all30 reader page-end four-question/result summaries, all30 technical per-page judgments/quoted checks/unverified scopes, own initial all issues and8strong passes, full technical U1addendum and raw saved artifact. Relevant reader first checkpoints19.4/19.6/19.7/19.12/20.10 read; unrelated raw checkpoints not reread or claimed as fully read.',
        'source': 'Original full integration30-main plus13details reading remains in sealed initial. This cross did targeted19.3/19.6/19.12 main rereads, complete7.1/7.13/8.1/11.12/12.11 prerequisite pages, targeted earlier-name search and raw control source extraction. No full30 reread.',
        'new_visual_scope': 'No disputed new figure/position issue; initial32figure renders and8context images remain the actual visual scope. No additional view or interaction claimed.',
        'execution': 'No model rerun, training, downloading, deployment or external communication. JSON/hash/label counting is read-only checking, not experimental observation.'},
    'issues_dispositions': issues, 'strong_pass_rechecks': strong,
    'deduplication': {'reported_origin_issue_ids': 7, 'unique_relationships_after_dedup': 6, 'merged': [{'relationship': 'NFKC name mapping', 'origin_ids': ['INT-04', 'continuity-integration-NFKC-name']}],
        'required_relationships': 1, 'optional_adopted_relationships': 2, 'optional_retained_relationships': 1, 'earlier_taught_route_aid_retained': 1, 'scoped_unknown_resolved_without_curricular_repair': 1, 'unresolved_curricular_issue_candidates': 0},
    'unknown_scopes': [
        'Raw pair member generated strings/conditions and semantic judgments were not independently rescored. Stored aggregate counts and grouping contract checked; literal narrative example requires exact matching raw member records.',
        'Whole UI interaction, all30 page layouts, live downloads/install/resume and general mature-model execution remain unverified; declared initial figure/position sampling is retained.',
        'U1 conclusion depends on a saved technical scoped run under torch2.14.1+cpu, not a new independent run or originaltorch2.8 rerun; no new heldout/continuation/full-test claim.',
        'No general soft-bridge causality, MoE architecture superiority or speaker-isolated generalization was tested or inferred. The source does not need those stronger claims for the retained relation.'
    ],
    'rewrite_assessment': {'required_scale': 'paragraph', 'required_location': '19.12 compact control/baseline result block', 'page_rewrite_required': False, 'chapter_rewrite_required': False, 'chapter_order_change_required': False, 'optional_scale': 'paragraph name expansions at7.13 and20.10', 'ctc_main_move_required': False},
    'preservation': 'Initial reports, main notes, tutorial and technical addendum unchanged; source oversight acknowledged after exposure and own CTC necessity reassessment remain visibly post-mutual.'
}
path = R / 'cross-integration.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for p in input_records:
    assert sha(p['path']) == p['sha256']
seal = {'schema_version': 1, 'reviewer': '/root/continuity_integration', 'group': 'integration', 'report_path': str(path), 'report_sha256': sha(path), 'sealed_at_utc': datetime.now(timezone.utc).isoformat(), 'input_reports': input_records, 'own_initial_sha256_unchanged': ownseal['sha256'], 'own_main_notes_sha256_unchanged': mainseal['sha256'], 'source_extract_path': str(extract_path), 'source_extract_sha256': sha(extract_path), 'dedup_counts': report['deduplication'], 'rewrite_assessment': report['rewrite_assessment']}
(R / 'cross-integration-seal.json').write_text(json.dumps(seal, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(path), 'sha256': sha(path), 'seal': str(R/'cross-integration-seal.json'), 'counts': report['deduplication'], 'rewrite': report['rewrite_assessment']}, ensure_ascii=False, indent=2))
