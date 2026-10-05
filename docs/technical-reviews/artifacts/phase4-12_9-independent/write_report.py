"""Write this reviewer's complete new report without reading an old canonical report."""
import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
REL = ART.relative_to(ROOT).as_posix()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
env = {"python": "3.13.5", "torch": "2.14.1+cpu", "device": "CPU", "cuda": "None"}
commands = {
    "original-cpu/execution.json": (".venv/bin/python docs/review-tools/section_facts.py course/chapters/12.md#12.9 --output /tmp/factual-12_9-independent-cpu --execute --timeout 30", "exit 0; original fence prints (1,19,264) and projector gradient True; no guard event", env),
    "bounded-cpu.json": ("CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-12_9-independent/check_cpu.py > docs/technical-reviews/artifacts/phase4-12_9-independent/bounded-cpu.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_9-independent/bounded-cpu.stderr.txt", "exit 0; 440/high and 200/low shape/target/CE/gradient/causal/no-weight-update assertions pass", env),
    "raw-reaggregation.json": (".venv/bin/python docs/technical-reviews/artifacts/phase4-12_9-independent/check_raw.py > docs/technical-reviews/artifacts/phase4-12_9-independent/raw-reaggregation.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_9-independent/raw-reaggregation.stderr.txt", "exit 0; existing generated_ids exact-match reaggregation 11/14, EOS 14/14, target tokens 62; no model evaluation", {"python": "3.13.5", "device": "CPU, JSON-only"}),
    "render-receipt.json": (".venv/bin/python docs/technical-reviews/artifacts/phase4-12_9-independent/render_page.py > docs/technical-reviews/artifacts/phase4-12_9-independent/render.stdout.txt 2> docs/technical-reviews/artifacts/phase4-12_9-independent/render.stderr.txt", "exit 0 on corrected render; current page matches original code and was viewed at desktop/mobile; prerequisite SVGs rendered/viewed; first file URL blocked attempt preserved", {"python": "3.13.5", "device": "CPU", "browser": "Chromium 151.0.7922.173 /usr/bin/chromium --no-sandbox"}),
}
artifact_ids = {}
artifacts = []
for p in sorted(ART.rglob("*")):
    if not p.is_file() or p.name.startswith(("checker.", "report-generation.")) or p.suffix == ".pyc":
        continue
    assert p.suffix != ".pt"
    local = p.relative_to(ART).as_posix()
    identifier = "a-" + local.replace("/", "-").replace(".", "-")
    artifact_ids[local] = identifier
    item = {"id": identifier, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p),
        "kind": "code" if p.suffix == ".py" else "figure_render" if p.suffix == ".png" else "derivation" if local == "inspection.md" else "source_snapshot",
        "description": "This reviewer's retained " + local + "; original bytes, actual command output or personally inspected evidence as identified in inspection.md."}
    if local in commands:
        command, result, environment = commands[local]
        item.update(kind="execution", command=command, result=result, environment=environment)
    artifacts.append(item)
A = lambda *names: [artifact_ids[n] for n in names]
def repo(identifier, title, path, read):
    return {"id": identifier, "kind": "repository_code", "title": title, "verified": True,
        "path": path, "sha256": sha(ROOT / path), "version": "Exact current SHA; matches audio run code_sha256 where applicable",
        "inspection_note": read}
sources = [
    repo("repo-data", "ByteTokenizer special IDs and UTF-8 byte IDs", "tiny_perceptron/data.py", "AST then lines1-28; vocab264, eight special IDs, answer UTF-8 bytes+8."),
    repo("repo-model", "TinyLM embeddings/logits and masked_loss", "tiny_perceptron/model.py", "AST then lines1-105; model width/vocab and masked_loss sum divided by nonignored count."),
    repo("repo-modal", "Tone, audio encoder, modality expansion and autoregressive generation", "tiny_perceptron/multimodal.py", "AST then lines1-174; tone/defaults, log_mel, AudioEncoder, expansion and one shift, forward/projector and generate_modal contract."),
    repo("repo-attention", "Causal attention mask", "tiny_perceptron/attention.py", "AST then lines1-74; keys<=queries mask and causal forward actually used by ModelConfig(width=8)."),
    repo("repo-recipe", "Existing audio experiment computation and exact-match criterion", "scripts/course_experiments/modalities.py", "AST field/function ranges first; read27-53,73-133,195-283,287-374,377-384,884-908. Did not read run_audio result interpretation scope value at909. Dataset pairs, splits, optimizer, generation and EOS criterion inspected without executing recipe."),
]
commit = "5c4886908584029761b579af026dcfb627c84070"
for identifier, original, read in [
    ("torch-backward", "torch/_tensor.py", "Tensor.backward lines566-625; computes chain-rule leaf gradients and accumulates grad, not optimizer updates."),
    ("torch-loss", "torch/nn/modules/loss.py", "CrossEntropyLoss lines1200-1307; class-index formula, ignore_index factor and nonignored target mean denominator."),
    ("torch-linear", "torch/nn/modules/linear.py", "Linear lines53-140; affine x A^T+b, input/output last-axis widths and forward F.linear."),
]:
    sources.append({"id": identifier, "kind": "official_source", "title": "PyTorch " + original,
        "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{original}",
        "version": "PyTorch installed2.14.1+cpu; immutable upstream git " + commit,
        "authority_reason": "Original implementation and API contract in the official pytorch/pytorch repository; immutable commit equals installed torch.version.git_version.",
        "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
        "inspection_note": read + " Personally read exact original snapshot, re-fetched its HTTPS URL with TLS verification, HTTP200 and identical SHA; see sources/https-receipt.json."})
sources += [
    {"id": "scheduled-paper", "kind": "paper", "title": "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks, Bengio et al.",
        "url": "https://arxiv.org/pdf/1506.03099v3", "version": "arXiv1506.03099v3,23 Sep2015",
        "authority_reason": "Authors' original Google Research paper; personally verified title/authors and explicit arXiv version/date on the original PDF first page.",
        "accessed_on": "2026-10-05", "verified": True, "checked_original": True,
        "inspection_note": "Read PDF abstract/Introduction,Sec2.1 eq1-3,Sec2.2/2.3 via preserved pdftotext lines1-145; supports true prior target at training vs generated prior token at inference. RNN paper supports this conditioning contrast, not the repository's tone score or any recommendation of scheduled sampling."},
    {"id": "exec-original", "kind": "execution", "title": "Original 12.9 CPU fence execution", "verified": True, "artifact_id": artifact_ids["original-cpu/execution.json"]},
    {"id": "exec-bounded", "kind": "execution", "title": "Short random-weight alignment/gradient/200Hz checks", "verified": True, "artifact_id": artifact_ids["bounded-cpu.json"]},
    {"id": "exec-raw", "kind": "execution", "title": "JSON-only reaggregation of existing raw audio samples", "verified": True, "artifact_id": artifact_ids["raw-reaggregation.json"]},
]
E = lambda source, loc, supports: {"source_id": source, "locator": loc, "supports": supports}
V = lambda expected, observed, details, **extra: {"method": "executed", "expected": expected, "observed": observed, "details": details, **extra}
denominators = {"training_records":16,"validation_records":8,"test_records":14,"test_frequency_families":7,
    "conditions_per_frequency":2,"test_target_tokens_with_eos":62,"historical_training_steps":300,"historical_effective_targets":5409}
claims = [
    {"id":"audio-entry","kind":"software","statement":"440 Hz is high by the >300 Hz task rule; the audio marker supplies a position, and eleven audio feature vectors are projected into the language width without containing a prewritten high target.",
        "location":"course/chapters/12.md:267-289","status":"verified","scope":"This repository's artificial-tone task and randomly initialized MultiModalLM input path; not learned frequency recognition.",
        "evidence":[E("repo-data","ByteTokenizer lines10-21","audio_id6 is an input marker; high is four separate byte IDs."),E("repo-modal","tone44-46;AudioEncoder72-81;MultiModalLM104-135;expand84-100","1600 waveform samples,11 frame features,projector16→8,marker replaced by features and not target text."),E("torch-linear","Linear53-75,130-135","Affine last-axis width transformation preserves time/batch axes."),E("exec-bounded","bounded-cpu.json /cases/0","Actual 440/high feature/projection shapes and original IDs.")],
        "artifact_ids":A("bounded-cpu.json","original-cpu/execution.json","extracted/fence-1.py"),
        "verification":V("Marker expands to11 projected vectors of width8;440>300→high.","Audio feature[1,11,16],projected[1,11,8];prefix[1,3,6,2,4].","Original fence run plus bounded instrumentation;ByteTokenizer uses high bytes112,113,111,112,distinct from audio marker6.")},
    {"id":"lengths-targets","kind":"numeric","statement":"Five prefix positions plus high's four bytes and EOS give10 positions; replacing one marker with11 makes20, one shift gives logits(1,19,264);four letters+EOS are five effective targets.",
        "location":"course/chapters/12.md:291-293","status":"verified","scope":"The default0.1second440Hz waveform and the exact quoted five-token prefix; not a fixed length for arbitrary audio.",
        "evidence":[E("repo-data","ByteTokenizer16-21","Vocab264=8special IDs+256byte IDs;high has4ASCII bytes."),E("repo-modal","log_mel62-69;expand84-100","Actual11 frame features;expand first then input[:-1],target[1:]."),E("exec-original","original-cpu/stdout.txt","Original exact fence prints(1,19,264)."),E("exec-bounded","bounded-cpu.json /cases/0/shifted_labels,/effective_positions,/manual_ce_mean","Five target positions14-18,CE arithmetic independently gathered.")],
        "artifact_ids":A("bounded-cpu.json","original-cpu/execution.json"),
        "verification":V("10-1+11=20;20-1=19;264vocab;5nonignored targets.","Shape[1,19,264],unshifted20,nonignored14-18=112,113,111,112,2.","All dimensions/counts checked exactly;loss5.856474876403809 matches gathered mean over5 targets,not19 or264.",tolerance="Shapes/IDs/counts exact;independentCE atol1e-6 rtol0.")},
    {"id":"ignored-loss","kind":"concept","statement":"Prefix and inserted audio labels use-100, so direct answer cross-entropy only counts answer letters and EOS;audio features can still affect those supervised predictions.",
        "location":"course/chapters/12.md:293","status":"verified","scope":"Class-index cross-entropy with repository IGNORE=-100;ignoring direct labels does not remove the audio embedding path from the computation graph.",
        "evidence":[E("torch-loss","CrossEntropyLoss1222-1236 and ignore_index1275-1278","The unreduced loss indicator excludes ignored targets;the mean divides by nonignored targets."),E("repo-model","loss_sum92-100;masked_loss103-105","Repository explicitly sums cross entropy and divides by nonignored target count."),E("repo-modal","expand89-100","All inserted feature target labels areIGNORE before the single shift."),E("torch-backward","Tensor.backward570-579","Chain rule still carries answer-loss gradients to used leaf parameters.")],
        "artifact_ids":A("bounded-cpu.json")},
    {"id":"training-context","kind":"concept","statement":"Answer loss uses correct preceding answer tokens,whereas actual autoregressive generation conditions on previously generated tokens.",
        "location":"course/chapters/12.md:293","status":"verified","scope":"Teacher-forcing versus autoregressive input conditioning;not evidence that the random model generates correct high.",
        "evidence":[E("scheduled-paper","Abstract/Introduction;p2Sec2.1eq1-3;p3Sec2.3","Ground-truth prior target at training and unavailable target replaced by generated prior token at inference."),E("repo-modal","expand98-100;generate_modal149-172","Loss aligns shifted original answer inputs;generation appends each selected token toids."),E("repo-attention","attention_mask10-18;CausalAttention64-74","Scores at first answer position cannot access future answer inputs."),E("exec-bounded","bounded-cpu.json /cases/*/future_answer_change_first_answer_logit_max_abs_difference","Changing future answer input leaves first-answer scores exactlyunchanged.")],
        "artifact_ids":A("sources/scheduled-sampling-1506.03099v3.pdf","bounded-cpu.json")},
    {"id":"gradient-and-exercise","kind":"software","statement":"Nonzero audio projector gradients show a connected loss path,not learned ability;no optimizer step runs.The200Hz exercise must change both answer and effective labels tolow.",
        "location":"course/chapters/12.md:295-297","status":"verified","scope":"Original demonstration and one specified short200/low variation.All model parameters remain exactlyunchanged;no frozen-language claim is made.",
        "evidence":[E("torch-backward","Tensor.backward570-580,623-625","Backward computes/accumulates gradients via chain rule."),E("repo-modal","MultiModalLM104-135 andexpand84-100","Loss depends on projected audio andlabelsare aligned afterexpansion."),E("exec-original","original-cpu/stdout.txt","OriginalprojectorgradientTrue."),E("exec-bounded","bounded-cpu.json /cases/0,/cases/1","Both positivegradients;unchangedparameters;low labels are116,119,127,2.")],
        "artifact_ids":A("original-cpu/execution.json","bounded-cpu.json","bounded-200-low.py"),
        "verification":V("440/high gradient>0,no weightupdate;200/low gradient>0 and4effective targets.","Gradientnorms0.07666922360658646 and0.02479259856045246;changed_parameter_names=[]both;200shape[1,18,264].","Executed random-weight forward/backward only;exacttorch.equal before/after;exercise uses low inbothids/labels;no optimizer.")},
    {"id":"historical-11-of-14","kind":"empirical","statement":"The separate existing trained-tone experiment generated11 correct complete answers out of14test questions.",
        "location":"course/chapters/12.md:295","status":"verified","scope":"One pre-existing seeded artificial-tone run;exact raw answer-token match plus separately checked EOS,not speech understanding or anew model evaluation.",
        "evidence":[E("exec-raw","audio.json /results/test/samples,/results/test/examples,/correct,/effective_tokens;raw-reaggregation.json","Recomputed each complete answer from original generated_ids andverified11/14,EOS14/14,62targettokens."),E("repo-recipe","_audio_records195-216;_evaluate287-347;run_audio884-904","Frequency-family splits,pair conditions,exact byte-answer criterion beforeEOS,anddistinct training/evaluation workflow."),E("exec-raw","audio.json /revision,/device,/seed,/torch_version,/python_version,/code_sha256;/results/training/history","Historicalrevision/version/device and300step rawhistory5409targets;necessarycurrentcodehashesequalrecordedversion.")],
        "artifact_ids":A("raw-reaggregation.json","inputs/docs/course-experiments/results/audio.json"),
        "verification":V("14rawtestsample rows;11exactanswers;finitehistoryrepresentstheseparatehistoricalrun.","11/14=0.7857142857142857;14/14EOS;62targettokens;train/validation/test16/8/14;300historicalsteps,5409effective targets.","JSON-only reaggregation checks eachgenerated_ids/target/record pair andcodeSHA;notretrainingornoveltrainedmodel evaluation.",denominators=denominators)},
    {"id":"duration-scope","kind":"empirical","statement":"The original sentence says errors concentrate at the boundary and duration variations.",
        "location":"course/chapters/12.md:295,quote:錯誤集中在邊界與時長變化","status":"unverified","scope":"Boundary-location part is verified;isolated duration effect is not supported because amplitude andduration are paired,and bothdurations occur inevery split.",
        "evidence":[E("exec-raw","audio.json /results/test/samples/4,/6,/7;/results/data/splits/*/records;raw-reaggregation.json /failed_rows","Threefailedrows all290/300Hz;conditions0.25/0.1and0.5/0.12occurtogether;noisolateddurationcontrol."),E("repo-recipe","_audio_records195-215","Everyfrequencyusespaired(amplitude,seconds)of(0.25,0.1)and(0.5,0.12),so durationandamplitude cannotbe separated.")],
        "artifact_ids":A("raw-reaggregation.json","inputs/docs/course-experiments/results/audio.json","inspection.md"),
        "verification":V("A duration-specific conclusion would require isolated duration variation or explicit scope limiting the claim to paired conditions.","Failedrows4,6,7 at290/300Hz;2short0.1s and1long0.12s;amplitude changes simultaneously;noindependentduration evidence.","Recomputed failures fromoriginalgeneratedIDs;allthree are nearfrequencyboundary.Donotassume a duration cause or add new experiment to rescue the wording.",denominators=denominators)},
]
report = {"schema_version":1,"review_stage":"technical","lesson_id":"12.9","source":"course/chapters/12.md#12.9",
    "source_sha256":sha(ART/"extracted/section.md"),"figure_sha256":{},"verdict":"revise",
    "reviewer_task":"/root/phase4_factual_coordinator/factual_12_9","reviewer_context":"fresh","author_tasks":[],
    "reviewed_on":"2026-10-05","read_scope": "Full current12.9 andnecessary12.8/10.7/11.3,originalcode contracts,officialsource originals andnamedrawaudioJSON pointers.See personallywritten inspection.md.",
    "frozen_input":{"path":REL+"/inputs/course/chapters/12.md","sha256":sha(ART/"inputs/course/chapters/12.md"),"meaning":"WholeMarkdown bytes frozen at initialread;not anunqualified currentwholechapterversion."},
    "sources":sources,"artifacts":artifacts,"claims":claims,
    "issues":[{"id":"duration-confound","status":"unresolved","claim_ids":["duration-scope"],"location":"course/chapters/12.md:295",
        "original":"錯誤集中在邊界與時長變化","evidence":"Allthree errorsare at290/300Hz;amplitudeandduration arecompletelypaired ineachsplit.Noisolatedduration comparisonexists.",
        "impact":"The sentence canleadreaders toattribute an errorpattern toduration,whenrecords cannotseparate duration fromamplitude andfrequency-boundarydifficulty.",
        "recommendation":"Replacewith anexactscope:三個錯例都在290/300Hz；資料把振幅與時長一起改動，不能分辨兩者各自影響。Keep11/14andthegradient/training distinction.",
        "resolution":"Unresolved in the reviewed originalsource;reviewer hasnotchanged教材.Must personallyinspectcoordinator's correction beforeupdating verdict."}],
    "checks":{
        "factual_accuracy":{"status":"revise","details":"Input/path/masking/teacherforcing/gradient claimsverified;duration-scope empiricalphrase remainsunsupported.","claim_ids":[c["id"] for c in claims]},
        "numeric_verification":{"status":"pass","details":"Axes,widths,20→19 positions,264vocabulary,5targets,200variant4targets and11/14/62-token arithmetic verified by actualshortCPU/rawJSONchecks.","claim_ids":["lengths-targets","historical-11-of-14"]},
        "figure_consistency":{"status":"not_applicable","details":"12.9hasnoimageorSVG;actualdesktop/mobilepagewasrendered andviewed.Thetwo necessaryprerequisitefigureswerealsorendered/viewedandmatchtheirsource;theyareseparateprerequisiteevidence.","claim_ids":[]},
        "source_verification":{"status":"pass","details":"OfficialPyTorchimmutableoriginalsource personallyread andHTTPSbytesverified;originalBengioPDF version/title/author andrelevantsectionsread.RawJSONpreserved andinspectedonly throughnamedpointers;priortechnical/reader/authorinterpretationsunread.","claim_ids":[c["id"] for c in claims]},
        "limitations":{"status":"revise","details":"Originaldemo isonlyrandomweightforward/backward;historicalJSON isonefiniteartificialtonerun.Noretrainingorcheckpointload.Pairedamplitude/durationdesignprevents verifyingtheoriginaldurationphrase;needswording correction.","claim_ids":["gradient-and-exercise","historical-11-of-14","duration-scope"]}},
    "limitations":["No model/data download,existing-model re-evaluation,GPU orpaidjob;complete recipe notrun.","No new.ptremains;opaque runtimeworkspacecopy mistake was corrected bydeletingonlythisreviewer'snewcopy,eventretained;no semanticexposure to copiedreports.","Browserfile://prerequisiteSVG restrictionlogged;authorizedrawSVGbytesrenderedviapage.set_content,same scientificverdict."],
    "independence_event_record":REL+"/artifact-copy-correction.json"}
# Keep the delivered record readable in the task's language.
claim_text = {
    "audio-entry": ("按本題 >300 Hz 的規則，440 Hz 的答案是 high。<audio> 提供插入位置，11 條聲音特徵經接頭轉成文字寬度，標記沒有預寫答案。", "只核本 repo 的人工單音任務與隨機初始化 MultiModalLM 入口，沒有證明學會頻率辨識。", "預期 440>300，標記展開為 11 條寬 8 向量。", "實得聲音特徵 [1,11,16]、接頭 [1,11,8]，前綴 [1,3,6,2,4]。", "原 fence 加短 CPU instrumentation；high 的 bytes 為 112,113,111,112，與 audio marker 6 分開。"),
    "lengths-targets": ("5 個前綴加 high 的 4 bytes 與 EOS 共 10；1 marker 換 11 向量得 20，一次移位得 logits (1,19,264)，有效目標為 4 字母與 EOS。", "只適用本例預設 0.1 秒、440 Hz 波形及五位置前綴，任意音訊不固定為 11 框。", "10−1+11=20，20−1=19，詞表 264，5 個有效目標。", "實得 [1,19,264]、20 展開位置、索引 14–18 的目標為 112,113,111,112,2。", "形狀、ID、計數精確；loss 5.856474876403809 與 5 目標的獨立平均交叉熵相同，沒有用 19 或 264 當分母。"),
    "ignored-loss": ("前綴與聲音插入位置的 −100 標籤不直接計答案交叉熵；聲音仍能影響後面的受監督預測。", "class-index 交叉熵使用 IGNORE=-100；忽略直接 target 不等於移除聲音計算圖。", "", "", ""),
    "training-context": ("計算答案損失使用正確的答案前文，真正逐 token 生成則使用自己上一個輸出。", "teacher forcing 與 autoregressive conditioning 的區分；不是隨機模型可生成正確 high 的證據。", "", "", ""),
    "gradient-and-exercise": ("非零聲音接頭梯度表示答案損失通路相连；本例沒有 optimizer step。200 Hz 練習必須同時把答案與有效目標改成 low。", "原示範加指定的 200/low 短變體；所有模型參數精確不變，本節沒有聲稱凍結語言模型。", "兩個頻率的接頭都收到非零梯度且參數不變，low 有 4 個有效目標。", "梯度 norm 為 0.07666922360658646 與 0.02479259856045246；兩例 changed_parameter_names 都空；200 Hz logits [1,18,264]。", "只執行隨機初始 forward/backward，以 torch.equal 檢查前後參數，沒有 optimizer；low 同時進 ids 與 labels。"),
    "historical-11-of-14": ("另一次既有單音訓練的 14 題 test，生成完整答案正確 11 題。", "只核一個既有 seeded 人工單音 run 的完整答案 byte exact-match，另外核 EOS；不代表真人語音理解或本次重評模型。", "14 個原始測試樣本，11 個完整答案相符；獨立歷史訓練另有原始 history。", "11/14=0.7857142857142857、EOS 14/14、目標 62 tokens；split 16/8/14；歷史 300 步、5409 有效目標。", "JSON-only 重新彙總每個 generated_ids/target/record 及必要 code SHA，沒有訓練或重評既有模型。"),
    "duration-scope": ("原文稱「錯誤集中在邊界與時長變化」。", "邊界部分已驗證；振幅與時長完全成對，兩種時長在每個 split 都出現，不能單獨確認時長影響。", "時長的獨立解釋需要只改時長的控制，或明確限制為振幅與時長成對的條件。", "錯例 row 4/6/7 全在 290/300 Hz；2 個 0.1 秒、1 個 0.12 秒，振幅也同時不同，沒有單獨時長證據。", "以原 generated_ids 重新計算錯例；三個都在頻率邊界附近，不替時長原因作假設，也不新增訓練挽救文句。"),
}
support_text = {
    "audio-entry": ["audio_id=6 是輸入標記，high 是另四個 byte IDs。", "波形 1600 samples，11 框特徵，接頭 16→8；用特徵取代 marker，沒有把答案文字放入 marker。", "affine 最後軸寬度變換保持 batch 與時間軸。", "實際核 440/high 的 feature/projector shapes 與原 IDs。"],
    "lengths-targets": ["264=8 特殊 IDs+256 byte IDs；high 有 4 ASCII bytes。", "實得 11 框；先展開再 input[:-1] 與 target[1:]。", "原 fence 真正印出 (1,19,264)。", "索引 14–18 共 5 目標，獨立擷取 logits 重算交叉熵平均。"],
    "ignored-loss": ["原公式指示因子排除 ignore_index，平均分母只算有效 target。", "repo 明確求交叉熵總和除有效 target 數。", "插入特徵的 labels 在一次移位前填 IGNORE。", "chain rule 仍把答案損失傳回有參與計算的 leaf parameters。"],
    "training-context": ["訓練時用真實上一 target，推理時無法取得它，改用自生成上一 token。", "損失輸入是原答案序列的一次移位；生成逐次將選出 token 接回 ids。", "第一答案位置看不到未來答案輸入。", "改未來答案輸入後，第一答案 scores 差精確零。"],
    "gradient-and-exercise": ["backward 用 chain rule 計算並累積梯度。", "答案 loss 依賴 projected audio，展開後 labels 對齊。", "原聲音接頭 gradient 為 True。", "兩例梯度正、參數不變，low targets 為 116,119,127,2。"],
    "historical-11-of-14": ["由原 generated_ids 重新核完整答案 11/14、EOS 14/14、62 target tokens。", "核 frequency-family split、成對條件、EOS 前完整 byte exact-match，以及獨立訓練/測試流程。", "原記錄 revision/version/device；history 300 步、5409 targets；必要 code hash 與現原碼相同。"],
    "duration-scope": ["三個錯例全在 290/300 Hz；0.25/0.1 與 0.5/0.12 條件成對出現，沒有只改時長的控制。", "每個頻率都用 (振幅,秒數)=(0.25,0.1),(0.5,0.12)，無法分離兩者。"],
}
for claim in claims:
    statement, scope, expected, observed, details = claim_text[claim["id"]]
    claim.update(statement=statement, scope=scope)
    if "verification" in claim:
        claim["verification"].update(expected=expected, observed=observed, details=details)
    for ref, supports in zip(claim["evidence"], support_text[claim["id"]], strict=True):
        ref["supports"] = supports
report["read_scope"] = "完整親讀現 12.9 與必要 12.8/10.7/11.3，核原實作契約、官方原文及具名原始 audio JSON pointers；本人完整記錄在 inspection.md。"
report["frozen_input"]["meaning"] = "初讀時凍結的完整 Markdown bytes，不冒稱其為日後目前整章版本。"
issue = report["issues"][0]
issue.update(evidence="三錯全在 290/300 Hz；各 split 的振幅與時長完全配對，沒有時長獨立控制。", impact="讀者可能把錯例歸因時長，資料卻不能區分時長、振幅與邊界難度。", recommendation="改成：三個錯例都在 290/300 Hz；資料把振幅與時長一起改動，不能分辨兩者各自影響。保留 11/14 與梯度/成果區分。", resolution="所審原文仍未修正；本人沒有修改教材，須親自複核協調者修正文才更新判定。")
for name, details in {
    "factual_accuracy": "入口、標籤、teacher forcing 與梯度 claims 已核，duration-scope 實質文句仍未受支持。",
    "numeric_verification": "實際短 CPU/raw JSON 核軸、寬度、20→19、264 詞表、5 有效目標、200 Hz 的 4 目標及 11/14、62 tokens。",
    "figure_consistency": "12.9 無 image/SVG，已真正 render/view 桌面與手機現頁；另 render/view 兩張必要前置圖並核其原 bytes，不列為本節引用圖。",
    "source_verification": "本人讀官方 PyTorch immutable 原文並核 HTTPS bytes；核 Bengio 原 PDF 作者、版本、相關段落。原 JSON 完整保存，只讀具名 pointers；未讀舊審閱或作者額外結果解釋。",
    "limitations": "隨機初始短例僅 forward/backward，歴史 JSON 是有限人工單音 run，沒有重新訓練或載入 checkpoint；成對振幅/時長設計不能核原時長文句，需修訂。",
}.items():
    report["checks"][name]["details"] = details
report["limitations"] = ["沒有模型/資料下載、既有模型重評、GPU、付費工作或完整 recipe 執行。", "正式 artifact 無 .pt；錯誤 opaque runtime copy 只刪本人新產生的 copy，未改原 inputs 或其他工作者檔案；沒有 load/evaluate/訓練或語義閱讀被複製報告；保留事件。", "Chromium file:// 前置 SVG 限制有記錄；以原 SVG bytes 使用 page.set_content 成功 render/view，沒有更改科學判定。"]
path=ROOT/"docs/technical-reviews/12.9.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
assert json.loads(path.read_bytes())["reviewer_task"] == "/root/phase4_factual_coordinator/factual_12_9"
print(json.dumps({"report":path.relative_to(ROOT).as_posix(),"reviewer_task":report["reviewer_task"],"verdict":report["verdict"],"source_sha256":report["source_sha256"],"report_sha256":sha(path),"claims":len(claims),"artifacts":len(artifacts)},ensure_ascii=False))
