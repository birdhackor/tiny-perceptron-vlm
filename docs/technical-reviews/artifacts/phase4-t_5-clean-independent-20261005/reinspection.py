"""Actual original-reviewer callback: whole new T.5 and the narrow changed navigation contract."""
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
body=lambda raw,lesson: next(raw[h.start():(heads[i+1].start() if i+1<len(heads) else len(raw))] for heads in [list(re.finditer(rb'(?m)^## [^\r\n]+',raw))] for i,h in enumerate(heads) if h[0].startswith(('## '+lesson+' ').encode()))
new=body((ROOT/'course/training.md').read_bytes(),'T.5')
old=(HERE/'extraction/section.md').read_bytes()
old_sentence='使用示例與原始欄位見[實驗說明](../docs/course-experiments/README.md)。'.encode()
new_sentence='固定實驗入口與基底核對提醒見[實驗說明](../docs/course-experiments/README.md)。'.encode()
assert old.count(old_sentence)==1 and new.count(new_sentence)==1
assert old.replace(old_sentence,new_sentence)==new
assert hashlib.sha256(new).hexdigest()=='e6a035980e108a79f6bf53e0b85aa2920f0209e9da8473ffcf2b0eee447b45c3'
callback=HERE/'callback'
callback.mkdir(exist_ok=True)
(callback/'section.md').write_bytes(new)
(callback/'extraction.json').write_bytes(Path('/tmp/t5-clean-callback-20261005/extraction.json').read_bytes())
dependencies={}
for name in ['scripts/prepare_data.py','scripts/train.py','scripts/evaluate.py','scripts/fetch_training_assets.py','scripts/course_experiments/run.py','scripts/course_experiments/behavior.py','scripts/course_experiments/common.py','tiny_perceptron/alignment.py','tiny_perceptron/adapters.py','tiny_perceptron/data.py','scripts/course_release.py','docs/course-experiments/README.md','assets/training/manifest.json','assets/training/sources/pku-safe-rlhf.json','docs/course-experiments/results/style.json','docs/course-experiments/results/safety.json','docs/course-experiments/results/lora.json','docs/course-experiments/public-models.json','docs/course-experiments/releases/safety.json']:
    actual=sha(ROOT/name)
    prior=sha(HERE/'frozen'/name)
    assert actual==prior,name+' actual dependency changed'
    dependencies[name]={'current_sha256':actual,'initial_frozen_sha256':prior,'equal':True}
for name,lessons in [('course/chapters/08.md',['8.13','8.16']),('course/chapters/09.md',['9.1','9.2','9.4','9.6'])]:
    frozen=(HERE/'frozen'/name).read_bytes()
    current=(ROOT/name).read_bytes()
    for lesson in lessons:
        before=body(frozen,lesson);after=body(current,lesson)
        assert before==after
        dependencies[name+'#'+lesson]={'raw_section_sha256':hashlib.sha256(after).hexdigest(),'unchanged_required_section':True}
original=ROOT/'data/training/behavior-initial/pku-safe-rlhf/train-first-100.jsonl'
assert sha(original)==sha(HERE/'raw/pku-train-first-100.jsonl')
text=(ROOT/'docs/course-experiments/README.md').read_text().splitlines()
assert any('scripts.course_experiments.run --experiment' in line for line in text[10:30])
assert any('--dependencies' in line for line in text[10:30])
assert 'LoRA同時核對基底' in text[25]
assert '原始浮點基模並核對指紋' in new.decode() and 'merged-*.pt' in new.decode() and '不再加同一 adapter' in new.decode()
result={'reviewer_task':'/root/phase4_factual_coordinator/factual_t_5_clean','time_utc':datetime.now(timezone.utc).isoformat(),'command':'.venv/bin/python '+str(Path(__file__).relative_to(ROOT)),'environment':{'python':platform.python_version(),'device':'cpu','operation':'read/hash/AST-contract reinspection only; no retraining or new model measurement'},'actual_read_scope':['new whole course/training.md#T.5 (all'+str(len(new.splitlines()))+'lines)','docs/course-experiments/README.md lines11–30 only, no lines1–10 body','tiny_perceptron/adapters.py lines55–92','scripts/course_experiments/behavior.py current _merge_lora230–244','matching original safety behavior run_safety490–508 and _pku_pilot455–477'],'initial_source_sha256':hashlib.sha256(old).hexdigest(),'current_source_sha256':hashlib.sha256(new).hexdigest(),'only_T5_change':{'old_sentence':old_sentence.decode(),'new_sentence':new_sentence.decode(),'exact_replacement_verified':True},'fence_and_figure_scope':'Both original bash fences unchanged; no Python fences or SVG in T.5. Previous real CPU contract checks remain tied to unchanged implementations and raw material.','frozen_whole_input':{'path':str((HERE/'frozen/course/training.md').relative_to(ROOT)),'sha256':sha(HERE/'frozen/course/training.md'),'meaning':'initial full Markdown bytes only, retained; not current whole chapter fingerprint'},'excluded_adjacent_scope':'T.6/T.8 are outside actual necessary read dependencies; their later changes do not trigger invented checks.','navigation_actual_support':{'readme_sha256':sha(ROOT/'docs/course-experiments/README.md'),'runner_locator':'lines13–21: fixed runner flags, output root and explicit --dependencies','base_reminder_locator':'line26: LoRA base verification reminder','technical_claims_preserved':'native float base and fingerprint plus merged-not-reapplied instructions unchanged; full original source/loader and nonzeroB merge tests already independently verified'},'independent_adequacy_judgment':'New link accurately promises fixed-run entry and a base-check reminder. T.5 teaches separating style, instruction and safety goals; it does not promise an adapter-loading lesson. The local text still supplies the necessary native-base and no-double-application rules. Replacing the inaccurate navigation wording does not remove or weaken those substantive rules, and no unresolved material claim remains.','callback_tool_issue':'First callback command exited1 due to an incorrect fixed line index for --dependencies in this reviewer script, not a source claim problem. Original stderr preserved; corrected to the actual 11–30 range and rerun.','dependency_fingerprints':dependencies,'raw_original_copy_sha256':sha(original),'verdict':'pass','issues_remaining':[]}
(callback/'reinspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
