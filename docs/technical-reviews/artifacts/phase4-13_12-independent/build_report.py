"""Write this review's new complete canonical report without reading an old one."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
DEST=Path(__file__).resolve().parent
PREFIX=DEST.relative_to(ROOT).as_posix()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

spec=importlib.util.spec_from_file_location('sf',ROOT/'docs/review-tools/section_facts.py')
sf=importlib.util.module_from_spec(spec);spec.loader.exec_module(sf)
raw,whole,first_line=sf.original_section(ROOT/'course/chapters/13.md','13.12')
frozen=DEST/'frozen-input/section.md'
assert raw==frozen.read_bytes(),'section changed since this actual review; do not certify another version'
original_execution=json.loads((DEST/'original-fence.execution.json').read_bytes())
original_env=json.loads((DEST/'original-fence.environment.json').read_bytes())
bounded_env=json.loads((DEST/'bounded-cpu.environment.json').read_bytes())
assert original_execution['exit_code']==0
assert (DEST/'original-fence.stdout.txt').read_text().strip()=='更新前後的機率比 [2.0, 0.5]'
assert (DEST/'bounded-cpu.stderr.txt').read_bytes()==b''
assert len(json.loads((DEST/'render-result.json').read_bytes())['screenshots'])==2
fetch=json.loads((DEST/'official-fetch-provenance.json').read_bytes())
accessed=fetch[0]['accessed_on']

base_env={k:str(v) for k,v in bounded_env.items()}
render_env={'Chromium':'151.0.7922.173','Playwright':'1.63.0','Markdown':'3.11','device':'CPU/headless'}
commands={
 'original-fence.stdout.txt':(original_execution['command'],'exit 0; original unmodified fence printed [2.0, 0.5]',base_env),
 'bounded-cpu.stdout.txt':("CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python "+PREFIX+'/verify_bounded_cpu.py','exit 0; arithmetic, batch/action axes, old/new gradient separation, detached-storage alias, saved clone and bounded underflow variant passed; zero optimizer updates',base_env),
 'render.stdout.txt':('timeout 60s .venv/bin/python '+PREFIX+'/render_section.py','exit 0 after using page.set_content; both screenshots generated and independently opened',render_env),
 'render.attempt1.stderr.txt':('timeout 60s .venv/bin/python '+PREFIX+'/render_section.py (exact attempt1 code retained as render_section.attempt1.py)','exit 1; file:// URL rejected with ERR_BLOCKED_BY_ADMINISTRATOR; no timeout',render_env),
 'official-fetch.stdout.txt':('.venv/bin/python '+PREFIX+'/fetch_official_sources.py','exit 0; three official PyTorch API originals and exact OpenAI repository commit fetched',{'python':base_env['python'],'device':'CPU','network':'HTTPS original authority retrieval only'}),
}
descriptions={
 'inspection-ledger.md':'Own reading ranges, original authority locators, arithmetic and units, visual inspection and honest limitations.',
 'commands.txt':'Actual command history, original worker invocation and results, including first rendering failure and second successful attempt.',
 'original-fence.execution.json':'Actual execution argv, exit status, elapsed time and original output hashes from the section helper.',
 'original-fence.environment.json':'Actual original-fence CPU environment, torch version and guard policy; no model training.',
 'bounded-cpu.result.json':'Own numerical and autograd measurements, batch/action-axis construction and historical-storage behavior.',
 'bounded-cpu.environment.json':'Actual CPU version/device/dtype/thread record for independent checks.',
 'official-fetch-provenance.json':'Fetch URL, resolved URL, original version/commit, date and SHA for personally inspected sources.',
 'section-1280x800.png':'Actual desktop screenshot, opened with view_image; isolated frozen section rather than final course theme.',
 'section-390x844.png':'Actual mobile screenshot, opened with view_image; isolated frozen section rather than final course theme.',
 'render-result.json':'Actual viewport sizes and document widths; both screenshots created after successful render.',
 'frozen-input/chapter13-frozen-input.md':'Original whole-chapter frozen input snapshot, explicitly not a live entire-chapter fingerprint.',
 'frozen-input/section.md':'Exact original UTF-8 section bytes underlying source_sha256.',
 'frozen-input/fence-1.py':'Actual unchanged short original Python fence, executed on CPU.',
 'frozen-input/bootstrap.py':'Actual extracted course bootstrap used by original helper worker.',
 'frozen-input/extraction.json':'Actual original-byte extraction hashes, line locator, fence metadata and no-figure inventory.',
 'ppo-1707.06347v2.pdf':'Original authoritative PPO v2 PDF, personally read and version-checked on PDF page 1.',
 'instructgpt-2203.02155v1.pdf':'Original authoritative InstructGPT v1 PDF, personally read and version-checked on PDF page 1.',
}
artifacts=[];artifact_for={}
for p in sorted(DEST.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name in {'artifact-manifest.json','report-generation.stdout.txt','report-generation.stderr.txt','canonical-written.json','checker.stdout.txt','checker.stderr.txt','report.sha256'} or '__pycache__' in p.parts:continue
 name=p.relative_to(DEST).as_posix();identifier='artifact-'+name.replace('/','-').replace('.','-')
 kind='figure_render' if p.suffix=='.png' else ('code' if p.suffix=='.py' else 'source_snapshot')
 a={'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'kind':kind,
    'description':descriptions.get(name,('Own actual review code or frozen original method: ' if p.suffix=='.py' else 'Preserved original source, raw output or provenance: ')+name)}
 if name in commands:
  command,result,env=commands[name];a.update(kind='execution',command=command,result=result,environment=env)
 artifacts.append(a);artifact_for[name]=identifier
dump(DEST/'artifact-manifest.json',[{'path':a['path'],'sha256':a['sha256'],'kind':a['kind']} for a in artifacts])

sources=[
 {'id':'ppo-paper','kind':'paper','title':'Proximal Policy Optimization Algorithms','url':'https://arxiv.org/pdf/1707.06347v2','version':'arXiv:1707.06347v2, 2017-08-28','accessed_on':accessed,'authority_reason':'Original PPO method paper by Schulman et al. at OpenAI','verified':True,'checked_original':True,'inspection_note':'Personally checked PDF page-1 v2 identifier, sections 2.2–3 formula 6–7 and section 5 Algorithm 1 on PDF pages 2–5; original PDF/text retained.'},
 {'id':'instructgpt-paper','kind':'paper','title':'Training language models to follow instructions with human feedback','url':'https://arxiv.org/pdf/2203.02155v1','version':'arXiv:2203.02155v1, 2022-03-04','accessed_on':accessed,'authority_reason':'Original InstructGPT SFT-to-PPO RLHF method paper by Ouyang et al.','verified':True,'checked_original':True,'inspection_note':'Personally checked page-1 v1 identifier and section 3.2 reinforcement-learning paragraph/equation 2, PDF page 9; learned policy differs from starting pi_SFT reference.'},
]
for key,file in [('detach','detach-2.9.html'),('log','log-2.9.html'),('exp','exp-2.9.html')]:
 record=next(r for r in fetch if r['path']==file)
 sources.append({'id':'torch-'+key,'kind':'official_docs','title':'PyTorch '+('Tensor.detach' if key=='detach' else 'torch.'+key)+' API','url':record['url'],'version':'PyTorch 2.9 documentation; executed environment 2.14.1+cpu','accessed_on':record['accessed_on'],'authority_reason':'Official PyTorch versioned API documentation','verified':True,'checked_original':True,'inspection_note':'Personally read complete '+key+' API article from fetched official HTML; original and API-only text retained. Execution confirmed the relevant contract on installed 2.14.1+cpu.'})
record=next(r for r in fetch if r['path']=='openai-spinningup-ppo.py')
sources.append({'id':'openai-ppo-source','kind':'official_source','title':'OpenAI Spinning Up PyTorch PPO original source','url':'https://github.com/openai/spinningup/blob/'+record['version']+'/spinup/algos/pytorch/ppo/ppo.py','version':record['version'],'accessed_on':record['accessed_on'],'authority_reason':'Original PPO implementation maintained in OpenAI official repository','verified':True,'checked_original':True,'inspection_note':'AST function inventory first, then personally read buffer store/get lines 12–40,71–84; compute_loss_pi 227–243; collection/update 295–316,335–354. Log probabilities are saved during collection, current logp recomputed in the objective.'})
for sid,path,loc in [
 ('repository-helper','tiny_perceptron/posttraining.py','AST located bandit_advantage 24–28 and ppo_clipped_objective 31–50; read 1–53 and executed small objective helper only.'),
 ('repository-rollout','scripts/course_experiments/posttraining.py','AST identified run_posttraining and old/reference assignment ranges; actually read 285–290 and 324–369. Read method only, not result dictionary 461–533; did not run training.')]:
 p=DEST/'frozen-input'/path
 sources.append({'id':sid,'kind':'repository_code','title':path+' frozen original implementation','path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'version':'Exact frozen original source SHA-256 '+sha(p),'verified':True,'inspection_note':loc})
sources += [
 {'id':'arithmetic','kind':'derivation','title':'Independent selected-action ratio and surrogate arithmetic','verified':True,'details':'For positive p_old, exp(log p_new-log p_old)=p_new/p_old. Separately conditioned selected probabilities: .4/.2=2, .1/.2=.5, .3/.2=1.5. The dimensionless ratio times fixed reward-unit advantage .6 is 1.2; it is an objective term, not measured quality. A batch/action construction with two normalized rows verifies the sample axis. Recorded in inspection-ledger.md and executed bounded-cpu code.'},
 {'id':'original-execution','kind':'execution','title':'Actual unchanged original fence CPU run','verified':True,'artifact_id':artifact_for['original-fence.stdout.txt']},
 {'id':'bounded-execution','kind':'execution','title':'Actual independent bounded CPU variants','verified':True,'artifact_id':artifact_for['bounded-cpu.stdout.txt']},
]
def ev(source,locator,supports): return {'source_id':source,'locator':locator,'supports':supports}
def verification(expected,observed,details):return {'method':'executed','expected':expected,'observed':observed,'details':details,'tolerance':'1e-12 float64; displayed original float32 ratio checked after round(value,4)'}
def claim(cid,kind,statement,location,scope,evidence,files=(),verify=None):
 c={'id':cid,'kind':kind,'statement':statement,'location':location,'scope':scope,'status':'verified','evidence':evidence,'artifact_ids':[artifact_for[f] for f in files]}
 if verify:c['verification']=verify
 return c
claims=[
 claim('C1','concept','old policy is the policy that collected the batch; ratio is new/old probability of the same selected card for the same context.','13.12 paragraphs 1–2','Ratio 1 refers to the unchanged selected probability, not the entire policy; old selected probabilities must be positive.',[ev('ppo-paper','Section 3 definition of r_t and equation 6, PDF p3; Algorithm 1 p5','Same state/action likelihood divided by the collecting policy likelihood; sample under old policy then update.'),ev('openai-ppo-source','PPOBuffer.store 30–40 and compute_loss_pi 227–233','Store sampled logp and compare current selected logp against it.')],['ppo-1707.06347v2.pdf','openai-spinningup-ppo.py']),
 claim('C2','numeric','The independent selected probabilities .2→.4 and .2→.1 have ratios [2,.5]; the two entries need not sum to one.','13.12 original Python fence and following paragraph','Batch axis contains two different selections, not one complete two-action distribution; ratio is dimensionless.',[ev('arithmetic','inspection-ledger.md: Independent arithmetic, axes and execution','Explicit divisions and normalized (batch=2,action=2) rows with selected action gather.'),ev('original-execution','original-fence.stdout.txt','Unmodified fence displays [2.0,0.5].'),ev('bounded-execution','bounded-cpu.result.json: /ratio, /old_distribution_batch_action, /new_distribution_batch_action, /per_context_probability_sums','Independent normalized-row construction and selected-action ratios.')],['original-fence.stdout.txt','bounded-cpu.stdout.txt','bounded-cpu.result.json'],verification('[2,0.5]; each full context distribution sums to one','original [2.0,0.5]; float64 [2.0,0.5000000000000001]; row sums [1,1]','Two selected-action samples, each with its own unobserved complementary action; no mixing of batch and action axes.')),
 claim('C3','software','The original code uses natural log, elementwise exponentiation and detach to form a fixed-old probability ratio.','13.12 Python fence, lines 380–388','Arithmetic demonstration: original inputs do not require gradient, and there is no backward or optimizer call. Enabled-gradient variation separately checks detach.',[ev('torch-log','torch.log API: elementwise natural logarithm definition','Natural-log semantics used by Tensor.log.'),ev('torch-exp','torch.exp API: elementwise e^x definition','Exp of a log difference yields the ratio for positive inputs.'),ev('torch-detach','Tensor.detach API: graph and storage contract','Old log graph is disconnected; detach is not a storage copy.'),ev('original-execution','original-fence.execution.json and stdout','Actual original fence completed on installed CPU PyTorch.'),ev('bounded-execution','verify_bounded_cpu.py old/new requires_grad branch; bounded-cpu.result.json /old_log_gradient, /new_log_gradient','Gradient flows to new logp but not detached old logp.')],['frozen-input/fence-1.py','original-fence.stdout.txt','bounded-cpu.stdout.txt','bounded-cpu.result.json'],verification('Original output [2.0,0.5]; old gradient None; new sum-of-unclipped-term gradient [1.2,-0.2]','All observed; old gradient null, new [1.2,-0.20000000000000007]','API coverage includes torch.tensor, Tensor.log/exp/detach/tolist and rounding in actual execution. Official docs are version 2.9; actual environment is 2.14.1+cpu.')),
 claim('C4','concept','Detach prevents old log gradient, but historical values must actually be saved; recomputing the denominator from the new policy would remove the change.','13.12 paragraph following code','Graph separation and historical persistence are distinct. The current fence log creates a separate result, and original training captures selected old values once per rollout.',[ev('torch-detach','Tensor.detach API and shared-storage note','Disconnects graph but shares storage; cannot alone guarantee immutability.'),ev('openai-ppo-source','PPOBuffer.store lines 30–40; compute_loss_pi 227–233','Historical logp stored independently; not recomputed as current logp in denominator.'),ev('repository-rollout','run_posttraining 335–353','Captures old_selected with clone before the repeated epochs; current selected values recomputed per epoch.'),ev('bounded-execution','bounded-cpu.result.json /detached_shares_storage_observed and /saved_clone_retained_history','Detached alias changes with live storage; historical clone remains unchanged.')],['bounded-cpu.stdout.txt','bounded-cpu.result.json','frozen-input/scripts/course_experiments/posttraining.py']),
 claim('C5','concept','Implementations often store log probabilities to handle very small probability values.','13.12 paragraph following code','Log representation postpones probability underflow; it does not guarantee every log difference or exp is numerically safe.',[ev('openai-ppo-source','PPOBuffer.__init__/store 19–40; compute_loss_pi 227–233','Actual implementation stores logp and exponentiates only the log difference.'),ev('torch-log','torch.log API natural-log definition','Positive probabilities can be represented by their natural log.'),ev('torch-exp','torch.exp API exponential definition','Ratio can be computed directly from the difference of retained log values.'),ev('bounded-execution','verify_bounded_cpu.py underflow branch and bounded-cpu.result.json /small_log_ratio','Retained log values -1000,-999 yield exp(1), whereas converting both to float64 probabilities first gives 0/0.')],['bounded-cpu.stdout.txt','bounded-cpu.result.json','openai-spinningup-ppo.py']),
 claim('C6','numeric','A fixed advantage .6 multiplied by ratio 2 gives the surrogate term 1.2, which is not a measured doubling of answer quality.','13.12 advantage paragraph','One selected-action surrogate term, in advantage/reward units; neither training nor quality evaluation occurs.',[ev('ppo-paper','Section 3 equation 6, PDF p3','The objective is the empirical mean of ratio times estimated advantage.'),ev('arithmetic','inspection-ledger.md arithmetic','2*.6=1.2; dimensionless ratio does not by itself measure quality.'),ev('bounded-execution','bounded-cpu.result.json /ratio_times_advantage','Independent product yields first term 1.2.')],['bounded-cpu.stdout.txt','bounded-cpu.result.json'],verification('First unclipped term 1.2','1.2','Advantage .6 is fixed as in 13.11; no optimizer or measured quality comparison.')),
 claim('C7','concept','old sampling policy is refreshed for a new collection round; reference can remain the posttraining starting policy throughout.','13.12 old/reference paragraph and supplementary source note','Roles in this chapter\'s fixed-reference, SFT-to-PPO setting; this is not a universal requirement for every posttraining algorithm.',[ev('ppo-paper','Section 5 Algorithm 1, PDF p5','Reuse collected samples for K epochs, then theta_old←theta before another collection iteration.'),ev('instructgpt-paper','Section 3.2 RL paragraph and equation 2, PDF p9','Learned RL policy regularized against distinct starting SFT policy.'),ev('repository-rollout','run_posttraining 289–290 and 326–353','Frozen independent SFT reference; rollout-specific old_selected snapshot and separate current PPO policy.')],['ppo-1707.06347v2.pdf','instructgpt-2203.02155v1.pdf','frozen-input/scripts/course_experiments/posttraining.py']),
 claim('C8','numeric','Changing first new probability to .3 gives [1.5,.5]; incorrectly dividing by the current new probability gives [1,1].','13.12 exercise paragraph','Positive selected-action probabilities, same context/action in numerator and denominator; wrong denominator erases the likelihood-change statistic.',[ev('arithmetic','inspection-ledger.md arithmetic','.3/.2=1.5 and .1/.2=.5; new/new=1 for both.'),ev('bounded-execution','bounded-cpu.result.json /exercise_ratio and /wrong_denominator_ratio','Independent altered-probability and wrong-denominator branches executed.')],['bounded-cpu.stdout.txt','bounded-cpu.result.json'],verification('[1.5,.5] and [1,1]','[1.4999999999999996,.5000000000000001] and [1.0,1.0]','Both entries are checked separately at 1e-12 tolerance.')),
 claim('C9','concept','PPO clipping limits the incentive from continuing to push an old sample in a favorable direction.','13.12 advantage paragraph: next-section clipping preview','Statement concerns the clipped objective\'s incentive, not a hard bound on every final policy probability.',[ev('ppo-paper','Section 3 equation 7 and explanation immediately below, PDF p3','Min of unclipped/clipped surrogate removes incentive beyond the range only when the change would improve the objective.'),ev('repository-helper','ppo_clipped_objective lines 37–47','Actual helper forms ratio*fixed_advantage and the minimum clipped/unclipped surrogate.')],['ppo-1707.06347v2.pdf','frozen-input/tiny_perceptron/posttraining.py']),
]
ids=[c['id'] for c in claims]
report={
 'schema_version':1,'review_stage':'technical','lesson_id':'13.12','source':'course/chapters/13.md#13.12',
 'source_sha256':sha(frozen),'verdict':'pass','reviewer_task':'/root/phase4_factual_coordinator/factual_13_12',
 'reviewer_context':'fresh','author_tasks':[],'figure_sha256':{},'reviewed_on':accessed,
 'read_scope':{'section':'complete 13.12, original UTF-8 bytes from line '+str(first_line),'prerequisites':['13.4 complete original section for reference definition','13.11 complete original section for fixed advantage and contextual bandit'],'frozen_chapter_input':{'path':PREFIX+'/frozen-input/chapter13-frozen-input.md','sha256':sha(DEST/'frozen-input/chapter13-frozen-input.md'),'meaning':'frozen input snapshot at review start, not current whole-chapter version'},'introduction':'not chapter-first section; no introduction review required'},
 'independence':{'prior_review_conclusions_read':False,'author_correction_summaries_read':False,'exposure_event':None,'filename_inventory':'Initial broad rg --files printed filenames only, no contents; only named original authority/source/method files were opened.'},
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[],
 'checks':{
  'factual_accuracy':{'status':'pass','claim_ids':ids,'details':'Nine substantive claims personally verified against PPO original formulas/algorithm, original InstructGPT reference equation, official PyTorch APIs, original OpenAI source and exact repository contracts.'},
  'numeric_verification':{'status':'pass','claim_ids':['C2','C3','C6','C8'],'details':'Unchanged fence and independent positive-probability variants executed on CPU; samples=2, action-axis construction separately normalized, ratios dimensionless, target in advantage units, float64 tolerance 1e-12 and original rounded output exact.'},
  'figure_consistency':{'status':'not_applicable','claim_ids':[],'details':'No referenced figure. The selected entries need no spatial diagram. Actually rendered and opened desktop/mobile isolated frozen-section screenshots; labels and numbers visible without horizontal overflow. Final built course theme not checked.'},
  'source_verification':{'status':'pass','claim_ids':ids,'details':'Personally read exact paper versions and APIs and original implementation ranges. Immutable locators used only to find original PDFs. Original files, HTTPS versioned URLs, fetch dates, commit, precise locators and own support record permanently retained.'},
  'limitations':{'status':'pass','claim_ids':ids,'details':'Arithmetic and bounded helper/gradient checks only: no optimizer updates, full recipe, model download, GPU training, existing-model reevaluation, deployment or quality-improvement claim. Version gap documented: APIs 2.9, installed PyTorch 2.14.1+cpu. First Chromium file URL blocked by administrator; preserved error, then actual set_content render succeeded. Course theme not verified.'},
 },
 'limitations':['Current example is two finite-card contextual-bandit selections, not a token-level LM RLHF implementation.','Original fence does arithmetic only; enabled-gradient variant verifies detachment separately.','Numerical log-probability example has positive recorded selected probabilities; no guarantee against all underflow/overflow.','Final course site theme and expanded supplementary details were not browser-inspected; complete supplement source was read.'],
}
canonical=ROOT/'docs/technical-reviews/13.12.json'
dump(canonical,report)
# Confirm that this exact new object was written before any separate checker command.
assert json.loads(canonical.read_bytes())==report
dump(DEST/'canonical-written.json',{'path':canonical.relative_to(ROOT).as_posix(),'sha256':sha(canonical),'source_sha256':report['source_sha256'],'verdict':report['verdict'],'claims':len(claims),'written_and_readback_confirmed':True})
print(json.dumps({'canonical':str(canonical),'report_sha256':sha(canonical),'source_sha256':report['source_sha256'],'verdict':report['verdict'],'claims':len(claims),'artifacts':len(artifacts)},ensure_ascii=False))
