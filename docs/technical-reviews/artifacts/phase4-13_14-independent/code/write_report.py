import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = 'docs/technical-reviews/artifacts/phase4-13_14-independent'
A = ROOT / BASE
def digest(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
env = {'python':'3.13.5', 'torch':'2.14.1+cpu', 'device':'cpu', 'cuda_build':'None'}
artifacts = []
def artifact(identifier, filename, kind, description, command=None, result=None):
    path = BASE + '/' + filename
    entry = {'id':identifier,'path':path,'sha256':digest(path),'kind':kind,'description':description}
    if kind == 'execution': entry.update(command=command, result=result, environment=env)
    artifacts.append(entry)
for identifier, filename, kind, description in [
 ('section','frozen-input/section.md','source_snapshot','Original UTF-8 section bytes, including heading; no normalization.'),
 ('frozen-chapter','frozen-input/chapter-13-frozen.md','source_snapshot','Initial frozen full chapter only, not current whole-chapter assertion.'),
 ('extraction','frozen-input/extraction.json','source_snapshot','Actual extraction facts, fence/source/figure hashes and line numbers.'),
 ('fence1','code/fence-1.py','code','Unchanged first original Python fence.'),
 ('fence2','code/fence-2.py','code','Unchanged second original Python fence.'),
 ('helper','frozen-input/section_facts.py','code','Inspected actual extraction/execution helper.'),
 ('bootstrap','code/bootstrap.py','code','Original extracted bootstrap used for source fence execution.'),
 ('posttraining-code','code/posttraining.py','code','Immutable complete inspected finite-model helper source.'),
 ('recipe-code','code/course-experiment-posttraining.py','code','Immutable complete original experiment source; inspected computation ranges recorded separately.'),
 ('bounded-code','code/check.py','code','Independent bounded numeric, gradient, shape, network update and original measurement check.'),
 ('writer-code','code/write_report.py','code','Complete report generator; canonical replacement does not read prior report.'),
 ('own-support','own-support.md','derivation','Personal derivations, primary source support, read scope and explicit limitations.'),
 ('commands','execution/commands.json','source_snapshot','Actual invoked commands and inspection mechanisms.'),
 ('original-stdout','execution/stdout.txt','source_snapshot','Actual stdout of both source fences.'),
 ('original-stderr','execution/stderr.txt','source_snapshot','Actual empty stderr of successful source fence run.'),
 ('original-env','execution/environment.json','source_snapshot','Actual CPU runtime, attempted fences, guard events and module hashes.'),
 ('bounded-stderr','execution/check.stderr.txt','source_snapshot','Actual empty stderr of bounded successful check.'),
 ('bounded-exit','execution/check.exit.txt','source_snapshot','Actual bounded-check exit code zero.'),
 ('initial-attempt','execution/initial-worker-attempt.json','source_snapshot','Honest record of initial direct worker invocation failure and correction; not raw traceback.'),
 ('raw-results','frozen-input/original-posttraining-results.json','source_snapshot','Full original unmodified result JSON, including untouched annotations; only specified raw pointers inspected.'),
 ('pointers','frozen-input/measurement-pointers.json','source_snapshot','Exact inspected measurement/provenance JSON pointers and discovery-only key/type maps.'),
 ('read-ranges','frozen-input/read-ranges.json','source_snapshot','Personally inspected lesson and implementation ranges; explicit non-dependence on 13.9.'),
 ('svg','figure/model-roles.svg','source_snapshot','Original section SVG bytes.'),
 ('render640','figure/model-roles-640.png','figure_render','Actual Inkscape 640px rendering, loaded with view_image.'),
 ('render390','figure/model-roles-390.png','figure_render','Actual Inkscape 390px rendering, loaded with view_image.'),
 ('visual-note','figure/inspection.txt','derivation','Personal visual observations and boundary of SVG versus browser layout checks.'),
 ('chromium-status','figure/chromium.exit.txt','source_snapshot','Actual Chromium timeout exit 124.'),
 ('chromium-stderr','figure/chromium.stderr.txt','source_snapshot','Actual Chromium stderr from timed-out attempt.'),
 ('inkscape-stderr','figure/inkscape.stderr.txt','source_snapshot','Actual Inkscape render stderr.'),
 ('ppo-pdf','sources/ppo-arxiv-1707.06347v2.pdf','source_snapshot','Original primary PPO v2 PDF, personally read at header and §5.'),
 ('dpo-pdf','sources/dpo-arxiv-2305.18290v3.pdf','source_snapshot','Original primary DPO v3 PDF, personally read at header and §3 Eq3.'),
 ('instructgpt-pdf','sources/instructgpt-arxiv-2203.02155v1.pdf','source_snapshot','Original primary InstructGPT v1 PDF, personally read at header and RM/RL methods.'),
 ('ppo-text','sources/ppo.txt','source_snapshot','pdftotext layout extraction of original PPO paper.'),
 ('dpo-text','sources/dpo.txt','source_snapshot','pdftotext layout extraction of original DPO paper.'),
 ('instructgpt-text','sources/instructgpt.txt','source_snapshot','pdftotext layout extraction of original InstructGPT paper.'),
 ('paper-authority','sources/authority-fetch.json','source_snapshot','Actual HTTP200 arxiv exact-version-page fetch records.'),
 ('ppo-abs','sources/ppo-arxiv-abs.html','source_snapshot','Personally inspected exact v2 arxiv title/author/version history.'),
 ('dpo-abs','sources/dpo-arxiv-abs.html','source_snapshot','Personally inspected exact v3 arxiv title/author/version history.'),
 ('torch-loss','sources/torch-loss-v2.9.0.py','source_snapshot','Personally inspected official PyTorch v2.9.0 loss source/docstrings.'),
 ('torch-functional','sources/torch-functional-v2.9.0.py','source_snapshot','Personally inspected official PyTorch v2.9.0 softmax/log_softmax source/docstrings.'),
 ('torch-autograd','sources/torch-autograd-v2.9.0.py','source_snapshot','Personally inspected official PyTorch v2.9.0 backward source/docstrings.'),
 ('torch-fetch','sources/torch-source-fetch.json','source_snapshot','Actual pinned upstream HTTP200 fetches, hashes and AST ranges.')
]: artifact(identifier,filename,kind,description)
artifact('original-execution','execution/execution.json','execution','Actual original-fence child process facts.',
 '/workspace/tiny-perceptron-vlm/.venv/bin/python -I /workspace/tiny-perceptron-vlm/docs/review-tools/section_facts.py --worker /workspace/tiny-perceptron-vlm/outputs/reviewer-tools/phase4-13_14-independent',
 'exit=0; attempted fences=[1,2]; stdout=0.36 -1.2 twice, KL=0.1927; no guard events.')
artifact('bounded-execution','execution/check.stdout.json','execution','Actual numeric variants, two bounded optimizer epochs and original raw-trace arithmetic.',
 "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python " + BASE + '/code/check.py',
 'exit=0; all assertions pass; values .36/−1.2 and .16/+.8; KL=.19274475702175736 and 0; policy/critic changed; fixed roles/records unchanged; 64 samples×3 original reuse epochs checked.')

sources=[]
def primary(identifier,kind,title,url,version,reason,note):
 sources.append({'id':identifier,'kind':kind,'title':title,'url':url,'version':version,
 'authority_reason':reason,'verified':True,'checked_original':True,'accessed_on':'2026-10-05','inspection_note':note})
primary('ppo','paper','Proximal Policy Optimization Algorithms','https://arxiv.org/pdf/1707.06347v2','arXiv:1707.06347v2, 2017-08-28',
 'Original PPO paper by Schulman et al., OpenAI, on the authors’ arxiv publication record.',
 'Personally checked original PDF header/title/authors/version, §5 PDF pp4–5 Eq9–12 and Algorithm1; arxiv exact-v2 title/author/version page independently fetched HTTP200 and inspected.')
primary('dpo','paper','Direct Preference Optimization: Your Language Model is Secretly a Reward Model','https://arxiv.org/pdf/2305.18290v3','arXiv:2305.18290v3, 2024-07-29',
 'Original DPO paper by Rafailov et al. on the authors’ arxiv publication record.',
 'Personally checked original PDF header/title/authors/version and §3 PDF pp3–4 Eq3 plus following reference-policy definition; exact-v3 arxiv title/author/version page independently fetched HTTP200 and inspected.')
primary('instructgpt','paper','Training language models to follow instructions with human feedback','https://arxiv.org/pdf/2203.02155v1','arXiv:2203.02155v1, 2022-03-04',
 'Original InstructGPT methods paper by Ouyang et al., OpenAI.',
 'Personally checked original PDF header/title/authors/version, §3.5 PDF pp8–9 Eq1–2, reward prompt/completion inputs and per-token KL; used only as primary support for RLHF role distinction and scope boundary.')
for identifier,title,path,note in [
 ('torch-loss-source','PyTorch MSELoss and KLDivLoss official source','torch/nn/modules/loss.py','Personally read original source/docstrings KLDivLoss lines465–512 and MSELoss568–608, including KL direction and reduction denominators.'),
 ('torch-functional-source','PyTorch softmax and log_softmax official source','torch/nn/functional.py','Personally read original softmax2096–2136 and log_softmax2213–2244, formula, dimension and log equivalence.'),
 ('torch-autograd-source','PyTorch backward official source','torch/autograd/__init__.py','Personally read original backward243–290: graph differentiation and leaf gradient accumulation, not optimizer update.')
]: primary(identifier,'official_source',title,'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/'+path,'PyTorch v2.9.0 tag',
 'Official pytorch/pytorch upstream source pinned to release tag; fetched HTTP200.',
 note+' Installed runtime is 2.14.1+cpu; these stable API semantics were separately executed in the installed CPU runtime. docs.pytorch.org endpoints returned403; the pinned upstream source was inspected instead.')
for identifier,path,title,note in [
 ('helper-code','tiny_perceptron/posttraining.py','Finite candidate policy/reward/value and exact KL helpers','AST names/ranges located first; personally read1–86, especially exact_kl15–21, bandit_advantage24–28 and model classes53–86.'),
 ('recipe','scripts/course_experiments/posttraining.py','Original finite posttraining experiment recipe','AST names/ranges located first; personally read31–55,151–162,256–430: fixed reference/reward, cloned old logs/values, fixed advantages, separate policy KL and RM-only value target.'),
 ('measurements',BASE+'/frozen-input/original-posttraining-results.json','Original posttraining result measurements','Full original JSON retained unmodified; only measurement-pointers.json named raw trace/provenance/parameter/hash pointers inspected. No attached commentary values read.')
]: sources.append({'id':identifier,'kind':'repository_code','title':title,'path':path,'sha256':digest(path),
    'version':'Original bytes at independent 2026-10-05 review; hashes match original result code provenance where applicable.',
    'verified':True,'inspection_note':note})
sources.extend([
 {'id':'derivation','kind':'derivation','title':'Independent MSE expectation, gradients and enumerated KL calculation','verified':True,
  'details':'E[(v−R)^2] derivative2(v−E[R]) gives expectation; N=1 produces (v−1)^2 and2(v−1); KL=.8ln1.6+.2ln.4=.19274475702175753 nats; all axis, denominator and unit details are in own-support.md.'},
 {'id':'fence-run','kind':'execution','title':'Original two Python fences CPU execution','verified':True,'artifact_id':'original-execution'},
 {'id':'bounded-run','kind':'execution','title':'Independent CPU variants, contracts and raw measurements','verified':True,'artifact_id':'bounded-execution'}
])
def ev(source,locator,supports): return {'source_id':source,'locator':locator,'supports':supports}
claims=[]
def claim(identifier,kind,statement,location,scope,evidence,artifact_ids,verification=None):
 c={'id':identifier,'kind':kind,'statement':statement,'location':location,'scope':scope,'status':'verified','evidence':evidence,'artifact_ids':artifact_ids}
 if verification: c['verification']=verification
 claims.append(c)
def verification(expected,observed,details,tolerance=None,denominators=None):
 v={'method':'executed','expected':expected,'observed':observed,'details':details}
 if tolerance: v['tolerance']=tolerance
 if denominators: v['denominators']=denominators
 return v
claim('roles','concept','單步選卡的critic只看情境估計策略的平均結果；reward看情境和卡評分，兩者問題不同。','13.14 opening paragraph and figure reward/critic cards',
 'Conditional expected RM reward before the finite terminal action; not a new preference assessor or token-level future-return claim.',
 [ev('ppo','§5 pp4–5 Eq9–12','State-value estimate and a separate value error term.'),ev('instructgpt','§3.5 pp8–9 Eq1, RL paragraph','RM r_theta(x,y) scores prompt/completion, with separate value function.'),ev('helper-code','FiniteRewardModel64–75; FiniteValueModel78–86','Reward includes candidate identity; value input includes only context features.'),ev('derivation','own-support.md expectation derivation','Squared-error conditional optimum is mean sampled reward.')],['own-support','posttraining-code'])
claim('value-target','concept','以觀察到的分數作critic平方代價目標；多次不同選擇才能有機會估平均，一次1分不能證明每次1分。','13.14 second paragraph',
 'Illustrative single-step target under the sampling policy, without a convergence or trained-quality guarantee.',
 [ev('ppo','§5 Eq9 pp4–5','Squared value-target error is part of PPO actor-critic training.'),ev('derivation','E[(v−R)^2] derivative2(v−E[R])','An individual observed reward is a noisy target; the expected optimum is its conditional mean.')],['own-support'])
claim('mse-numbers','numeric','value=.4時代價.36、梯度−1.2；value=1.4時代價.16、梯度+.8；下降方向分別升高與降低。','13.14 first fence, following paragraph and exercise',
 'One scalar prediction and one target1, MSE mean denominator N=1; computation/gradient only.',
 [ev('derivation','L=(v−1)^2; dL/dv=2(v−1)','Exact numbers and descending sign.'),ev('fence-run','stdout lines1–2','Both original examples print .36 and−1.2.'),ev('bounded-run','/mse','Both original and changed initial predictions executed.')],
 ['original-execution','bounded-execution','fence1','fence2','own-support'],verification('.36/−1.2 and .16/+.8','Float32 .3600000143/−1.2000000477 and .1599999815/.7999999523','Target1, no implicit .5 coefficient; mean over one element. v_new=v−ηgrad gives stated directions.','absolute1e−6; displayed4-digit rounding exact'))
claim('gradient-only','software','本節兩段value程式只計梯度，沒有更新網路；此檢查不能證明回答進步。','13.14 after first fence and exercise',
 'backward accumulates gradient and code has no optimizer step or parameter assignment; no answering evaluation.',
 [ev('torch-autograd-source','autograd.backward243–290','Computes and accumulates graph-leaf gradients.'),ev('fence-run','attempted_fences[1,2], stdout','Actual original fence execution.'),ev('bounded-run','/mse/*/value_unchanged_after_backward','Values remain unchanged after backward.')],
 ['original-execution','bounded-execution','fence1','fence2'],verification('Gradients populated; original values unchanged','Both initial values unchanged after backward, and original fences contain no step','Executed original fences and fresh leaf variants; no training-quality measurement performed.'))
claim('old-and-reference','concept','PPO以收集時保存的critic值計固定優勢；old同輪固定下輪更換，reference是後訓練的固定SFT起點。','13.14 after first fence, figure old/reference cards and following paragraph',
 'Lesson recipe: old-policy rollout records versus fixed post-SFT reference; old is not another persistent full model.',
 [ev('ppo','§5 Algorithm1 pp5; Eq10–12','Collect samples/advantages before K optimization epochs, then replace old policy.'),ev('dpo','§3 Eq3 and following paragraph pp3–4','Reference is initial SFT policy distinct from trainable policy.'),ev('recipe','reference289; old collection338–345; reuse347–380','Reference frozen, old values/logs cloned and reused with fixed advantages.')],['recipe-code','own-support'])
claim('small-networks','software','policy/critic在PPO更新，reward/reference固定；PPO四個小網路加舊紀錄，不代表五個完整LLM。','13.14 figure and following paragraph',
 'Four roles in the PPO branch: current policy, reference, reward and critic. Separate DPO branch is not included; these are finite MLPs, not autoregressive LMs.',
 [ev('helper-code','classes53–86','Small finite networks produce4 candidate logits/scores or1 context value.'),ev('recipe','289–354','Frozen reference/reward and separately optimized policy/value.'),ev('bounded-run','/bounded_recipe','Parameter counts and update/freeze contracts executed.')],
 ['bounded-execution','posttraining-code','recipe-code','render640','render390','visual-note'],verification('Policy/critic change; reward/reference/old data fixed; finite small parameter counts','Two bounded epochs: policy/critic changed; fixed roles unchanged; policy148/reward241/value97','3 artificial contexts×4 candidates×2 epochs using new networks and original _update; no loaded weights or existing model re-evaluation.'))
claim('kl-numbers','numeric','KL([.8,.2]||[.5,.5])=Σp_i ln(p_i/q_i)約.1927；p=q時0，完整和涵蓋所有卡。','13.14 KL paragraph, exercise and second fence',
 'Natural-log KL in nats, candidate axis summed; all enumerated finite options, not one sampled-action log ratio.',
 [ev('torch-loss-source','KLDivLoss465–503','Pointwise p(log p−log q) and sum/reduction distinction; helper uses p weighting and q reference in correct direction.'),ev('derivation','own-support.md exact sum','Two candidate terms sum .19274475702175753 nats.'),ev('fence-run','stdout line3','Original fence displays .1927.'),ev('bounded-run','/kl','Float64 exact candidate-axis KL and identical-distribution zero.')],
 ['original-execution','bounded-execution','own-support'],verification('.19274475702175753 and0','Float64 .19274475702175736 and0; displayed original float32 .1927','Two contexts×two candidates; sum(-1) outputs shape(2,), not mean over candidates.','Float64 absolute1e−12; displayed4-digit rounding exact'))
claim('exact-kl-contract','software','exact_kl接收候選logits，log_softmax還原分布；指定機率先取log，reference不接受梯度。','13.14 details intro and second fence',
 '2D (context,candidate) equally shaped logits; positive normalized probability example; not arbitrary zero-probability input handling.',
 [ev('helper-code','exact_kl15–21','Shape guards, log_softmax, reference detach, weighted candidate sum.'),ev('torch-functional-source','softmax2096–2136; log_softmax2213–2244','Softmax normalized exponentials and log_softmax equals log(softmax).'),ev('bounded-run','/kl and /invalid_shapes','Gradient, shift invariance and rejection of invalid shapes executed.')],
 ['bounded-execution','posttraining-code','fence2'],verification('log(p) recoversp; per-context KL; no reference gradient; invalid shapes reject','Numerically correct KL, shift invariant; logits grads present, reference gradNone; two ValueErrors','Tested 2 contexts×2 candidates, same-distribution row, independent logit shifts,1D input and shape mismatch.'))
claim('reference-objective','concept','參考偏離量乘係數成额外代價；DPO論文§3公式3呈現獎勵減相對reference的KL目標。','13.14 final details paragraph',
 'RLHF objective cited from the DPO paper, not a claim that this penalty is the DPO preference loss or that every LLM has identical feedback.',
 [ev('dpo','§3 Eq3 pp3–4','max E[r_phi]−βDKL(policy||reference), β controls departure from initial SFT reference.'),ev('instructgpt','§3.5 p9 RL paragraph and Eq2','Per-token KL implementation gives primary support for stated scope distinction.')],['dpo-pdf','instructgpt-pdf','own-support'])
claim('recipe-kl-target','software','本章有限選項PPO把完整reference KL另加在策略代價，critic仍估計獎勵模型分數。','13.14 final two sentences',
 'Inspected specific finite recipe only: sampled centered/scaled RM target, exact full-option KL penalty; not every token-level RLHF recipe.',
 [ev('recipe','_normalized_scores160–162; rollout338–354','Sample normalized RM scores; clipped loss plus.1 mean exact KL; separate MSE against rewards.'),ev('bounded-run','/bounded_recipe','Same separation and update/freeze behavior reproduced in bounded independent CPU check.')],
 ['recipe-code','bounded-execution','own-support'],verification('Policy loss includes separate KL; critic target is sampled normalized RM score only','Bounded check executes surrogate+.1KL and MSE(critic,reward) separately, changing only intended networks','Original computation personally inspected; independent two-epoch contracts do not rerun the full recipe or prove learned answering quality.'))
claim('original-records','empirical','既有PPO原紀錄支持同輪保存old log、old value和優勢；reference前後未改。','13.14 recipe role and fixed-time explanation; supplemental original measurement support',
 'Evidence arithmetic on existing original run, not a new training run or performance evaluation; 64 first-rollout samples reused3 times.',
 [ev('measurements','Named raw pointers in measurement-pointers.json','First-rollout raw arrays and before/after reference state hashes.'),ev('bounded-run','/existing_measurements','Original array invariance and advantage/ratio arithmetic independently recomputed.')],
 ['bounded-execution','raw-results','pointers'],verification('Fixed arrays across3 reuse epochs; A=R−V_old and ratio=exp(newlog−oldlog); equal reference hashes','All fixed arrays equal; max advantage error5.96046e−8, ratio error6.08395e−8; reference hashes equal','Explicit raw/provenance pointers only; implementation SHA provenance matches inspected code; no commentary values used.',denominators={'rollout_samples':64,'reuse_epochs':3,'candidate_options':4}))

source_path = ROOT/'course/chapters/13.md'
raw=source_path.read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw));i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 13.14 '))
body=raw[heads[i].start():heads[i+1].start()]
assert hashlib.sha256(body).hexdigest() == '93a03d7a64e34c60439efa889e2739a37127ed98fd2337110144534aac2bab78'
assert digest('course/figures/rewrite-13-model-roles.svg') == 'b18bbd67c517b81f3f215755d70c948156283de29aac2aa4f1a068381ec53459'
report={'schema_version':1,'review_stage':'technical','lesson_id':'13.14','source':'course/chapters/13.md#13.14',
 'source_sha256':hashlib.sha256(body).hexdigest(),'reviewer_task':'/root/phase4_factual_coordinator/factual_13_14','reviewer_context':'fresh','verdict':'pass',
 'reviewed_on':'2026-10-05','figure_sha256':{'course/figures/rewrite-13-model-roles.svg':digest('course/figures/rewrite-13-model-roles.svg')},
 'frozen_input':{'path':BASE+'/frozen-input/chapter-13-frozen.md','sha256':digest(BASE+'/frozen-input/chapter-13-frozen.md'),
 'meaning':'Initial frozen full chapter snapshot only; current source_sha256 is the original UTF-8 bytes of13.14. Later unrelated13.9 edit was not read or relied upon.'},
 'read_scope':{'section':'13.14 complete original text, both fences and SVG','prerequisites':['13.11','13.12','13.13'],'implementation_ranges':'Recorded in read-ranges.json; AST names/ranges inspected before computation reads.',
 'independence':'New reviewer; no old technical/reader/history conclusions or attached author result commentary read; immutable indexes used only to locate original papers.'},
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[],
 'checks':{
 'factual_accuracy':{'status':'pass','details':'Primary papers and original finite implementation support every substantive role, saved baseline, MSE and reference-objective claim within the single-step scope.','claim_ids':[c['id'] for c in claims]},
 'numeric_verification':{'status':'pass','details':'Both source fences and required value/KL variants executed; mean denominator1, candidate sum axis and nats checked; existing64×3 trace arithmetic independently recomputed.','claim_ids':['mse-numbers','kl-numbers','original-records']},
 'figure_consistency':{'status':'pass','details':'Original SVG hash checked; actual Inkscape640/390 PNGs loaded with view_image and inspected. All five input/output/fixed-time role cards agree with text and recipe. Chromium20s timeout124 recorded; webpage browser layout unverified.','claim_ids':['roles','old-and-reference','small-networks']},
 'source_verification':{'status':'pass','details':'Personally read original versioned PPO v2, DPO v3 and InstructGPT v1 PDFs; PPO/DPO arxiv version pages fetched and inspected. Original upstream PyTorch v2.9.0 API source fetched and read, installed2.14.1+cpu executed. Full code/raw JSON provenance and snapshot SHA preserved.','claim_ids':[c['id'] for c in claims]},
 'limitations':{'status':'pass','details':'Gradient demonstrations and two artificial update epochs establish math/contracts only. Existing trace inspection is not rerun training or model evaluation. Finite4-candidate RM-only critic target plus exact KL is specific to this recipe; no universal token-level RLHF or answering-improvement assertion. Browser page layout unverified; source-doc403 and initial helper worker limitation honestly recorded and corrected for the executed fences.','claim_ids':['gradient-only','small-networks','reference-objective','recipe-kl-target','original-records']}}}
serialized=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
(A/'new-report.json').write_bytes(serialized)
canonical=ROOT/'docs/technical-reviews/13.14.json'
canonical.write_bytes(serialized)
assert canonical.read_bytes() == serialized
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],'report_sha256':hashlib.sha256(serialized).hexdigest(),'canonical':str(canonical),'artifacts':len(artifacts),'claims':len(claims)},ensure_ascii=False))
