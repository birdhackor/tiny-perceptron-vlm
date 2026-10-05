import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-13_11-independent'
REL=BASE.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((BASE/'input-manifest.json').read_text())
spec=importlib.util.spec_from_file_location('section_facts',ROOT/'docs/review-tools/section_facts.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
current,_,_=helper.original_section(ROOT/'course/chapters/13.md','13.11')
assert current==(BASE/'inputs/section.md').read_bytes(), 'Section changed; do not publish a stale verdict'
fenv=json.loads((BASE/'original-fence-execution/environment.json').read_text())
cpu=json.loads((BASE/'cpu-variants-result.json').read_text())
render=json.loads((BASE/'render-result.json').read_text())
detach=json.loads((BASE/'detach-source-acquisition.json').read_text())
assert render['status']=='rendered' and detach['status']=='fetched'
env={'python':fenv['python'],'torch':fenv['torch'],'device':'cpu','cuda_build':fenv['cuda_build'],
 'cuda_available':fenv['cuda_available'],'torch_git_version':fenv['torch_git_version']}
artifacts=[]
def artifact(id,path,kind,description,**extra):
 artifacts.append({'id':id,'path':REL+'/'+path,'sha256':sha(BASE/path),'kind':kind,'description':description,**extra})
artifact('section','inputs/section.md','source_snapshot','Original UTF-8 bytes of 13.11, without newline normalization.')
artifact('frozen-chapter','inputs/13.md','source_snapshot','Frozen full chapter input used at review start; this hash does not describe later chapter revisions.')
artifact('input-manifest','input-manifest.json','source_snapshot','Exact source-byte and named input snapshot hashes.')
artifact('original-fence','original-fence-execution/fence-1.py','code','Exact original Python fence; no replacement or repair.')
artifact('fence-execution','original-fence-execution/execution.json','execution','Actual original fence process status and stdout/environment hashes.',
 command='.venv/bin/python docs/review-tools/section_facts.py course/chapters/13.md#13.11 --output /tmp/phase4-13_11-cpu --execute --timeout 30',
 result='exit 0; one original fence executed; stdout [0.6, -0.4] and False; no guard events.',environment=env)
artifact('fence-stdout','original-fence-execution/stdout.txt','source_snapshot','Actual original fence stdout.')
artifact('fence-environment','original-fence-execution/environment.json','source_snapshot','Actual installed version, CPU device and loaded-module fingerprints.')
artifact('fence-bootstrap','original-fence-execution/bootstrap.py','code','Exact generic course bootstrap used by the worker.')
artifact('variants-code','cpu_variants.py','code','Independent bounded CPU assertions for arithmetic, detach, PPO and separate actor/value updates.')
artifact('variants-execution','cpu-variants-result.json','execution','Actual synthetic checks and raw float32 measurements.',
 command='.venv/bin/python docs/technical-reviews/artifacts/phase4-13_11-independent/cpu_variants.py > docs/technical-reviews/artifacts/phase4-13_11-independent/cpu-variants-stdout.txt 2> docs/technical-reviews/artifacts/phase4-13_11-independent/cpu-variants-stderr.txt',
 result='exit 0; all arithmetic/shape/autograd assertions passed; actor preserved value parameters, separate MSE critic step changed them.',environment=env)
artifact('variants-stdout','cpu-variants-stdout.txt','source_snapshot','Actual independent CPU script stdout.')
artifact('method-inspection-code','inspect_original.py','code','AST-based original method selection and exact range inspection.')
artifact('method-inspection','original-method-inspection-stdout.txt','source_snapshot','Personally inspected original helper/model/update/scaling/rollout methods, excluding later result explanations.')
artifact('helper-snapshot','inputs/tiny_perceptron/posttraining.py','code','Exact helper and finite model implementation bytes; only stated method ranges were read.')
artifact('experiment-snapshot','inputs/scripts/course_experiments/posttraining.py','code','Exact original experiment source bytes; method ranges only, no full training or model loading.')
artifact('own-derivation','derivation-and-inspection.md','derivation','Own arithmetic, units, baseline/terminal reduction, original source support and limits.')
for id,name in [('ppo','ppo-v2'),('gae','gae-v6'),('instructgpt','instructgpt-v1')]:
 artifact(id+'-pdf','sources/'+name+'.pdf','source_snapshot','Exact original versioned paper bytes personally checked.')
 artifact(id+'-text','sources/'+name+'.txt','source_snapshot','pdftotext extraction of the original paper; locators in report refer to this file.')
artifact('pytorch-source','sources/pytorch-v2.9.0-_tensor.py','source_snapshot','First-party tagged v2.9.0 Tensor.detach source contract; differs from executing 2.14.1+cpu.')
artifact('source-acquisition','source-acquisition.json','source_snapshot','Versioned HTTPS URL, acquisition and original PDF hashes.')
artifact('source-locators','original-paper-locator-provenance.json','source_snapshot','Lookup-only immutable index pointers; original content verified independently.')
artifact('source-acquisition-code','acquire_sources.py','code','Actual original-PDF copy/fetch and pdftotext commands.')
artifact('pytorch-acquisition','detach-source-acquisition.json','source_snapshot','Actual official tagged source HTTPS URL/version/hash.')
artifact('pytorch-acquisition-code','fetch_detach_source.py','code','Actual first-party tagged source acquisition and exact inspection range.')
artifact('pytorch-docs-failure','detach-docs-stderr.txt','source_snapshot','Actual optional docs-host HTTP 403 failure, not cited as verified evidence.')
artifact('render-code','render_section.py','code','Actual self-contained section HTML and two-viewport Chromium render method.')
artifact('render-html','section-render.html','source_snapshot','Exact section HTML passed to Chromium page.set_content.')
artifact('render-execution','render-result.json','execution','Successful two-viewport render dimensions and renderer version.',
 command=render['command'],result='rendered; desktop 1280x800 and mobile 390x844; both screenshots personally viewed.',
 environment={'renderer':render['environment']['renderer'],'chromium':render['environment']['chromium_version'],'device':'cpu'})
artifact('desktop-render','desktop.png','figure_render','Personally viewed desktop rendering of exact section Markdown; no source figure exists.')
artifact('mobile-render','mobile.png','figure_render','Personally viewed mobile rendering; text wraps and code block scrolls horizontally.')
artifact('render-initial-failure','render-file-url-attempt-result.json','source_snapshot','Actual Chromium file-URL policy failure; not a timeout, preserved before successful set_content render.')
artifact('commands','commands.json','source_snapshot','Actual commands, exit statuses and failure/recovery boundaries.')
artifact('report-generator','write_report.py','code','Complete independent new report generator; does not read the old canonical report.')

sources=[]
def paper(id,title,url,version,inspection):
 sources.append({'id':id,'kind':'paper','title':title,'url':url,'version':version,'accessed_on':'2026-10-05',
  'authority_reason':'Original research paper by its method authors, versioned on arXiv.',
  'verified':True,'checked_original':True,'inspection_note':inspection})
paper('gae','High-Dimensional Continuous Control Using Generalized Advantage Estimation','https://arxiv.org/pdf/1506.02438v6','arXiv:1506.02438v6, 20 October 2018; ICLR 2016 paper',
 'Personally verified first-page authors/version and read §1, §2 Eq.(1)–(3), advantage-sign interpretation, §3 Eq.(10)–(16); saved gae-v6.txt lines 1–175 and 232–336. The authority supports policy/baseline/relative advantage, return-minus-baseline reduction and sequential credit assignment; it is not evidence for this repository model performance.')
paper('ppo','Proximal Policy Optimization Algorithms','https://arxiv.org/pdf/1707.06347v2','arXiv:1707.06347v2, 28 August 2017',
 'Personally verified first-page authors/version and read §2.1 Eq.(1)/(2), §3 Eq.(6)/(7), §5 Eq.(9)–(12) and Algorithm 1; saved ppo-v2.txt lines 1–245. Advantage estimates are computed before multiple surrogate epochs, with an old-policy probability denominator; value prediction has its own error objective. No claim of hard probability or KL bounds is made.')
paper('instructgpt','Training language models to follow instructions with human feedback','https://arxiv.org/pdf/2203.02155v1','arXiv:2203.02155v1, 4 March 2022',
 'Personally verified first-page title/authors/version and read §3.5 RM scalar scores/Eq.(1)/normalization and prompt-completion bandit, per-token KL, §C.4 generalized advantage and value-function training; saved instructgpt-v1.txt lines 1–39, 425–507 and 2319–2335. Whole-completion bandit formulations coexist with token-level credit treatment; the finite-card example is not full LM RLHF.')
sources.append({'id':'pytorch-detach','kind':'official_source','title':'PyTorch Tensor.detach original source contract',
 'url':'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor.py','version':'Official PyTorch v2.9.0 tag; executing environment is separately 2.14.1+cpu',
 'accessed_on':'2026-10-05','authority_reason':'Tagged original source in the official pytorch/pytorch repository.',
 'verified':True,'checked_original':True,'inspection_note':'Personally read lines 769–785: detach returns a tensor detached from the current graph and never requiring gradient; storage sharing is stated. This is an API contract citation at v2.9.0, not a claim to have browsed 2.14.1 docs; actual installed-version behavior was executed.'})
for id,path,note in [
 ('helper-code','tiny_perceptron/posttraining.py','AST-located and personally read bandit_advantage:24–28, ppo_clipped_objective:31–50, FiniteResponsePolicy:53–61, FiniteRewardModel:64–75, FiniteValueModel:78–86.'),
 ('experiment-code','scripts/course_experiments/posttraining.py','AST-located _update:151–157 and _normalized_scores:160–162; personally read run_posttraining calculation range 315–389, especially sampling/old-value detachment before epoch reuse and separate MSE update. Result-explanation strings outside these ranges were not read; no full experiment was run.')]:
 sources.append({'id':id,'kind':'repository_code','title':path+' original method contract','path':path,'sha256':sha(ROOT/path),
  'version':'Exact original source bytes frozen in this review input manifest','verified':True,'inspection_note':note})
sources.extend([
 {'id':'own-math','kind':'derivation','title':'Independent single-terminal-step advantage reduction and arithmetic','verified':True,
  'details':'With one terminal action, empirical return G=R; A_hat=G−V_old=R−V_old. Score units are shared, no temporal reduction or correctness rate denominator. 1−.4=.6, 0−.4=−.4; 1−.9=.1, 0−.9=−.9. See own-derivation artifact and CPU checks.'},
 {'id':'original-run','kind':'execution','title':'Exact original fence CPU execution','verified':True,'artifact_id':'fence-execution'},
 {'id':'variant-run','kind':'execution','title':'Independent bounded CPU arithmetic/autograd/update checks','verified':True,'artifact_id':'variants-execution'},
])
def evidence(source,locator,support):return {'source_id':source,'locator':locator,'supports':support}
claims=[]
def claim(id,kind,statement,location,scope,refs,arts,verification=None,**extra):
 c={'id':id,'kind':kind,'statement':statement,'location':location,'scope':scope,'status':'verified','evidence':refs,'artifact_ids':arts,**extra}
 if verification:c['verification']=verification
 claims.append(c)
claim('policy-baseline-relative-advantage','concept',
 '策略是按情境選行動的機率規則；本單步例的基準估計更新前策略平均分數，優勢估計是實得分數減基準，正負表示相對預期而非答案正誤。',
 'course/chapters/13.md#13.11 paragraphs 1–2 (frozen lines 350–352)',
 'The baseline is a value estimate under the behavior policy; the sampled advantage is an estimate, not the exact expected A function or guaranteed per-update improvement.',
 [evidence('gae','§2 Eq.(1)–(3), text after Eq.(3); gae-v6.txt:96–156','Defines stochastic policy, expected value and Q−V advantage; better/worse-than-average action update interpretation.'),evidence('own-math','derivation-and-inspection.md paragraphs 2–3','Terminal one-action sampled return reduces the estimator to R−V_old.')],
 ['gae-pdf','gae-text','own-derivation','variants-execution'])
claim('advantage-numbers','numeric','1/0 rewards with .4 baseline give [.6,−.4]; replacing only the baseline by .9 gives [.1,−.9].',
 'course/chapters/13.md#13.11 fence and exercise (frozen lines 358–365,371)',
 'Two sampled choices, shape [2], per-element score units; no success-rate or token denominator. Float32 raw numbers differ slightly before two-decimal display.',
 [evidence('own-math','R−V_old direct four subtractions','Checks numbers, units and lack of reduction.'),evidence('original-run','stdout.txt lines 1–2; execution.json exit_code','Exact original rounded output.'),evidence('variant-run','cpu-variants-result.json /base and /exercise','Independently asserts original and changed-baseline arithmetic.')],
 ['own-derivation','original-fence','fence-execution','fence-stdout','variants-code','variants-execution'],
 {'method':'executed','expected':'[0.6, -0.4] and [0.1, -0.9] at two decimals.',
  'observed':'Original stdout [0.6, -0.4]; raw [.6000000238, -.4000000060]; exercise raw [.1000000238, -.8999999762], rounded [.1, -.9].',
  'details':'CPU float32 elementwise assertions; both sample count and vector axis checked.',
  'tolerance':'absolute 1e-7, relative 0 for raw float32; exact equality for two-decimal rounded values.'})
claim('fence-and-detach-contract','software','bandit_advantage calculates reward−old_value and detaches it; the original fence prints False although old_values requires gradients.',
 'course/chapters/13.md#13.11 fence and post-fence paragraph (frozen lines 355–365)',
 'The fence does arithmetic and checks returned autograd metadata; it contains no backward or optimizer step. Detach blocks gradient flow through this advantage computation; it does not freeze all possible shared-parameter paths in arbitrary actor-critic architectures.',
 [evidence('helper-code','bandit_advantage:24–28','Matching nonempty-shape validation and detached subtraction.'),evidence('pytorch-detach','official v2.9.0 torch/_tensor.py:769–785','detach removes autograd graph relation and the result never requires gradient.'),evidence('original-run','stdout.txt:1–2; environment.json /attempted_fences','Exact original one-fence execution on installed torch 2.14.1+cpu.'),evidence('variant-run','cpu-variants-result.json /base, /contract_errors, /ppo_detach_contract','Detached grad_fn/invalid-shape checks and old-baseline/advantage gradient isolation.')],
 ['helper-snapshot','pytorch-source','fence-execution','fence-stdout','fence-environment','variants-execution'],
 {'method':'executed','expected':'Printed advantage.requires_grad False; grad_fn None; invalid inputs raise ValueError; PPO new logprob receives gradients but old logprob and fixed advantage do not.',
  'observed':'All expected outputs/assertions matched on CPU; original helper guard_events empty.',
  'details':'Coverage groups torch.tensor(requires_grad), Tensor.detach/requires_grad/grad_fn, tolist and two-decimal round display, matching-shape/empty validation, and PPO logprob ratio autograd. Routine display APIs are not separate factual claims.'},
 api_coverage=['torch.tensor and requires_grad','bandit_advantage','Tensor.detach/autograd metadata','tolist/round display','helper shape validation','PPO exp/logprob/advantage detach'])
claim('value-separate-objective','software','The chapter baseline model learns with its own prediction loss, while the policy update treats the sampled advantage as fixed.',
 'course/chapters/13.md#13.11 post-fence paragraph (frozen line 365)',
 'This chapter uses separate policy and value networks and optimizers; a bounded synthetic step checks their contract, without claiming a trained model outcome. PPO permits shared trunks in general.',
 [evidence('ppo','§5 Eq.(9), ppo-v2.txt:208–222','Value-function squared prediction error is a distinct optimization objective.'),evidence('experiment-code','run_posttraining:329–360 and _update:151–157','Separate FiniteValueModel/optimizer; old_values captured before reuse; F.mse_loss(value(features), rewards) used for the value update.'),evidence('variant-run','cpu-variants-result.json /separate_value_update','One actor step keeps every value parameter unchanged, followed by one separate MSE step changing value parameters.')],
 ['ppo-text','experiment-snapshot','method-inspection','variants-code','variants-execution'],
 {'method':'executed','expected':'Actor step produces no value gradients or value parameter change; separate critic prediction loss changes value parameters.',
  'observed':'actor_step_kept_value_parameters=true; critic_step_changed_value_parameters=true; value MSE .3039046526 → .2376844436.',
  'details':'Two synthetic contexts, one SGD actor step, one SGD critic step; illustrates gradient/update separation rather than model learning capability.'})
claim('one-action-contextual-bandit','concept','Choosing one entire prewritten response card and ending after its scalar reward is a contextual bandit; the sampled single-step advantage can be R−V_old.',
 'course/chapters/13.md#13.11 contextual-bandit paragraph (frozen line 367)',
 'Action is a categorical card index conditioned on structured context; the cards are prewritten, not generated or understood by this finite network.',
 [evidence('instructgpt','§3.5 RL environment; instructgpt-v1.txt:484–489','A prompt/response bandit ends after the reward for the response.'),evidence('gae','§2 Eq.(2)/(3), §3 Eq.(15); gae-v6.txt:128–135,284–293','Return-minus-value-baseline estimator reduces to a scalar immediate reward in terminal one-action case.'),evidence('helper-code','FiniteResponsePolicy:53–61, FiniteRewardModel:64–75','Four prewritten action logits conditioned on structured features; RM scores context+candidate identity.'),evidence('experiment-code','run_posttraining:335–345','One multinomial action per sampled context and scalar selected reward, with no successive within-episode decisions.')],
 ['instructgpt-pdf','instructgpt-text','gae-text','helper-snapshot','experiment-snapshot','own-derivation'])
claim('token-credit-and-rlhf-scope','concept','Autoregressive token answers involve successive choices whose early tokens influence later text and final reward; the finite-card subtraction example is not complete LM RLHF.',
 'course/chapters/13.md#13.11 contextual-bandit paragraph second half (frozen line 367)',
 'Distinguishes categorical prewritten-card implementation from sequential token actions; a whole-completion LM objective can itself be modeled as a bandit. No claim that every token requires an external score or a mandatory particular GAE variant.',
 [evidence('gae','§1 delayed credit assignment; §2 trajectories; §3 Eq.(15)/(16); gae-v6.txt:42–58,96–107,284–334','Delayed feedback and sequential actions require return/temporal credit treatment.'),evidence('instructgpt','§3.5 bandit and per-token KL; Appendix C.4; instructgpt-v1.txt:484–489,2329–2334','Concrete LM RLHF implementation combines prompt-completion reward with per-token penalties and generalized advantage estimates.')],
 ['gae-pdf','gae-text','instructgpt-pdf','instructgpt-text','own-derivation'])
claim('ppo-fixed-advantage-ratio-reuse','concept','PPO uses fixed sampled advantages and new/old action probability ratios while reusing the old response batch for optimization.',
 'course/chapters/13.md#13.11 PPO paragraph (frozen line 369)',
 'Supports sampled-batch clipped-surrogate method and fixed old-policy reference during inner updates; not a guarantee that all ratios or KL stay within hard bounds.',
 [evidence('ppo','§3 Eq.(6)/(7) and Algorithm 1; ppo-v2.txt:105–131,233–245','Probability ratio denominator is old policy; compute advantages first, optimize K epochs before resetting old policy.'),evidence('helper-code','ppo_clipped_objective:31–50','Ratio from new−detached-old logprob; fixed detached advantage.'),evidence('experiment-code','run_posttraining:337–360','Sample once, capture old probabilities/values/advantages before epoch loop, reuse in inner updates.')],
 ['ppo-pdf','ppo-text','helper-snapshot','experiment-snapshot','variants-execution'])
claim('reward-scale-not-binary','concept','Reward scale should be recorded, and a real learned reward network is not intrinsically restricted to scores 0 and 1.',
 'course/chapters/13.md#13.11 PPO paragraph final sentence (frozen line 369)',
 'The 0/1 example is hand-set feedback, not a claim that all learned RM outputs are binary or calibrated probabilities.',
 [evidence('instructgpt','§3.5 RM scalar output, Eq.(1), score centering; instructgpt-v1.txt:425–432,446–454,481–482','Scalar preference log-odds differences and zero-centered learned rewards do not impose a binary output range.'),evidence('experiment-code','_normalized_scores:160–162 and run_posttraining:315–323','Actual finite method computes/records a training score scale before normalizing RM rewards.')],
 ['instructgpt-pdf','instructgpt-text','experiment-snapshot','method-inspection','own-derivation'])

report={'schema_version':1,'review_stage':'technical','lesson_id':'13.11','source':'course/chapters/13.md#13.11',
 'source_sha256':manifest['source_sha256'],'figure_sha256':{},'verdict':'pass',
 'reviewer_task':'/root/phase4_factual_coordinator/factual_13_11','reviewer_context':'fresh','author_tasks':[],
 'reviewed_on':'2026-10-05','read_scope':{'section':'All original section bytes, frozen lines 348–372',
  'prerequisites':['13.8','13.10'],'intro_read':False,'intro_reason':'Not the first section of chapter 13.',
  'implementation':'AST-selected method ranges explicitly recorded in original-method-inspection.json and own derivation.',
  'original_result_json':'None cited by this section; no pre-existing empirical model score is claimed.',
  'prior_review_or_author_result_summaries_read':False,'visual':'Self-contained exact section desktop/mobile renders personally viewed; no referenced source figure.'},
 'frozen_input_snapshot':{'path':REL+'/inputs/13.md','sha256':manifest['whole_chapter_frozen_input_sha256'],
  'meaning':'Full chapter bytes frozen at initial read, not a current whole-chapter fingerprint after concurrent revisions.'},
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[],
 'checks':{
  'factual_accuracy':{'status':'pass','details':'All eight substantive claim groups independently checked against original GAE/PPO/InstructGPT sources, original method contracts and own derivation. No unresolved material discrepancy found.','claim_ids':[c['id'] for c in claims]},
  'numeric_verification':{'status':'pass','details':'Original fence and .9 exercise executed; per-element score units, sample axis, terminal reduction and float32 tolerance checked. No performance-rate denominator exists here.','claim_ids':['advantage-numbers']},
  'figure_consistency':{'status':'not_applicable','details':'No image/SVG is referenced; the two samples and scalar subtraction do not require a spatial diagram. Exact section rendered/viewed on desktop and mobile; whole-course integration is outside scope.','claim_ids':[]},
  'source_verification':{'status':'pass','details':'Personally read original versioned PPO v2, GAE v6 and InstructGPT v1 text; exact authority/version/locators/support recorded. Optional PyTorch docs403 preserved, then official v2.9.0 source read; installed 2.14.1 CPU behavior executed.','claim_ids':[c['id'] for c in claims]},
  'limitations':{'status':'pass','details':'Exact fence has no backward/update; bounded variant checks use two synthetic samples and one actor/critic step. No download of training data/model, complete training, GPU work, existing model reevaluation, or capability acceptance. Sampled advantage relative to a learned baseline is not answer correctness or a guaranteed improvement.','claim_ids':['fence-and-detach-contract','value-separate-objective','one-action-contextual-bandit','token-credit-and-rlhf-scope','ppo-fixed-advantage-ratio-reuse','reward-scale-not-binary']}
 },
 'tool_limitations':[{'event':'Chromium file URL load blocked by administrator, not timeout','evidence':REL+'/render-file-url-attempt-result.json','resolution':'Same saved HTML rendered with page.set_content; both actual screenshots viewed.'},
  {'event':'Optional official PyTorch docs HTTPS returned403','evidence':REL+'/detach-docs-stderr.txt','resolution':'Personally checked official tagged v2.9.0 raw source; executing version explicitly separate and verified on CPU.'}]}
raw=(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
(BASE/'report.json').write_bytes(raw)
canonical=ROOT/'docs/technical-reviews/13.11.json'
canonical.write_bytes(raw)
assert canonical.read_bytes()==raw
print(json.dumps({'canonical':str(canonical),'verdict':report['verdict'],'source_sha256':report['source_sha256'],
 'report_sha256':hashlib.sha256(raw).hexdigest(),'canonical_equals_independent_report':True,'claims':len(claims),'artifacts':len(artifacts)},ensure_ascii=False))
