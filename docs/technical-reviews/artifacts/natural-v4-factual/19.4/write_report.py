"""Write this reviewer's fresh report from the completed evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).parent
REL = BASE.relative_to(ROOT)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
data = json.loads((BASE/'audit-results.json').read_text())
receipts = {r['file']:r for r in json.loads((BASE/'retrieval-receipts.json').read_text()) if 'sha256' in r}
sources = []

def official(identifier, file, title, kind, version, reason, note):
    r = receipts[file]
    sources.append(dict(id=identifier, kind=kind, title=title, url=r['url'], version=version,
                        verified=True, checked_original=True, accessed_on='2026-10-04',
                        authority_reason=reason, inspection_note=note,
                        retrieval_sha256=r['sha256']))

official('s_instruct','instructgpt.pdf','Training language models to follow instructions with human feedback','paper','arXiv 2203.02155v1',
         'Original authors describe continuing a pretrained policy with supervised demonstration training.',
         'Read section 3.1, Step 1 and surrounding three-step description. It supports changed objectives on pretrained weights, not universal superiority of every staged recipe.')
official('s_switch','switch.pdf','Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity','paper','JMLR 23 (2022), article 21-0998',
         'Original sparse-transformer paper defines expert routing and auxiliary balancing.',
         'Read section 2.1 equation 2 and section 2.2 printed page 6 equations 4–6 and alpha=1e-2 paragraph. Paper is top-1; project uses a documented top-2 adaptation with selected-slot load normalization. Balancing is encouraged, not guaranteed.')
official('s_dpo','dpo.pdf','Direct Preference Optimization: Your Language Model is Secretly a Reward Model','paper','arXiv 2305.18290v3',
         'Original method paper defines the preference/reference objective.',
         'Read section 4 equation 7 and following DPO outline. Compared the reference-relative preferred/rejected log probabilities to project preference_loss and frozen_reference. Replay CE/balancing are project additions; no PPO sequence or improvement guarantee is implied.')
official('s_torch','torch-functional.py','PyTorch functional.cross_entropy source for running build','official_source','git 5c4886908584029761b579af026dcfb627c84070; running torch 2.14.1+cpu',
         'Official PyTorch source at torch.version.git_version of the actually executed installation.',
         'Read torch/nn/functional.py lines 3478–3570: unnormalized input logits, class-index targets, ignore_index=-100, reduction=sum, and cross_entropy_loss dispatch. Confirmed actual CPU sum/count, ignored gradients and dtype/shapes; no GPU backend claim.')
official('s_checkpoint','checkpoint-tutorial.py','PyTorch general checkpoint saving/loading tutorial source','official_source','tutorials main retrieved 2026-10-04; content SHA '+receipts['checkpoint-tutorial.py']['sha256'],
         'Official PyTorch tutorial source distinguishes a weight state_dict from a checkpoint for resume.',
         'Read saving_loading_models.py lines 255–305, including torch.save dictionary with model/optimizer/epoch/loss, and their separate state_dict restoration. Documentation HTML was HTTP 403; this original raw source was retrieved successfully. Used only general checkpoint semantics, not a cross-device exact-resume guarantee.')
official('s_hashlib','hashlib.html','CPython hashlib documentation','official_docs','Python 3.13 documentation retrieved 2026-10-04',
         'Official language library documentation defines bytes-based SHA construction and hexadecimal digest.',
         'Read Hash algorithms, hash.digest_size, hash.hexdigest, and sha256 examples. Digest identifies bytes; hex output has twice the byte length. It provides no model capability or historical-run authentication.')

code_notes = {
    'tiny_perceptron/capstone.py': 'Read default_config/PadSafeBlock/CapstoneModel (34–120), build_dataset (139–282), prompt_ids/encode_record/prepare_batch (299–338), preference pairs/loss/reference (341–376), evaluation/protocol (499–606), save/load (612–652). Full SHA matches original report and independently retrieved historical run source.',
    'scripts/course_experiments/capstone.py': 'Read balanced sampling (62–67), train_stage (70–277): predecessor checks/load 96–116, split filtering/reference 117–126, sampler/state 125–163, objective/count 197–218, validation/test flag 241–274. Read _run_context 280–293 and first official test use in run_deployment 312–362. Full SHA matches independently retrieved historical run source.',
    'tiny_perceptron/data.py': 'Read IGNORE/SPECIALS/ByteTokenizer (10–28), shifted and render_chat (46–68), pad_batch (71–86): UTF-8 byte IDs, EOS/role ids and targets separate from context.',
    'tiny_perceptron/model.py': 'Read ModelConfig/Block (15–50), TinyLM (53–89) and loss_sum/masked_loss (92–106): Dense/MoE construction, [B,T,V] scores, sum-reduced CE then effective-label count.',
    'tiny_perceptron/modern.py': 'Read DenseFFN (35–53), MoEFFN (56–92): independently stored FFNs, selected top-k dispatch, normalized top-2 weights, selected-slot load fractions and importance dot product.',
    'tiny_perceptron/alignment.py': 'Read sequence_log_probability and dpo_loss (29–43): summed nonignored response log probabilities, reference-relative margin, -logsigmoid(beta*margin).',
}
code_ids = {}
for i,(path,note) in enumerate(code_notes.items(),1):
    identifier = f's_code{i}';code_ids[path]=identifier
    sources.append(dict(id=identifier,kind='repository_code',title=path,path=path,sha256=sha(ROOT/path),
                        version='Current full-file bytes personally inspected 2026-10-04',verified=True,inspection_note=note))

for stage in ['pretrain','sft','joint','dpo']:
    suffix = 'preference' if stage=='dpo' else stage
    official('s_run_'+stage,'public-capstone-'+suffix+'.json','Original '+stage+' GPU stage result','official_source',
             'Public repo commit 1df335318bda03fd771807f66976953231d5a00b',
             'Primary project record of the completed fixed GPU run, not reviewer-generated training.',
             'Fetched exact published bytes and matched local report SHA. Read environment/seed/data_version/steps/effective_tokens/history/parent_checkpoint_sha256/inference_export/validation_summary/test_evaluated. audit.py independently rebuilds manifests, sample counts, objective arithmetic and all row scores. Reported device L4 is record evidence, not a fresh reviewer benchmark.')
    official('s_rows_'+stage,'public-validation-'+stage+'.json','Original '+stage+' validation row records','official_source',
             'Public repo commit 1df335318bda03fd771807f66976953231d5a00b',
             'Primary saved action/runtime/final-generation traces for the original fixed run.',
             'Retrieved exact bytes matching local validation.json. Every one of 84 records was checked against rebuilt ids/families/questions/answers/prompt ids; generated ids, decoded raw text, EOS, strict action parameters, runtime sum and terminated final answer were independently rescored. No checkpoint was downloaded or generation repeated.')

sources.append(dict(id='s_math',kind='derivation',title='UTF-8 target counts and effective-target mean',verified=True,
                    details='Style: user32 + newline1 + answer9 + EOS1 =43 pretrain; answer9+EOS1=10 SFT. Concept: user23+newline1+answer28+EOS1=53; answer28+EOS1=29. Two-row SFT batch: 10+29=39 supervised positions; loss=-sum_valid log_softmax(scores)[target]/39. SHA-256=256 bits=32 bytes, two hex digits per byte =>64 characters. Full derivation and CPU gathered-probability comparison in analysis.md/audit results.'))

def ev(s, locator, supports): return dict(source_id=s,locator=locator,supports=supports)
def verify(expected, observed, details, denominators=None, tolerance=None):
    v = dict(method='executed',expected=expected,observed=observed,details=details)
    if denominators: v['denominators']=denominators
    if tolerance: v['tolerance']=tolerance
    return v
claims=[]
def add(identifier,kind,statement,location,evidence,scope,verification=None,artifact_ids=None):
    c=dict(id=identifier,kind=kind,statement=statement,location=location,status='verified',evidence=evidence,
           artifact_ids=artifact_ids or ['a_analysis'],scope=scope)
    if verification: c['verification']=verification
    claims.append(c)

add('c1','concept','A checkpoint snapshots weights and related state; continued learning requires the intended previous weights and compatible architecture/tokenizer. A file digest identifies bytes, not capability.',
    'Opening two paragraphs; checkpoint and fingerprint explanation',
    [ev('s_checkpoint','saving_loading_models.py lines 255–305','Model/optimizer/progress snapshot and separate restoration'),ev('s_instruct','section 3.1 Step 1','Pretrained weights continued with supervised demonstration learning'),ev('s_hashlib','Hash algorithms; hash.digest_size; hash.hexdigest','File bytes determine digest, with 64 SHA-256 hexadecimal characters')],
    'Compatible source state is necessary for the claimed sequence. File identity does not itself prove correctness, capacity or exact resume across environments.')
add('c2','concept','The pretraining objective predicts all shifted short-text targets; SFT supplies the prefix as context while supervising only answer bytes and EOS. IGNORE excludes loss, not input visibility; batches average by all effective targets.',
    'Paragraphs after code and final exercise; 7.3/5.2 prerequisites',
    [ev('s_torch','functional.py cross_entropy lines 3478–3570, ignore_index and reduction','Targets marked -100 contribute neither CE nor score gradients'),ev('s_code1','encode_record/prepare_batch lines 308–338','Target shift and separate ids/valid/labels'),ev('s_code4','loss_sum/masked_loss lines 92–106','Sum loss then divide by nonignored positions')],
    'This project byte protocol and assistant-only target choice. Context remains subject to causal attention; ignored targets are not an automatic data-quality filter.',artifact_ids=['a_analysis','a_cpu','a_results'])
add('c3','software','build_dataset returns splits/manifest with task,user,answer records; next selects the stated training examples. The snippet constructs an untrained Dense model and evaluates [B,T,264] logits via keyword batch arguments; formal default is a top-2 four-expert MoE.',
    'Code block and explanatory paragraphs before/after it',
    [ev('s_code1','default_config 34–48; CapstoneModel 74–105; build_dataset 139–282; prepare_batch 321–338','Actual returned records, model inputs and configuration'),ev('s_code4','Block 31–47; TinyLM.forward 68–86','Dense/MoE FFNs and score dimensions')],
    'Only random CPU forward/loss plumbing; no formal trained model or optimizer update was involved.',
    verify('Seed42 selects style 29 and concept 3/6; dense=True uses DenseFFN; finite losses and logits match labels x264.',
           'Both stated records selected, all example FFNs DenseFFN, logits [1,43,264]/[1,76,264] and [1,53,264]/[1,86,264], four finite losses.',
           'audit.py examples; default_config(dense=True) executed independently for each task with seed42; int64 inputs and labels inspected.'),['a_cpu','a_results','a_code'])
add('c4','numeric','Style has 43 pretrain and 10 SFT effective targets; concept has 53 and 29. Both losses are finite, and padding does not enter the mean-loss denominator.',
    'Effective target paragraph and final style-to-concept exercise',
    [ev('s_math','Style and concept UTF-8 byte derivations; 39-target mean','Independent count and denominator arithmetic'),ev('s_torch','cross_entropy ignore_index/reduction','Nonignored loss terms')],
    'Exact integer counts for these seed42 records. Finite random-model loss shows connectivity and gives no quality comparison between different objectives.',
    verify('43/10 and 53/29, all finite; two-row SFT count39; ignored positions contribute zero.',
           '43/10 and 53/29, all finite. Independent sum233.24468293939282 /39 = mean5.980632895881867; ignored gradient max0; changing ignored logits leaves mean exactly unchanged.',
           'Executed exact snippet plus exercise. Independent raw UTF-8 lengths and gathered float64 log probabilities verify both counting and averaging.',
           tolerance='Integers/finite booleans exactly equal; independent float64 mean absolute tolerance1e-12, observed exact agreement.'),['a_cpu','a_results','a_analysis'])
add('c5','concept','The formal first three stations add coefficient0.01 router balancing to CE to encourage expert use; the DPO comparison uses a frozen predecessor reference and replay CE, so its reported CE target count is not a full equal-cost training budget.',
    'Paragraph below stage table; pipeline fixed-reference label',
    [ev('s_switch','section 2.2 printed page6 equations4–6 and alpha=10^-2 paragraph','Auxiliary differentiable balancing principle and coefficient example'),ev('s_dpo','section4 equation7 and DPO outline','Policy/reference preferred-rejected scoring'),ev('s_code2','train_stage 117–163, 197–218','Frozen reference, .01 balancing, .2 replay CE and separate target counter'),ev('s_code5','MoEFFN 56–92','Project top-2 adaptation'),ev('s_code6','sequence_log_probability/dpo_loss 29–43','Preference scores are separate from replay CE count')],
    'Balancing is an incentive, not guaranteed uniform routing or expert semantics. Switch paper is top-1; project top-2 adaptation and added CE replay are not claimed identical to pure DPO. GPU history formula residual <=1.765e-7 is FP32 rounding.',artifact_ids=['a_analysis','a_cpu','a_results'])
add('c6','empirical','The fixed recorded recipe uses capstone-small-world-v2, seed42, NVIDIA L4; completed pretrain/SFT/joint/DPO updates are300/1400/600/100. All stages evaluate the same84 validation rows; the formal final90 are deferred until deployment.',
    'Fixed recipe paragraph, update-count table and last report-navigation paragraph',
    [ev('s_run_'+s,'environment; results.data_manifest/stage/steps/schedule_completed/test_evaluated','Original completed fixed-run configuration and validation-only stage status') for s in ['pretrain','sft','joint','dpo']]+[ev('s_code2','train_stage validation 241–274; run_deployment 312–337','Official stage evaluation separates validation from later test')],
    'Primary recorded single-seed GPU runs were audited, not rerun. Flags/source establish the documented protocol, not behavior outside recorded execution. No claim about industry-scale pretraining.',
    verify('Same seed/data/manifests; completed300/1400/600/100; validation84; test90 held for later.',
           'Four exact original reports and rebuilt manifests agree; all completed, test_evaluated=false; counts552 train/84 validation/90 test; L4/torch2.14.1+cu126 recorded.',
           'audit.py compares reports/results/train-report/manifests and primary environment; source inspection locates validation and later official test call.',
           {'seed':42,'stage_updates':{'pretrain':300,'sft':1400,'joint':600,'dpo':100},'train_records':552,'text_stage_train_records':360,'validation_records':84,'deferred_test_records':90,'batch_size':24}),['a_cpu','a_results','a_retrieval'])
add('c7','empirical','Recorded CE target totals are364409/647067/249100/41403, counting repeated byte/EOS targets; DPO41403 counts CE replay only, excluding its policy/reference pair scoring.',
    'Target-count and budget paragraph below table',
    [ev('s_run_'+s,'results.effective_tokens; results.history','Primary reported counts and objective components') for s in ['pretrain','sft','joint','dpo']]+[ev('s_code2','_balanced_sample 62–67; sampler125; training197–218','Draw order and CE-only counter')],
    'Sampling/count recomputation without model training. DPO extra scored response targets94500 also omit prompt and other computation; no equal-FLOP or total-cost conclusion.',
    verify('Exactly364409/647067/249100/41403 under the fixed sampler, including DPO pair RNG draws.',
           'All four independent UTF-8 target sums equal reports. DPO1200 pair occurrences additionally score94500 response targets across policy/reference/chosen/rejected, excluded from CE counter.',
           'Independent Python Random task-balanced sampler with seed42+1000*stage_index, batch24; count raw UTF-8 bytes+EOS, and consume twelve DPO choices each update. No forward/backward training.',
           {'seed':42,'sampler_seeds':[42,1042,2042,3042],'batch_size':24,'updates':[300,1400,600,100],'sample_occurrences':[7200,33600,14400,2400],'ce_target_positions':[364409,647067,249100,41403],'dpo_pair_occurrences':1200,'dpo_pairs_per_update':12,'dpo_extra_response_targets':94500}),['a_cpu','a_results','a_analysis'])
add('c8','empirical','Exact terminated action and final-answer rescoring gives0/84,42/84,75/84,71/84. SFT is42/42 text and0/42 modalities; joint is42/42 text and33/42 modalities with nine shape failures. The quoted pretrain4+4 output is exact; the DPO decrease supports choosing joint in this run.',
    'Stage table and following exact-protocol, pretrain-failure and task-distribution paragraphs',
    [ev('s_rows_'+s,'records[0:84].action_trace/runtime/final_trace and expected_action/expected_final','All primary saved generations independently rescored') for s in ['pretrain','sft','joint','dpo']]+[ev('s_code1','evaluate_rows 499–547; parse_action/calculator_runtime 550–574','Exact action+EOS, ordered calculator arguments and second generation')],
    'Audit of336 saved rows, not fresh generation/quality replication. All results are narrow synthetic exact-protocol scores; pretrain0/84 does not imply zero continuation ability, nor DPO universally harmful.',
    verify('Aggregate0/42/75/71 out of84, stated text/modal splits, nine shape failures, quoted 4+4 trace.',
           'Independent row totals0/84,42/84,75/84,71/84. SFT text42/42 modal0/42; joint text42/42 modal33/42 with9 shape failures; DPO adds4 joint failures. Pretrain trace exactly多4孔。算回算23.',
           'Verified generated ids decoding/EOS and matching prompt/row/family IDs, strict action strings, actual ordered tool parameters, recomputed runtime sums and terminated DIRECT final output, then independently aggregated per task.',
           {'seed':42,'saved_rows_audited':336,'validation_rows_per_stage':84,'text_rows_per_stage':42,'modality_rows_per_stage':42,'image_shape_rows':9,'joint_rows':18,'calculator_rows':10,'task_counts':data['stage_record_audit'][0]['validation_recomputed']['by_task']}),['a_cpu','a_results','a_retrieval'])
add('c9','software','All three full64-character parent/export fingerprint links match; pretrain has null parent. The trainer loads the required completed predecessor, preserves architecture/tokenizer/data identity and records its actual source-file digest.',
    'Parent-checkpoint comparison paragraph; random first station null',
    [ev('s_code2','train_stage 96–116 and metadata159–163; _run_context280–293','Actual predecessor load, checks and byte-file hash recording'),ev('s_code1','load_capstone644–652; save_capstone612–640','Tokenizer/config/state restoration'),ev('s_hashlib','hash.hexdigest','64-character content digest comparison')]+[ev('s_run_'+s,'results.parent_checkpoint_sha256/results.inference_export.sha256','Complete primary reported links') for s in ['pretrain','sft','joint','dpo']],
    'Confirms primary recorded linkage and load mechanism; reviewer did not download/reload historical GPU checkpoint bytes. A roundtrip self-created checkpoint validates local API behavior and all tensor equality, not an independent historical run reproduction.',
    verify('pretrain parent null; three complete predecessor/export hashes identical, 64 hex each; self-created checkpoint reload equal.',
           'All three full links match exactly and first is null. Self-created random Dense checkpoint reload has identical state tensors/tokenizer; one-byte mutation changes digest.',
           'Compare complete report fields, original remote bytes and inspected source file hashes; CPU save/load probe uses only self-created untrained state.'),['a_cpu','a_results','a_retrieval'])
add('c10','concept','The figure represents one core continued A→B→C, followed by a DPO comparison branch C→D with fixed reference; repeated evaluation can reveal regression, and this run recommends C/joint rather than assuming newest weights are best.',
    'capstone_pipeline.svg entire diagram and surrounding stage narrative',
    [ev('s_instruct','section3.1 Step1','Continuing a pretrained core to demonstration training'),ev('s_dpo','section4 equation7 and DPO outline','Separate reference-relative preference learning'),ev('s_run_joint','results.validation_summary.end_to_end_correct','75/84 joint'),ev('s_run_dpo','results.validation_summary.end_to_end_correct','71/84 branch')],
    'Personally viewed rendered arrows/labels and needed prerequisite diagrams. No universal quality guarantee, semantic expert assignment or additional mandatory PPO/student training stage is inferred.',
    artifact_ids=['a_analysis','a_render_receipt','a_figure_pipeline','a_figure_posttrain','a_figure_resources','a_figure_routes','a_results'])

artifacts=[]
def artifact(identifier,kind,file,description,**extra):
    artifacts.append(dict(id=identifier,kind=kind,path=str(REL/file),sha256=sha(BASE/file),description=description,**extra))
artifact('a_code','code','audit.py','Reviewer-authored bounded CPU count/loss/checkpoint/primary-record audit; no training.')
artifact('a_cpu','execution','audit.stdout.txt','Actual full CPU audit stdout; all assertions pass.',command='.venv/bin/python '+str(REL/'audit.py'),result='Exit0; four finite Dense losses, exact counts/token budgets, every saved validation row recomputed, three complete parent links match.',environment=data['environment'])
artifact('a_stderr','source_snapshot','audit.stderr.txt','Actual CPU audit stderr (empty), preserved separately.')
artifact('a_execution','source_snapshot','audit.execution.json','Actual command, cwd, exit and elapsed execution receipt.')
artifact('a_results','source_snapshot','audit-results.json','Own analysis results and exact SHA map of original inputs/registered code; contains no model weights or full paper.')
artifact('a_analysis','derivation','analysis.md','Own byte-count derivations, original-source locators, empirical audit scope and personal figure observations.')
artifact('a_retrieval','source_snapshot','retrieval-receipts.json','Own original HTTPS URL/status/version-SHA retrieval receipt; includes genuine403 and successful official-source fallback.')
artifact('a_prerequisites','source_snapshot','prerequisite-read-receipt.json','Actual full prerequisite section reading identity/hash/use receipt.')
artifact('a_original','source_snapshot','section.original.md','Own raw UTF-8 current assigned section snapshot including trailing blank lines.')
artifact('a_render_receipt','execution','render.execution.json','Actual four Inkscape commands, exit codes, stdout/stderr and original/render hashes.',command='Inkscape export commands listed exactly in render.execution.json',result='All four exports exit0; personally viewed all four renders using view_image. GTK/Pango warnings did not prevent correct rendering.',environment={'renderer':'Inkscape1.4 (e7c3feb100,2024-10-09)','device':'CPU'})
for identifier,name in [('pipeline','capstone_pipeline'),('posttrain','posttrain_stages'),('resources','capstone_resources'),('routes','ppo_dpo_routes')]:
    artifact('a_figure_'+identifier,'figure_render',name+'.render.png','Personally rendered/viewed '+name+'; arrows, text positions and claims compared with section/prerequisites.')

figures={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'course/figures'/f'{n}.svg' for n in ['capstone_pipeline','posttrain_stages','capstone_resources','ppo_dpo_routes']]}
report=dict(schema_version=1,review_stage='technical',lesson_id='19.4',source='course/chapters/19.md#19.4',reviewer_task='/root/v4_review_coordinator/factual_v4_19_4',reviewer_context='fresh',
            source_sha256=sha(BASE/'section.original.md'),figure_sha256=figures,verdict='pass',claims=claims,sources=sources,artifacts=artifacts,issues=[],
            checks={
                'factual_accuracy':dict(status='pass',details='All ten grouped substantive claims independently checked; no contradiction or unresolved claim within stated synthetic fixed-run scope.',claim_ids=[c['id'] for c in claims]),
                'numeric_verification':dict(status='pass',details='Fresh CPU examples43/10 and53/29, 39-target mean; independently rebuilt all CE sample budgets and all336 row scores; exact parent links. Historical GPU scores were audited, not regenerated.',claim_ids=['c4','c6','c7','c8','c9']),
                'figure_consistency':dict(status='pass',details='Direct and three necessary prerequisite SVGs rendered with Inkscape1.4 and personally viewed; directions A→B→C and C→D fixed-reference branch, C recommendation and separate PPO navigation agree with source and evidence.',claim_ids=['c10']),
                'source_verification':dict(status='pass',details='Original specified paper versions, official running PyTorch git source, CPython documentation and official checkpoint tutorial source actually read. Exact published primary records/historical code fetched and matched local full-file SHA. HTML403 resolved via original official raw source.',claim_ids=[c['id'] for c in claims]),
                'limitations':dict(status='pass',details='Maintains distinction between random Dense demo and formal MoE, CE targets and total DPO computation, recorded GPU audit and rerun, exact-protocol scores and general ability, fingerprint and capability, newest checkpoint and preferred version.',claim_ids=[c['id'] for c in claims])},
            review_scope=dict(full_current_section_read=True,introduction='not_applicable;19.4 is not the first numbered section',prerequisite_receipt=str(REL/'prerequisite-read-receipt.json'),old_report_preserved_unread=str(REL/'assigned-report.pre-review.unread.json'),external_raw_research='outputs/natural-v4/factual-research/19.4/',no_heavy_training_or_downloads=True))
target=ROOT/'docs/technical-reviews/19.4.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(BASE/'first-report.original.json').write_bytes(target.read_bytes())
print(json.dumps({'report':str(target.relative_to(ROOT)),'sha256':sha(target),'verdict':report['verdict'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts)},indent=2))
