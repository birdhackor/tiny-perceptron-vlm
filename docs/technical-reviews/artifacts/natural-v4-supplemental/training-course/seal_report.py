from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[5];D=Path(__file__).parent;p=D/'report.json';initial=p.read_bytes();(D/'report.round1.initial.json').write_bytes(initial);R=json.loads(initial)
locators={
'simple_models':'/results/data/{train,validation,test}; /results/runs/{bigram,mlp1,mlp3,mlp5}/{parameters,steps,before_nll,after_nll_same_post_update_time,samples}',
'text_foundation':'/results/data; /results/training/{steps,effective_tokens,parameters}; /results/{before,after}/{validation,test}/{nll,nll_sum,effective_tokens,samples}',
'real_text':'/results/runs/{tinystories,chinese-poetry}/{data,training,before,after}; /after/{validation,test}/{nll,nll_sum,effective_tokens}',
'tokenizer':'/results/data; /results/raw_document_schedule_sha256; /results/runs/{byte256,bpe512}/{training,parameters,after/{validation,test}}',
'sft':'/results/data; /results/training; /results/after/{validation,test}; /results/pretrain_then_sft/{pretraining,sft,after_sft}; /results/ultrachat_pilot',
'sft_ablation':'/results (fixed clean/noisy/replay branch configuration and family split; original exact nested arithmetic locators in record-audit.json)',
'style':'/results/arithmetic_data; /results/content_{training,evaluation}; /results/default_style_runs/{concise,vivid}/{training,after/test/rubric}; /results/{data,training,after/test/{matches,eos_rate,rubric,samples}}',
'safety':'/results/data; /results/runs/{safety-only,model}/{training,safety/test,arithmetic/test}; /results/held_out_wording; /results/pku_pilot',
'lora':'/results/{base_parameters,rank,alpha,scaling}; /results/runs/{concise,vivid}/{training,evaluation/test/rubric}; /results/full_sft/evaluation/test/rubric',
'encoders':'/results/{vision,audio}/{training,validation,test,data}; /results/{vision,audio}/test/{correct,samples}',
'projector':'/results/{training,validation,test}',
'vqa':'/results/variants/{projector_only,partial,all,all_replay,direct_vqa}/{training,validation,test}; /results/variants/direct_vqa/training/{steps,effective_tokens}',
'real_modal':'/results/{fashion-mnist,fsdd}/{data,training,validation,test,conversions}; /test/{correct,examples,effective_tokens,samples}',
'ocr':'/results/{data,training,validation,test}; /results/test/samples/{target,generated,generated_ids,exact_match,eos}',
'dpo':'/results/{data,reference_sha256,reference_unchanged}; /results/runs/{model,beta1}/{training,preference/test,arithmetic/test}; /results/format_only/{training,preference/test,arithmetic/test}; /results/ultrafeedback_pilot',
'capstone_student':'/results/teacher_checkpoint_sha256; /results/data_manifest; /results/branches/{ce,kd}; /results/evaluations/{test_ce,test_kd}/{count,action_correct,end_to_end_correct,by_task}',
'modern':'/results/{runtime,dataset}; /results/variants/{baseline,rope,rmsnorm,relu2,swiglu,tied}/{model,training/{effective_tokens,steps,warm_step_median_seconds},heldout/{validation,test}}',
'moe':'/results/{dataset,runtime}; /results/variants/{dense_active_top1,dense_active_top2,dense_total,top1_aux0,top1_aux0.01,top2_aux0,top2_aux0.01}/{budget,training,heldout}; /results/teacher_variant',
'efficiency':'/results/{source,dataset,runtime}; /results/models/{mha,gqa}; /results/update_variants; /results/manual_vs_sdpa/backend; /results/packing/actual_updates; /results/compile/{first_compiled_call_seconds,eager,compiled,estimated_calls_to_amortize_first_call}',
'flash_probe':'/results/{runtime,configuration}; /results/routes/{fp16,bf16}/{correctness/{output,gradients},profiles/{forward,forward_backward},measurements/{forward,forward_backward}/{manual,forced_flash}}',
'precision':'/results/{source,dataset,runtime}; /results/variants/{fp32,bf16,fp16}/{training/{requested_steps,optimizer_updates,skipped_updates,history},heldout/test,weights_dtype,observed_logits_dtype}',
'quantization':'/results/training/{steps,effective_supervised_tokens}; /results/runs/{fp32,packed4,packed8}/{storage/{tensor_bytes,file_bytes,parameter_count},validation/{correct,examples},test/{correct,examples,answer_nll,supervised_tokens,generated_samples},timing}',
'distillation':'/results/tasks/{attributes,style_transfer,moe_to_dense}/{teacher_provenance,teacher_storage,teacher_test,data,teacher_cache,hard_target_generation,out_of_domain_gsm8k,runs}; /results/tasks/attributes/runs/{w16_ce,w16_teacher_hard,w16_ce_kl,w32_ce,w32_teacher_hard,w32_ce_kl,w32_ce_kl_packed4}/{training,storage,test,teacher_agreement}; /results/tasks/{style_transfer,moe_to_dense}/runs/{w32_ce,w32_teacher_hard,w32_ce_kl}/{training,test,style}',
'multimodal_distillation':'/results/tasks/{vqa,joint}/{teacher_provenance,teacher_test,teacher_validation,data,visual_tokens,language_widths}; /results/tasks/{vqa,joint}/runs/{ce,ce_kl}/{training/{alignment,effective_answer_tokens},storage,test,blank_image_test,blank_audio_test}',
'joint':'/results/{training,validation,test}',
'rag':'/results/{split,icl_samples_same_model_weights,metrics,samples,paired_counterfactuals/{changed_address_context,changed_source_context}/{both_answers_correct,changed_answer_correct}}',
'tools':'/results/{accuracy,completion_rate,first_action_json_rate,correct_parameters_given_executed_call,status_counts,samples,one_step_cap_samples,injected_protocol_checks_not_model_scores}',
'reasoning':'/results/comparison/{direct,steps}/{training,budgets/{candidate_count,oracle_coverage,majority_accuracy,fully_verified_candidates,samples}}; /results/reinforce/{strict,weak_proxy}/{training,after/{greedy_accuracy,sample_accuracy,sample_proxy_reward}}; /results/gsm8k_training_pilot',
'posttraining':'/results/evaluations/{train,validation,test}/policies/{sft,ppo,dpo}/greedy_full_request_success/{numerator,denominator,rate}'
}
for c in R['claims']:
 for e in c['evidence']:
  if e['source_id'].startswith('record-'):e['locator']=locators[e['source_id'][7:]]
for f in ['cpu_probe.py','additional_cpu_probe.py','audit_records.py','record_diagnostics.py','final_evidence.py','browser_probe.py','retrieve_originals.py','retrieve_cards.py']:
 q=D/f;R['artifacts'].append({'id':'code-artifact-'+q.stem,'kind':'code','path':str(q.relative_to(ROOT)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'description':'Exact reviewer-owned execution program for the cited real receipt. Original failures/earlier programs retained separately.'})
R['report_preservation']={'first_true_report':str((D/'report.round1.initial.json').relative_to(ROOT)),'first_true_report_sha256':hashlib.sha256(initial).hexdigest(),'refinement':'Same-owner round1 metadata clarification only: add explicit primary-record JSON locators and reviewer execution-program hashes. No source hash rebinding, no evidence/verdict change.','source_unchanged':True}
p.write_text(json.dumps(R,ensure_ascii=False,indent=2)+'\n')
# Local schema/hash sanity only, not project numbered checker and not an independence proof.
sids={s['id'] for s in R['sources']};arts={a['id']:a for a in R['artifacts']};checks=[]
for a in arts.values():
 checks.append({'check':'artifact hash '+a['id'],'passed':hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']})
for s in R['sources']:
 if s['kind']=='repository_code':checks.append({'check':'source hash '+s['id'],'passed':hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()==s['sha256']})
for c in R['claims']:
 assert all(e['source_id'] in sids for e in c['evidence'])
 assert all(i in arts for i in c['artifact_ids'])
 if c['kind'] in ('numeric','software','empirical'):
  assert set(['method','expected','observed','tolerance','details'])<=set(c['verification'])
  assert any(arts[i]['kind']=='execution' for i in c['artifact_ids'])
 if c['kind']=='empirical':assert isinstance(c['verification']['denominators'],dict) and c['verification']['denominators']
assert len(R['figure_sha256'])==13 and all(f['personal_view']=='completed' for f in json.loads((D/'personal-view-notes.json').read_text())['figures'])
assert set(R['checks'])=={'factual_accuracy','numeric_verification','figure_consistency','source_verification','limitations'}
assert 'lesson_id' not in R
assert R['current_document_sha256']['course/training.md']==hashlib.sha256((ROOT/'course/training.md').read_bytes()).hexdigest()
assert all(x['passed'] for x in checks)
receipt={'scope':'Reviewer-local structural/hash sanity, not numbered checker, not proof of factual truth or independence','checks':checks,'claims':len(R['claims']),'sources':len(S:=R['sources']),'artifacts':len(arts),'figures':13,'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'owned_current_sha256':R['current_document_sha256'],'verdict':R['verdict']}
(D/'local-self-check.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2))
