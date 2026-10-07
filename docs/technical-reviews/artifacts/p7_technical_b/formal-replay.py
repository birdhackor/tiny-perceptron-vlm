"""Own cheap CPU re-verification for earlier shorthand command receipts.
Original provisional outputs/trace stay untouched; no optimizer or training run.
"""
import json,math,sys,hashlib,copy
from pathlib import Path
import torch
import torch.nn.functional as F
from tiny_perceptron.data import ByteTokenizer,render_chat,IGNORE,pad_batch
from tiny_perceptron.model import TinyLM,ModelConfig,masked_loss
from tiny_perceptron.attention import manual_attention
B=Path(__file__).parent; page=sys.argv[1]; old=json.loads((B/'pages'/f'{page}.json').read_text()); tok=ByteTokenizer(); computed={}
bad=[a for a in old['artifacts'] if a.get('kind')=='execution' and ('heredoc' in a.get('command','') or "<<'PY' (" in a.get('command',''))]
saved={a['id']:json.loads(Path(a['path']).read_text()) for a in bad}
def chat(q,a):
    x,y=render_chat([{'role':'user','content':q},{'role':'assistant','content':a}]);return x,y
def exact_report(report):
    n=0
    for s in report['samples']:
        ids=s['generated_ids'];content=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        good=content==tok.encode(s['expected']);assert good==s['exact'] and s['eos']==(tok.eos_id in ids);n+=good
    assert n==report['matches'] and len(report['samples'])==report['records']
    return {'matches':n,'records':report['records'],'eos':sum(s['eos'] for s in report['samples']),'nll_sum':report['nll_sum'],'effective_tokens':report['effective_tokens'],'nll':report['nll_sum']/report['effective_tokens']}
if page=='7.1':
    x,y=chat('1+1=?','2');assert x.tolist()==saved['cpu']['x'] and y.tolist()==saved['cpu']['y']
    for row in saved['variation']:
        x,y=chat(row['q'],row['r']);assert x.tolist()==row['x'] and y.tolist()==row['y'] and y[y!=IGNORE].tolist()==row['effective']
    computed={'original_x':saved['cpu']['x'],'original_y':saved['cpu']['y'],'four_variations_recomputed':True}
elif page=='7.2':
    prompt=[tok.bos_id,tok.user_id]+tok.encode('1+1=?')+[tok.eos_id,tok.assistant_id];assert prompt==saved['cpu']['prompt'] and prompt[:-1]==saved['cpu']['without_assistant'];assert tok.encode('<assistant>')==saved['cpu']['content_changed_literal'];computed={'prompt':prompt,'literal':tok.encode('<assistant>')}
elif page=='7.3':
    x,y=chat('問','答');assert y[y!=IGNORE].tolist()==saved['cpu']['active_targets'];z=torch.zeros(1,len(y),264,requires_grad=True);masked_loss(z,y[None]).backward();assert z.grad[0,0].abs().sum()==0 and z.grad[0,6].abs().sum()>0
    for row in saved['vary']:
        x,y=chat('問',row['answer']);assert len(x)==row['length'] and int((y!=IGNORE).sum())==row['effective'] and y[y!=IGNORE].tolist()==row['targets']
    computed={'ignored_logit_row_grad_sum':0.,'active_logit_row_grad_positive':True,'answer_counts':[r['effective'] for r in saved['vary']]}
elif page=='7.4':
    for row in saved['cpu']['cases']:
        x,y=chat(row['question'],row['answer']);assert x.tolist()==row['x'] and y.tolist()==row['y'] and torch.nonzero(y!=IGNORE)[0].item()==row['first']
    for row in saved['vary']:
        x,y=chat(row['q'],row['r']);i=torch.nonzero(y!=IGNORE)[0].item();assert i==row['first'] and x[i].item()==row['x_first'] and y[i].item()==row['y_first']
    computed={'original_cases':saved['cpu']['cases'],'variations':saved['vary']}
elif page=='7.5':
    torch.manual_seed(42);x,y=chat('Z','A');m=TinyLM(ModelConfig(width=8));z=m(x[None])['logits'];z.retain_grad();masked_loss(z,y[None]).backward();a=z.grad[0,2].norm().item();b=m.embedding.weight.grad[x[2]].norm().item();assert a==0 and b>0 and abs(b-saved['vary']['Z_embedding_grad_norm'])<1e-8;computed={'logit_grad_norm':a,'Z_embedding_grad_norm':b}
elif page=='7.6':
    a=torch.tensor([[[0.,2.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]]);labels=torch.tensor([[1,2,0,-100,-100]]);loss=masked_loss(a,labels).item();assert abs(loss-saved['vary']['first_ignored_changed_to_PAD0'])<1e-7
    try:masked_loss(a,torch.full_like(labels,-100));raise AssertionError('Expected error')
    except ValueError as error:assert str(error)==saved['vary']['empty_error']
    computed={'first_ignored_changed_to_PAD0':loss,'all_ignored_ValueError':True}
elif page=='7.7':
    a=torch.ones(1,1,2,3);allowed=torch.zeros(1,1,2,2,dtype=torch.bool);out,w=manual_attention(a,a,a,allowed);nan=torch.full((2,),-torch.inf).softmax(-1).isnan().all().item();assert nan and not out.any() and not w.any();computed={'guarded_output':out.tolist(),'guarded_weights':w.tolist(),'raw_all_minus_inf_softmax_nan':nan}
elif page=='7.8':
    z=torch.arange(40.).reshape(2,4,5)
    for key,valid in [('cpu',[[True,True,False,False],[False,True,True,True]]),('vary',[[True,False,False,True],[False,True,True,True]])]:
        v=torch.tensor(valid);last=torch.arange(4).expand_as(v).masked_fill(~v,-1).amax(-1);values=z[torch.arange(2),last];assert last.tolist()==saved[key]['last'] and values.tolist()==saved[key]['selected']
    computed={'last_original':saved['cpu']['last'],'last_changed':saved['vary']['last'],'empty_last':-1}
elif page=='7.10':
    assert tok.encode('是')==saved['correction']['correct_answer_ids']==[238,160,183];computed={'correct_answer_ids':tok.encode('是'),'earlier_manual_metadata_error_retained':True}
elif page=='7.11':
    x,y=chat('1+1=?','3');torch.manual_seed(42);m=TinyLM(ModelConfig(width=8));z=m(torch.stack([x,x]))['logits'];assert y[y!=IGNORE].tolist()==saved['vary']['effective'] and list(z.shape)==saved['vary']['logits_shape'];computed={'effective':y[y!=IGNORE].tolist(),'logits_shape':list(z.shape)}
elif page=='7.12':
    pairs=[(c,s) for c in ['紅','藍'] for s in ['圓','方']];held={('紅','方')};train=[p for p in pairs if p not in held];assert [list(p) for p in pairs]==saved['cpu']['all_pairs'] and [list(p) for p in train]==saved['cpu']['train'];assert not set(train)&held;assert [list(p) for p in pairs if p[0]!='藍']==saved['cpu']['leave_blue_out_train'];computed={'all_pairs':[list(p) for p in pairs],'train':[list(p) for p in train],'held_out':[list(p) for p in held]}
elif page=='7.13':
    for key in ['cpu','vary']:
        rows=saved[key]['cases'] if isinstance(saved[key],dict) else saved[key]
        for row in rows:
            o=json.loads(row['reply']);scores={'格式符合':set(o)=={'answer'} and type(o.get('answer')) is int,'1+1內容正確':o.get('answer')==2};assert scores==row['scores']
    computed={'all_original_named_field_scores_recomputed':True,'bool_is_int_subclass':isinstance(True,int)}
elif page=='7.14':
    logits=torch.tensor([[0.,0.,3.,8.]]);loss3=F.cross_entropy(logits,torch.tensor([3])).item();loss2=F.cross_entropy(logits,torch.tensor([2])).item();assert abs(loss3-saved['vary']['new_wrong3_loss'])<1e-8 and abs(loss2-saved['vary']['true2_loss'])<1e-7
    raw=json.load(open('docs/course-experiments/results/sft_ablation.json'))['results'];reports={}
    for name in ['clean','noisy']:
        reports[name]={}
        for split in ['validation','test']:
            r=exact_report(raw['runs'][name]['attributes'][split]);o=saved['holdouts'][name][split];assert r['matches']==o['exact'] and r['records']==o['records'] and abs(r['nll']-o['meanNLL'])<1e-12;reports[name][split]=r
    computed={'new_wrong3_loss':loss3,'true2_loss':loss2,'all_heldout_reports':reports}
elif page=='7.15':
    measurements={s:{t:None for t in ['A舊任務','B新任務']} for s in ['微調前','只訓練B後']};assert measurements==saved['placeholder']['measurements'];computed={'measurements':measurements,'no_model_calls':True}
elif page=='7.17':
    cost={'article_cost':1000*1,'checked_chat_cost':50*20,'all_checked_chat_cost':1000*20,'variation_checked_100':100*20};assert all(saved['cost'][k]==v for k,v in cost.items());raw=json.load(open('docs/course-experiments/results/sft.json'))['results'];st=raw['pretrain_then_sft'];rr={'pretrain_before':exact_report(st['before_sft']['test']),'pretrain_after':exact_report(st['after_sft']['test']),'direct_after':exact_report(raw['after']['test'])}
    for name,r in rr.items():assert r['matches']==saved['rawcheck'][name]['matches'] and r['records']==saved['rawcheck'][name]['denominator'] and abs(r['nll']-saved['rawcheck'][name]['nll'])<1e-12
    assert [st['pretraining']['steps'],st['sft']['steps'],raw['training']['steps']]==[250,900,900];computed={'cost':cost,'all_three_raw_reports':rr,'recipe':[250,900,900]}
elif page=='7.18':
    computed={'demonstration':{'question':'只回數字：1+2=?','answer':'3'},'preference':{'question':'只回數字：1+2=?','chosen':'3','rejected':'答案是3喔！'},'feedback':{'question':'只回數字：1+2=?','sampled_answer':'4','reward':0}};assert computed==saved['cpu']
elif page=='7.21':
    a=torch.zeros(4,4,requires_grad=True);b=torch.zeros(4,4,requires_grad=True);l1=F.cross_entropy(a,torch.tensor([0,1,2,3]));l2=F.cross_entropy(b,torch.tensor([1,2,3,-100]),ignore_index=-100);total=l1+0*l2;total.backward();assert torch.count_nonzero(a.grad).item()==16 and torch.count_nonzero(b.grad).item()==0 and abs(total.item()-saved['zero']['weight0_total'])<1e-8;computed={'counts':[4,3],'weight0_total':total.item(),'next_grad_nonzero':16,'later_grad_nonzero':0}
elif page=='8.1':
    examples=[{'回答':'4','正確':True,'貼切比喻':False},{'回答':'4，就像兩雙筷子共有四根。','正確':True,'貼切比喻':True},{'回答':'5，數字如星光流動。','正確':False,'貼切比喻':False}];assert examples==saved['cpu'];computed={'manual_examples':examples,'labels_are_given':True}
elif page=='8.2':
    prompts=[s+':2+2=?' for s in ['用一句話回答','用貼切比喻回答']];prompts=[s.replace(':','：') for s in prompts];changed=[prompts[0],'只回一個數字：2+2=?'];assert prompts==saved['cpu']['prompts'] and changed==saved['variation'];computed={'prompts':prompts,'variation':changed,'model_calls':0}
elif page=='8.3':
    x,y=chat('2+2=?','4，共有四個');assert len(x)==saved['short']['input_count'] and int((y!=IGNORE).sum())==saved['short']['learn_count'] and y[y!=IGNORE].tolist()==saved['short']['active_labels'];raw=json.load(open('docs/course-experiments/results/style.json'))['results'];reports={'base':raw['content_evaluation']['test'],'concise':raw['default_style_runs']['concise']['after']['test'],'vivid':raw['default_style_runs']['vivid']['after']['test']}
    for name,r in reports.items():
        for s,o in zip(r['samples'],saved['baseline'][name],strict=True):
            ids=s['generated_ids'];content=ids[:ids.index(2)] if 2 in ids else ids;assert content==tok.encode(s['generated']) and ids==o['ids'] and (content==tok.encode(s['expected']))==o['exact'] and s['eos']
    computed={'input_count':len(x),'learn_count':int((y!=IGNORE).sum()),'all21_raw_ID_controls_rechecked':True}
elif page=='8.5':
    computed={}
    for text,oldrow in saved['nan'].items():
        o=json.loads(text);valid=isinstance(o,dict) and set(o)=={'answer'};correct=valid and o.get('answer')==4;row={'valid':valid,'correct':correct,'nan':isinstance(o.get('answer'),float) and math.isnan(o['answer'])};assert row==oldrow;computed[text]=row
else:raise ValueError(page)
out={'page_id':page,'purpose':'own explicit-command CPU recheck; original evidence retained unchanged','environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'},'recomputed':computed,'original_saved':saved,'original_saved_sha256':{a['id']:hashlib.sha256(Path(a['path']).read_bytes()).hexdigest() for a in bad}}
print(json.dumps(out,ensure_ascii=False,indent=2))
