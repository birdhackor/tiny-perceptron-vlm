from pathlib import Path
import json,hashlib,sys,platform
import torch
base=Path(__file__).resolve().parents[1]
c=json.loads((base/'primary/qwen-config.json').read_bytes());m=json.loads((base/'primary/qwen-model-metadata.json').read_bytes());t=c['text_config'];v=c['vision_config']
h=t['hidden_size'];i=t['intermediate_size'];d=t['head_dim'];q=t['num_attention_heads']*d;k=t['num_key_value_heads']*d
assert t['attention_bias'] is False and c['tie_word_embeddings'] is True
text_parts={'tied_token_embedding':t['vocab_size']*h,'decoder_blocks':t['num_hidden_layers']*(h*q+2*h*k+q*h+2*d+3*h*i+2*h),'final_rmsnorm':h}
vh=v['hidden_size'];vi=v['intermediate_size'];merge=vh*v['spatial_merge_size']**2;out=v['out_hidden_size']
merger=lambda norm_dim:2*norm_dim+merge*merge+merge+merge*out+out
vision_parts={'patch_conv3d':vh*v['in_channels']*v['temporal_patch_size']*v['patch_size']**2+vh,'position_embeddings':v['num_position_embeddings']*vh,'vision_blocks':v['depth']*(4*vh*vh+4*vh+2*vh*vi+vi+vh+4*vh),'merger':merger(vh),'deepstack_mergers':len(v['deepstack_visual_indexes'])*merger(merge)}
text_total=sum(text_parts.values());vision_total=sum(vision_parts.values());total=text_total+vision_total
assert total==m['safetensors']['total']==m['safetensors']['parameters']['BF16']==2_127_532_032
print(json.dumps({'text_parts':text_parts,'text_total':text_total,'vision_parts':vision_parts,'vision_total':vision_total,'unique_parameter_total':total,'note':'Tied output lm_head reuses embedding; image/text projector merger and 3 deepstack mergers are included.'},indent=2))
for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
 b=torch.empty((),dtype=dtype).element_size();print(dtype,'bytes/number',b,'number_bytes',total*b,'GiB',total*b/1024**3,'rounded',round(total*b/1024**3,2))
# Bounded example for LoRA Eq.(3): the adapter alone does not retain W0x.
w=torch.tensor([[1.,2.],[3.,4.]]);a=torch.tensor([[.5,1.]]);b=torch.tensor([[1.],[2.]]);x=torch.tensor([2.,3.]);delta=b@a
assert torch.equal(w@x+delta@x,(w+delta)@x)
assert not torch.equal(delta@x,(w+delta)@x)
print('lora_tiny_base_plus_delta',(w@x+delta@x).tolist(),'delta_only',(delta@x).tolist())
print(json.dumps({'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_build':str(torch.version.cuda),'platform':platform.platform(),'network':'Only local primary snapshots read; no model/data downloads.'},indent=2))
