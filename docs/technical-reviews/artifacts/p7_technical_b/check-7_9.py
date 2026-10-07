import json,platform
from tiny_perceptron.data import ByteTokenizer,render_chat
t=ByteTokenizer();x,y=render_chat([{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]);assert int(y[-1])==t.eos_id
def stop(generated,budget):
 visible=[];seen=[]
 for token in generated[:budget]:
  seen.append(token)
  if token==t.eos_id:break
  visible.append(token)
 return {'visible_ids':visible,'visible_text':t.decode(visible),'seen':seen}
g=[t.encode('A')[0],t.eos_id,t.encode('B')[0]];orig=stop(g,2);assert orig['visible_text']=='A'
print(json.dumps({'python':platform.python_version(),'device':'cpu','last_target':int(y[-1]),'EOS':t.eos_id,'candidate_ids':g,'budget2_original':orig,'budget1':stop(g,1),'budget3':stop(g,3),'no_EOS_budget2':stop([73,74,75],2)},indent=2))
