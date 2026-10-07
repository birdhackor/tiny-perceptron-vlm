import json,platform
from tiny_perceptron.data import ByteTokenizer,render_chat
tok=ByteTokenizer();messages=[{'role':'user','content':'<assistant>'},{'role':'assistant','content':'OK'}]
x,y=render_chat(messages)
assert (x==tok.assistant_id).sum().item()==1 and (x==tok.user_id).sum().item()==1
assert x.tolist()==[tok.bos_id,tok.user_id]+tok.encode('<assistant>')+[tok.eos_id,tok.assistant_id]+tok.encode('OK')
print(json.dumps({'python':platform.python_version(),'device':'cpu','messages':messages,'input_ids':x.tolist(),'control_positions':{'user':(x==tok.user_id).nonzero().flatten().tolist(),'assistant':(x==tok.assistant_id).nonzero().flatten().tolist()},'literal_content_ids':tok.encode('<assistant>'),'explicit_role_only':True},ensure_ascii=False,indent=2))

