import json
from tiny_perceptron.data import render_chat
question="2+2=?"
answers={"簡短":"4","生動":"4，就像兩雙筷子共有四根。"}
out={}
for style,answer in answers.items():
 inputs,labels=render_chat([{"role":"user","content":question},{"role":"assistant","content":answer}])
 out[style]={"input_count":len(inputs),"learn_count":int((labels!=-100).sum()),"answer_utf8_bytes":len(answer.encode()),"active_labels":labels[labels!=-100].tolist()}
 print(style,"總位置",len(inputs),"學習位置",int((labels!=-100).sum()))
inputs,labels=render_chat([{"role":"user","content":question},{"role":"assistant","content":answers["生動"]+"好"}]);out["variation_append_好"]={"input_count":len(inputs),"learn_count":int((labels!=-100).sum())}
assert out["簡短"]["learn_count"]==2 and out["生動"]["learn_count"]==len(answers["生動"].encode())+1
print(json.dumps(out,ensure_ascii=False))
