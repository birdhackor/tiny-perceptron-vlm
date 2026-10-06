from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from scripts.selftrained.evaluate import normalize,score_reply
rows=[]
for task in ['text','ocr']:
    record={'task':task,'messages':[{'role':'user','content':'讀字'},{'role':'assistant','content':'大小'}],'supervision':{'ocr_text':'大小'} if task=='ocr' else {}}
    output='大 小！'
    rows.append({'task':task,'target':'大小','output':output,'literal_equal':output=='大小','normalized_equal':normalize(output)==normalize('大小'),'score':score_reply(record,output,{'tool_call':None,'status':'no_tool','executed':False,'tool_result':None})})
print(json.dumps(rows,ensure_ascii=False,indent=2))
