import subprocess,json
from pathlib import Path
base=[".venv/bin/python","scripts/selftrained/chat.py","--model-dir","outputs/p7-technical-e-cache/moe-joint","--asset-dir","outputs/p7-technical-e-cache/data","--repo","birdhackor/tiny-perceptron-course-models","--revision","979cdfacc588ad0536f1c64fff96f264571cf054","--prefix","selftrained/v2/moe-joint","--manifest-sha256","f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e","--device","cpu","--max-new-tokens","128","--threads","2"]
for task,img,extra,expected in [
 ("vision_clothing","images/vision/c98cb897d639c880fde261eafc6d953d.png",["--image-layout","docs/selftrained/examples/v2/vision_clothing.layout.json"],"這是短靴。"),
 ("vision_relation","images/vision/13913fb756b87b5c3e0f52e4ba194fec.png",["--image-layout","docs/selftrained/examples/v2/vision_relation.layout.json"],"左邊是褲子。"),
 ("ocr","images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png",["--roi","18,50,90,86"],"大小")]:
 out="docs/technical-reviews/artifacts/p7_technical_e/executions/19.6-"+task+"-result.json"
 cmd=base+["--messages","docs/selftrained/examples/v2/"+task+".messages.json","--task",task,"--image",img,"--output",out]+extra
 print("actual_command",json.dumps(cmd),flush=True)
 r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout,r.stderr,flush=True);assert r.returncode==0
 v=json.loads(Path(out).read_text());assert v["assistant"]==expected,(task,v)
 print("expected_text_match",task,expected)
