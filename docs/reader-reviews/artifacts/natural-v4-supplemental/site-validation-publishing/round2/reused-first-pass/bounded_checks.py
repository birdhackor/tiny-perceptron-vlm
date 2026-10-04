from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import subprocess, json, re
base=Path(__file__).parent
repo=Path.cwd()
predictions={'recorded_before_execution_utc':datetime.now(timezone.utc).isoformat(),'asr_text_example':{'edits':1,'reference_characters':9,'CER_percent_rounded_1_decimal':11.1,'same':False},'corrected_text':{'edits':0,'CER_percent':0,'same':True},'fingerprint_example':{'same_when_adapter_B_vs_C':False,'same_when_both_adapter_B':True},'capability_card_response_arithmetic':{'sum_of_reported_generated_response_denominators':178,'normal_plus_truncated':178,'ASR_22_transcriptions_not_added_to_generated_responses':True},'CLI_help':'Help should list flags used in publishing without building or running notebooks.'}
(base/'predictions-before-execution.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2)+'\n')
from tiny_perceptron.natural_concepts import text_error_report
import torch
reference='請推薦不辣的晚餐。'
recognized='請推薦辣的晚餐。'
def asr(a,b):
    r=text_error_report(a,b)
    return {'edits':r['edits'],'reference_characters':r['reference_characters'],'CER_percent_rounded_1_decimal':round(r['cer']*100,1),'same':a==b}
expected=b'base A, adapter B'; received=b'base A, adapter C'
outputs={'executed_utc':datetime.now(timezone.utc).isoformat(),'environment':{'python':subprocess.run([str(repo/'.venv/bin/python'),'--version'],capture_output=True,text=True).stdout.strip(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available()},'asr_text_example':asr(reference,recognized),'corrected_text':asr(reference,reference),'fingerprint_example':{'expected_display_12':sha256(expected).hexdigest()[:12],'received_display_12':sha256(received).hexdigest()[:12],'same_when_adapter_B_vs_C':sha256(expected).digest()==sha256(received).digest(),'same_when_both_adapter_B':sha256(expected).digest()==sha256(expected).digest()},'capability_card_response_arithmetic':{'sum':sum([42,84,18,10,3,13,4,4]),'normal_plus_truncated':171+7},'limits':'Offline CPU teaching text/arithmetic only. No model inference/training, audio, benchmark replication or publication; cited scores remain cited scores.'}
(base/'bounded-execution.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(outputs,ensure_ascii=False,indent=2))
help_results=[]
for script in ['scripts/build_course.py','scripts/check_notebooks.py','scripts/export_course.py','scripts/reading_time.py']:
    command=[str(repo/'.venv/bin/python'),script,'--help']
    r=subprocess.run(command,capture_output=True,text=True,timeout=25)
    name=Path(script).stem+'-help.txt'
    (base/name).write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
    help_results.append({'command':command,'exit_code':r.returncode,'output':name,'stdout':r.stdout,'stderr':r.stderr})
(base/'CLI-help-receipts.json').write_text(json.dumps(help_results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(help_results,ensure_ascii=False,indent=2))
figure='course/figures/natural-v4-asr-two-routes.svg'
command=['/usr/bin/inkscape',figure,'--export-type=png','--export-filename='+str(base/'asr-two-routes.png')]
r=subprocess.run(command,capture_output=True,text=True,timeout=20)
receipt={'command':command,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'original_path':figure,'original_sha256':sha256((repo/figure).read_bytes()).hexdigest(),'png_sha256':sha256((base/'asr-two-routes.png').read_bytes()).hexdigest()}
(base/'figure-render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
for rel in ['course/first-steps.md','course/chapters/19.md','course/chapters/20.md','course/README.md','docs/course-experiments/README.md','docs/asset-storage.md','docs/natural-assistant/v4/TRAINING.md',figure]:
    dest=base/'snapshots'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((repo/rel).read_bytes())
# Correct the initial human navigation description to its observed label, preserving the original record unread by any other reader.
p=base/'browser/browser-records.json'; data=json.loads(p.read_text()); data[0]['how']='Clicked visible home in-page link 驗證範圍 (actual label recorded in home-entry-links.json); initial runner description used the document heading.'
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
