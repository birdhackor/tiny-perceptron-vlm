"""Independent bounded CPU verification; no neural model generation or training."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import torch

from tiny_perceptron.selftrained.tools import (
    CalculatorCall, ToolCallError, execute_tool_call, parse_tool_call,
    run_tool_loop, serialize_tool_result,
)
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, generation_report

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text())

print('ENVIRONMENT', json.dumps({'python':platform.python_version(),'torch':torch.__version__,
    'device':'cpu','neural_generation':'not performed','git_head':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}, ensure_ascii=False))
print('COMMAND /workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.7/verify.py')
chapter=(ROOT/'course/chapters/19.md').read_text()
section=re.search(r'^## 19\.7 .*?(?=^## )',chapter,re.M|re.S).group()
assert digest(A/'inputs/19.7.md')==hashlib.sha256(section.encode()).hexdigest()
code=re.search(r'```python\n(.*?)```',section,re.S).group(1)
print('ORIGINAL_FENCE', code)
exec(compile(code,'course/chapters/19.md#19.7','exec'),{})
assert execute_tool_call(parse_tool_call('{"tool":"calculator","operation":"multiply","a":26,"b":16}'))['result']==416
assert 26*16 == (20+6)*16 == 320+96 == 416
print('HAND_CALCULATION (20+6)*16 = 320+96 = 416; 25*16 = 400')

valid=[('different_legal_parameters',CalculatorCall('multiply',25,16),400),
       ('minimum',CalculatorCall('add',0,0),0),
       ('maximum',CalculatorCall('multiply',99,99),9801),
       ('negative_result',CalculatorCall('subtract',0,99),-99)]
for name,call,expected in valid:
    actual=execute_tool_call(call);assert actual['result']==expected
    print('VALID_VARIANT',name,json.dumps(call.to_dict()),json.dumps(actual))
invalid={
    'above_range':'{"tool":"calculator","operation":"add","a":100,"b":0}',
    'negative_operand':'{"tool":"calculator","operation":"add","a":-1,"b":0}',
    'boolean_operand':'{"tool":"calculator","operation":"add","a":true,"b":0}',
    'float_operand':'{"tool":"calculator","operation":"add","a":1.0,"b":0}',
    'unsupported_operation':'{"tool":"calculator","operation":"divide","a":1,"b":1}',
    'duplicate_key':'{"tool":"calculator","operation":"add","a":1,"a":2,"b":0}',
    'extra_key':'{"tool":"calculator","operation":"add","a":1,"b":0,"extra":0}',
    'missing_parameter':'{"tool":"calculator","operation":"add","a":1}',
    'expression_string':'{"tool":"calculator","operation":"add","a":"26*16","b":0}',
    'not_json':'__import__("os").getcwd()',
}
for name,text in invalid.items():
    try:parse_tool_call(text)
    except ToolCallError as exc:print('INVALID_VARIANT',name,type(exc).__name__,str(exc))
    else:raise AssertionError(name)

argv=read(A/'raw/tool_call-1-argv.json')
result=read(A/'raw/tool_call-1-result.json')
stdout=read(A/'raw/tool_call-1-stdout.txt')
assert argv['argv']==result['argv']
assert argv['argv'][argv['argv'].index('--device')+1]=='cpu'
assert '--tools' in argv['argv']
assert read(ROOT/'docs/selftrained/examples/v2/tool_call.messages.json')==argv['public_messages_before_execution']
assert result['returncode']==0
assert digest(A/'raw/tool_call-1-stdout.txt')==result['stdout_sha256']
assert digest(A/'raw/tool_call-1-stderr.txt')==result['stderr_sha256']
assert stdout['generations']==result['actual_generations']
assert stdout['tool_trace']==result['actual_tool_trace']
assert stdout['answer']==result['actual_answer']=='結果是416。'
assert stdout['model']['config']['architecture']=='moe'
assert stdout['public_source']['revision']=='979cdfacc588ad0536f1c64fff96f264571cf054'
catalog=read(A/'raw/selection-catalog.json')
selected=catalog['examples'][4]
for source,expected_hash in catalog['CLI_source_hashes'].items():
    assert digest(ROOT/source)==expected_hash
    print('ORIGINAL_CLI_SOURCE_MATCH',source,expected_hash)
validation=read(A/'raw/validation-tool_call.jsonl')
assert digest(A/'raw/validation-tool_call.jsonl')==selected['source_line_sha256']
assert validation['record']==selected['record']
assert validation['trace']==selected['actual_trace']
assert selected['record']['split']=='validation'
assert selected['record']['messages'][:-1]==argv['public_messages_before_execution']
assert selected['record']['supervision']['expected_call']==stdout['tool_trace']['tool_call']
assert selected['record']['supervision']['expected_result']==stdout['tool_trace']['tool_result']['result']==416
assert validation['trace']['initial_output']==stdout['tool_trace']['initial_output']
assert validation['trace']['final_output']==result['saved_GPU_validation_answer']==stdout['answer']
print('SELECTED_RAW_VALIDATION_PROVENANCE',json.dumps({'catalog_pointer':'/examples/4',
    'record_id':selected['record_id'],'split':selected['record']['split'],
    'source_line_number':selected['source_line_number'],'source_line_sha256':selected['source_line_sha256'],
    'CPU_input_equals_validation_public_history':True,
    'previous_raw_final_output':validation['trace']['final_output']},ensure_ascii=False))
tokenizer=CharacterTokenizer.load(A/'raw/tokenizer.json')
assert digest(A/'raw/tokenizer.json')==stdout['model']['files']['tokenizer.json']
gens=stdout['generations'];assert len(gens)==2
assert gens[0]['prompt_messages']==argv['public_messages_before_execution']
assert gens[0]['model_sha256']==gens[1]['model_sha256']==stdout['model']['files']['model.safetensors']
call=parse_tool_call(gens[0]['raw_output']);tool=execute_tool_call(call)
assert tool==stdout['tool_trace']['tool_result']=={'tool':'calculator','ok':True,'result':416}
expected_history=gens[0]['prompt_messages']+[
    {'role':'assistant','content':gens[0]['raw_output']},
    {'role':'tool','content':serialize_tool_result(tool)},
]
assert gens[1]['prompt_messages']==expected_history
assert stdout['messages']==expected_history+[{'role':'assistant','content':gens[1]['raw_output']}]
for i,g in enumerate(gens):
    report=generation_report(tokenizer,g['generated_ids'])
    assert report['answer']==g['raw_output']
    assert report['eos'] and not report['invalid_special_tokens']
    assert g['generated_ids'][-1]==tokenizer.eos_id==2
    assert g['generation_status']=='eos' and g['eos']
    assert g['prompt_unknown_tokens']==0
    print('ARCHIVED_GENERATION',i+1,'tokens',len(g['generated_ids']),
          json.dumps({'raw_output':g['raw_output'],'eos':report['eos'],
                      'model_sha256':g['model_sha256'],'prompt_messages':g['prompt_messages']},ensure_ascii=False))

# Callback replay verifies the runtime protocol, not the model's ability.
for enabled in (True,False):
    seen=[]
    def callback(messages):
        seen.append(messages)
        if len(seen)==1:return gens[0]['raw_output']
        actual=json.loads(messages[-1]['content'])
        return f"結果是{actual['result']}。" if actual['ok'] else '工具未開，請先開啟。'
    trace=run_tool_loop(callback,argv['public_messages_before_execution'],tools_enabled=enabled)
    assert len(seen)==2
    assert trace['executed']==enabled
    assert trace['status']==('executed' if enabled else 'unavailable')
    assert seen[1][-2]=={'role':'assistant','content':gens[0]['raw_output']}
    print('CALLBACK_PROTOCOL_ONLY',enabled,json.dumps(trace,ensure_ascii=False))

replacement={'tool':'calculator','ok':True,'result':417}
assert json.loads(serialize_tool_result(replacement))==replacement
assert replacement!=tool
print('COUNTERFACTUAL_SERIALIZATION_ONLY',serialize_tool_result(replacement),
      'test-only replacement; no neural dependence or calculator correctness claimed')

# Read only the raw tool aggregate and its threshold; do not rerun test inference.
metrics=read(A/'raw/moe-test-metrics.json')
tool_metric=metrics['per_task_final_reply']['tool_call']
for field in ('tool_valid','tool_executed','tool_parameters','tool_final','tool_roundtrip'):
    item=tool_metric[field]
    assert item['denominator']==tool_metric['count']==276
    assert item['rate']==item['numerator']/item['denominator']
assert tool_metric['tool_roundtrip']['rate']==0 < metrics['thresholds']['tool_roundtrip']==0.95
print('ARCHIVED_AGGREGATE_ONLY',json.dumps({'split':metrics['split'],'tool_call':tool_metric,
    'threshold':metrics['thresholds']['tool_roundtrip']},ensure_ascii=False))

for source in ('tiny_perceptron/selftrained/tools.py','tiny_perceptron/selftrained/inference.py',
               'tiny_perceptron/selftrained/model.py','scripts/selftrained/prepare_text_tools.py',
               'scripts/selftrained/evaluate.py'):
    p=ROOT/source;t=ast.parse(p.read_text())
    inventory=[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(t)
        if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    print('SOURCE_AST',source,digest(p),json.dumps(inventory))
print('RESULT PASS: bounded executor, arithmetic, archived CPU trace, decoding/EOS, callback protocol, and tool aggregate checked; no new neural generation or test evaluation.')
