"""Bounded CPU string-rule check, not a model judge, training, or inference."""
import contextlib,hashlib,io,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from tiny_perceptron.data import ByteTokenizer,CharTokenizer
out=Path(__file__).resolve().parent
code=(out/'original/fence-1.py').read_text(encoding='utf-8')
assert hashlib.sha256(code.encode()).hexdigest()=='10eb199559d9c5b8b53795191b677161ca55611f6d53bfbd84b231dfc10cf188'
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
records=[]
def execute_case(name,source,expected_lengths,expected_chosen,expected_count):
 ns={}; stdout=io.StringIO()
 with contextlib.redirect_stdout(stdout):exec(compile(source,name,'exec'),ns)
 text=ns['long'];parts=text.split('。');trimmed=parts[:-1]
 record={'name':name,'executed_code':source,'stdout':stdout.getvalue(),'lengths':ns['lengths'],'chosen_equals_long':ns['chosen']==text,'raw_split':parts,'after_slice':trimmed,'unique_pieces':sorted(set(trimmed)),'unique_count':len(set(trimmed)),'ending_separator':text.endswith('。')}
 assert record['lengths']==expected_lengths,(name,record)
 assert record['chosen_equals_long']==expected_chosen,(name,record)
 assert record['unique_count']==expected_count,(name,record)
 records.append(record)
execute_case('original-10-repeats',code,[5,50],True,1)
execute_case('exercise-2-repeats',code.replace('short * 10','short * 2'),[5,10],True,1)
useful='答案是4；兩組各兩個物件合起來共有四個。'
# Count independently from a literal list of expected Unicode code points.
expected_characters=['答','案','是','4','；','兩','組','各','兩','個','物','件','合','起','來','共','有','四','個','。']
assert list(useful)==expected_characters
execute_case('exercise-useful-explanation',code.replace('short * 10',repr(useful)),[5,len(expected_characters)],True,1)
execute_case('boundary-remove-final-separator',code.replace('short * 10',repr(useful[:-1])),[5,19],True,0)
execute_case('lexical-paraphrase-is-not-information-count',code.replace('short * 10',repr('答案是4。答案是四。')),[5,10],True,2)
# An incorrect but equally long answer ties under len and is selected when placed first.
wrong='答案是5。'; equal_candidates=[wrong,'答案是4。']
equal_choice=max(equal_candidates,key=len)
assert list(map(len,equal_candidates))==[5,5] and equal_choice==wrong
records.append({'name':'equal-length-correct-and-incorrect','candidates':equal_candidates,'lengths':[5,5],'chosen':equal_choice,'details':'len-key max returns the first maximal item; changing 4 to 5 is not inspected for arithmetic correctness.'})
byte_tokenizer=ByteTokenizer();char_tokenizer=CharTokenizer('答案是4。'+useful)
unit_records=[]
for text in ['答案是4。','答案是4。'*2,'答案是4。'*10,useful]:
 ids=byte_tokenizer.encode(text);char_ids=char_tokenizer.encode(text)
 assert len(ids)==len(text.encode('utf-8'))
 assert byte_tokenizer.decode(ids)==text
 assert len(char_ids)==len(text)
 unit_records.append({'text':text,'unicode_code_points':len(text),'utf8_bytes':len(text.encode('utf-8')),'byte_tokenizer_tokens_without_specials':len(ids),'char_tokenizer_tokens_without_specials':len(char_ids)})
assert unit_records[0]['unicode_code_points']==5 and unit_records[0]['byte_tokenizer_tokens_without_specials']==13
result={'scope':'Only a hand-coded length rule and literal string processing. Useful content and arithmetic are reviewed separately by reading the text, not by these counts. No model judge was invoked. ByteTokenizer/CharTokenizer are local deterministic representations with no data/model loading.','cases':records,'count_units':unit_records,'environment':{'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'tiktoken':'not installed or executed; upstream documentation only'},'input_hashes':{'fence-1.py':hashlib.sha256((out/'original/fence-1.py').read_bytes()).hexdigest(),'tiny_perceptron/data.py':hashlib.sha256((Path(__file__).resolve().parents[4]/'tiny_perceptron/data.py').read_bytes()).hexdigest()}}
(out/'bounded-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for item in records:print(json.dumps({k:v for k,v in item.items() if k!='executed_code'},ensure_ascii=False))
print(json.dumps({'count_units':unit_records,'environment':result['environment']},ensure_ascii=False))
print('All bounded literal string checks passed.')
