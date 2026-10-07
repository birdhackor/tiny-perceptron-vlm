import codecs,inspect,json,platform
from pathlib import Path
raw='貓🙂'.encode();assert list(raw)==[232,178,147,240,159,153,130]
rows=[]
for n in [1,3,4,7]:
 value=raw[:n].decode('utf-8',errors='replace');rows.append({'bytes_taken':n,'display':value,'original_slice':list(raw[:n])})
assert [r['display'] for r in rows]==['�','貓','貓�','貓🙂']
try:raw[:1].decode('utf-8');raise AssertionError('expected strict error')
except UnicodeDecodeError as e:err={'start':e.start,'end':e.end,'reason':e.reason}
independent=''.join(bytes([b]).decode('utf-8',errors='replace') for b in raw);assert independent!='貓🙂'
dec=codecs.getincrementaldecoder('utf-8')();stream=[dec.decode(bytes([b]),final=False) for b in raw];stream.append(dec.decode(b'',final=True));assert ''.join(stream)=='貓🙂'
replacement=raw[:1].decode('utf-8',errors='replace').encode();assert replacement!=raw[:1]
base=Path(__file__).resolve().parent;(base/'sources/python-incremental-utf8.py').write_text(inspect.getsource(codecs.BufferedIncrementalDecoder)+'\n'+inspect.getsource(codecs.getincrementaldecoder('utf-8')))
print(json.dumps({'python':platform.python_version(),'device':'cpu','bytes':list(raw),'prefix_observations':rows,'strict_error':err,'full_decode':raw.decode(),'one_byte_independent_decode':independent,'incremental_parts':stream,'incremental_join':''.join(stream),'replacement_resave_bytes':list(replacement),'source_bytes_unchanged':raw=='貓🙂'.encode()},ensure_ascii=False,indent=2))

