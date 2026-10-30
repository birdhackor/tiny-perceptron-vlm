"""只對原pytest timeout probe在取得fs後觸發GC；不改原test／SDK／timeout／取消斷言。"""
import gc,json,threading
from pathlib import Path

HERE=Path(__file__).resolve().parent
seen=set();events=[]

def trace(frame,event,arg):
    if event=='line' and frame.f_code.co_name=='probe' and frame.f_code.co_filename.endswith('tests/test_omni_mature_modal_sdk.py') and 'fs' in frame.f_locals:
        key=(id(frame),frame.f_locals.get('operation'),frame.f_code.co_firstlineno)
        if key not in seen:
            seen.add(key)
            events.append({'file':frame.f_code.co_filename,'line':frame.f_lineno,'operation':frame.f_locals.get('operation'),'gc_collected':gc.collect()})
    return trace

def pytest_configure(config):
    threading.settrace_all_threads(trace)

def pytest_unconfigure(config):
    threading.settrace_all_threads(None)
    (HERE/'exact-original-pytest-gc-trace.json').write_text(json.dumps({'control':'Only schedule gc.collect immediately after original fs assignment; all original pytest assertions and SDK methods unchanged.','events':events},indent=2)+'\n')
