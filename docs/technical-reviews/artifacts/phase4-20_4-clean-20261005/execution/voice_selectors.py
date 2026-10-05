import ast,json,hashlib
from pathlib import Path
base=Path(__file__).resolve().parents[1]
p=Path("tiny_perceptron/natural_assistant.py");tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_evaluate")
assignments=[n for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"rows","audio_rows","audio_chat_rows"} for t in n.targets)]
manifest=json.loads((base/"inputs/manifest.json").read_text());questions=json.loads((base/"inputs/voice-question-sources.json").read_text());results=[]
for split in ["validation","test"]:
 ns={"manifest":manifest,"options":type("Options",(),{"split":split})()};exec(compile(ast.Module(body=sorted(assignments,key=lambda n:n.lineno),type_ignores=[]),str(p),"exec"),ns)
 assert all(r["task"]=="speech_chat" for r in ns["audio_chat_rows"]);assert len(ns["audio_chat_rows"])==4
 for r in ns["audio_chat_rows"]:
  assert r["semantic_rubric"] and r["answer"] is None
  original=next(q for q in questions["audio_rows"] if q["id"]==r["id"]);assert r["user"]=="".join(original["original_spaced_source_transcription"].split());assert r["semantic_rubric"]==original["semantic_rubric"]
 results.append({"split":split,"audio_rows":len(ns["audio_rows"]),"audio_chat_rows":len(ns["audio_chat_rows"]),"chat_task_types":sorted({r["task"] for r in ns["audio_chat_rows"]})})
print(json.dumps({"source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"AST_executed_lines":[n.lineno for n in sorted(assignments,key=lambda n:n.lineno)],"results":results,"scope":"Only original list-selection assignments executed; ASR, model generation, training and scores not executed."},ensure_ascii=False,indent=2))
