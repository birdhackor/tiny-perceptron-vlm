"""Coordinator-authorized redisplay of the already unlocked unit; never prints future state."""
import argparse,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('session');a=p.parse_args();root=Path('.').resolve()
assert a.session.isalnum()
spec=importlib.util.spec_from_file_location('phase7_current',root/'docs/review-tools/phase7_review.py');tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
state=tool.read_json(root/'outputs/grouped-review'/a.session/'state.json');print(json.dumps(tool.display(state,root),ensure_ascii=False,indent=2))
