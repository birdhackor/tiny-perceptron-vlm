import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
p=Path("scripts/selftrained/train_local_stage.py")
spec=importlib.util.spec_from_file_location("wrapper",p);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
spec=importlib.util.spec_from_file_location("trainer","scripts/selftrained/train.py");t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
args=t.parser().parse_args(["--records","outputs/p7-technical-e-cache/data/text-tools-v2-train.jsonl","--asset-dir","outputs/p7-technical-e-cache/data","--output-dir","checkpoints/selftrained-local/moe-native","--stage","joint","--architecture","moe","--steps","4000","--batch-size","16","--context","512","--seed","20261006","--learning-rate","0.0002","--eval-every","1000","--save-every","1000","--freeze-perception-backbones","--sampling-mode","task-family","--tool-loss-weight","4","--numeric-run-loss-weight","1","--native-voice-loss-weight","4","--init-checkpoint","checkpoints/selftrained-local/moe-weighted/best.pt","--device","cpu"])
raw=json.loads(Path("docs/selftrained/results/training-raw/moe-native/raw/execution.json").read_text())["job"]
for k in ["stage","architecture","steps","batch_size","context","seed","learning_rate","eval_every","save_every","freeze_perception_backbones","sampling_mode","tool_loss_weight","numeric_run_loss_weight","native_voice_loss_weight"]:
 assert getattr(args,k)==raw[k],(k,getattr(args,k),raw[k])
 print(k,getattr(args,k))
prefix=[sys.executable,str(p),"--manifest","docs/selftrained/v2-manifest.json","--data-root","outputs/p7-technical-e-cache/data"]
for tail,needle in [(["--manifest-sha256","0"*64,"--","--output-dir","outputs/p7-technical-e-cache/never-created"],"Manifest differs"),(["--","--records","x"],"provided by the frozen manifest")]:
 r=subprocess.run(prefix+tail,capture_output=True,text=True);assert r.returncode!=0 and needle in r.stderr;print("expected guard",needle,"exit",r.returncode)
with tempfile.TemporaryDirectory() as d:
 root=Path(d);f=root/"a.txt";f.write_text("real bytes");item={"path":"a.txt","bytes":f.stat().st_size,"sha256":w.digest(f)}
 assert w.verified_file(root,item)==f
 f.write_text("changed")
 try:w.verified_file(root,item)
 except ValueError as e: print("expected byte/SHA guard",str(e))
 else:raise AssertionError("mutation accepted")
print("native_parser_matches_historical_job_15_fields_no_training_started")
