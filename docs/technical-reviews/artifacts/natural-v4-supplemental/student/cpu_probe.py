"""Bounded offline software/numeric checks; no real model inference."""
import base64
import copy
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts import fetch_natural_release as fetch
from tiny_perceptron import natural_ui as ui
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

OUT = Path(__file__).resolve().parent
E = ROOT / "docs/natural-assistant/evidence/v4-research/student-selected-public-ui"
manifest = fetch.read_json(ROOT / "docs/natural-assistant/v4/public-release.json")
results = {"environment":{"python":sys.version,"torch":torch.__version__,"device":"cpu","platform":platform.platform()},"scope":"Offline validation of release verification, routing/state/upload rules and record arithmetic; injected runner only, no model inference or install"}

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for relative, expected in manifest["code_files"].items():
    assert digest(ROOT / relative) == expected
    pinned = subprocess.check_output(["git","show","59a1eda4ed7b6e8609892ec2b9013c821ac93e69:"+relative],cwd=ROOT)
    assert pinned == (ROOT / relative).read_bytes()
for relative in ["requirements-natural.txt","scripts/fetch_natural_release.py","docs/natural-assistant/v4/public-release.json"]:
    pinned = subprocess.check_output(["git","show","59a1eda4ed7b6e8609892ec2b9013c821ac93e69:"+relative],cwd=ROOT)
    assert pinned == (ROOT / relative).read_bytes()
results["pinned_checkout_current_program_manifest_requirements_identical"] = True
listed=subprocess.run([sys.executable,"scripts/fetch_natural_release.py","--manifest","docs/natural-assistant/v4/public-release.json","--list"],cwd=ROOT,capture_output=True,text=True)
assert listed.returncode==0
results["actual_cli_list"]={"exit_code":listed.returncode,"stdout":listed.stdout,"stderr":listed.stderr}

with tempfile.TemporaryDirectory(prefix="offline-probe-",dir=OUT) as tmp:
    tmp=Path(tmp)
    calls=[]
    def fixture_download(repo,name,*,revision,token):
        calls.append({"repo":repo,"name":name,"revision":revision,"token":token})
        item=next(x for x in manifest["files"] if x["path"]==name)
        return E/"fetched-public-files"/(item["output"]+".raw")
    target=fetch.fetch_release(manifest,tmp/"release",downloader=fixture_download)
    assert fetch.verify_release(manifest,target)==target
    before={p.name:p.read_bytes() for p in target.iterdir()}
    try: fetch.fetch_release(manifest,target,downloader=fixture_download)
    except ValueError as e: existing_error=str(e)
    else: raise AssertionError("Existing directory overwritten")
    assert before=={p.name:p.read_bytes() for p in target.iterdir()}
    extra=target/"photo.png";extra.write_bytes(b"unapproved")
    try:fetch.verify_release(manifest,target)
    except ValueError as e:extra_error=str(e)
    else:raise AssertionError("Extra files accepted")
    extra.unlink()
    card=target/"README.md";card.write_bytes(card.read_bytes()+b"x")
    try:fetch.verify_release(manifest,target)
    except ValueError as e:corruption_error=str(e)
    else:raise AssertionError("Corruption accepted")
    assert card.exists()
    results["release_checks"]={"fixture_download_calls":calls,"verified":True,"existing_directory_preserved":existing_error,"extra_file_rejected":extra_error,"corrupted_file_preserved_and_rejected":corruption_error}
    cpu=fetch.student_options(manifest,target,device="cpu",local_files_only=True)
    gpu=fetch.student_options(manifest,target,device="cuda")
    half=fetch.student_options(manifest,target,device="cuda",dtype="float16")
    assert cpu.adapter is None and cpu.dtype=="float32" and cpu.local_files_only
    assert gpu.dtype=="bfloat16" and half.dtype=="float16"
    results["student_options"]={"cpu":cpu.dtype,"cuda_default":gpu.dtype,"cuda_fallback":half.dtype,"adapter":str(cpu.adapter),"local_files_only":cpu.local_files_only}
    try:fetch.check_runtime(manifest)
    except ValueError as e:results["current_tiny_venv_expected_runtime_rejection"]=str(e)
    else:raise AssertionError("3.13 unexpectedly meets 3.12 contract")
    # A missing cache must fail offline even if any attempted network request would fail.
    with patch("requests.sessions.Session.request",side_effect=AssertionError("Network attempted")) as request:
        try:hf_hub_download("Qwen/Qwen3-VL-2B-Instruct","config.json",revision=manifest["base_model"]["revision"],cache_dir=tmp/"empty-cache",local_files_only=True,token=False)
        except LocalEntryNotFoundError as e:results["missing_cache_offline"]={"exception":type(e).__name__,"network_calls":request.call_count}
        else:raise AssertionError("Missing cache accepted")
    upload_results=[]
    payloads={}
    for fmt,ext in [("PNG","png"),("JPEG","jpg"),("WEBP","webp")]:
        stream=io.BytesIO();Image.new("RGB",(3,2),(30,80,120)).save(stream,format=fmt)
        payload={"kind":"image","filename":"small."+ext,"base64":base64.b64encode(stream.getvalue()).decode()}
        assert ui.upload_contents(payload)[0]==stream.getvalue()
        payloads[fmt]=payload
        upload_results.append({"format":fmt,"bytes":len(stream.getvalue()),"accepted":True})
    def audio_payload(samples):
        stream=io.BytesIO();sf.write(stream,np.zeros(samples,dtype=np.float32),16000,format="WAV",subtype="PCM_16")
        return {"kind":"audio","filename":"speech.wav","base64":base64.b64encode(stream.getvalue()).decode()}
    audio=audio_payload(480000);ui.upload_contents(audio)
    try:ui.upload_contents(audio_payload(480001))
    except ValueError as e:duration_error=str(e)
    else:raise AssertionError("Over 30-second audio accepted")
    frames=[Image.new("RGB",(2,2),color) for color in ["red","blue"]]
    stream=io.BytesIO();frames[0].save(stream,format="WEBP",save_all=True,append_images=frames[1:],duration=100)
    try:ui.upload_contents({"kind":"image","filename":"animated.webp","base64":base64.b64encode(stream.getvalue()).decode()})
    except ValueError as e:animation_error=str(e)
    else:raise AssertionError("Animated image accepted")
    results["upload_rules"]={"static_image_formats":upload_results,"30_seconds_accepted":True,"over_30_seconds_rejected":duration_error,"animated_webp_rejected":animation_error,"max_upload_bytes":ui.MAX_UPLOAD_BYTES,"max_image_pixels":ui.MAX_IMAGE_PIXELS,"audio_extensions":sorted(ui.AUDIO_TYPES)}
    seen=[]
    def injected_generate(model,processor,row,data_root,options):
        seen.append(copy.deepcopy(row));return {"prediction":"CPU wiring fixture reply"}
    def injected_load_asr(options):return object(),object()
    def injected_transcribe(model,processor,path):return {"transcript":"我吃辣"}
    server=ui.create_server(None,None,cpu,tmp/"ui-data","127.0.0.1",0)
    upload_dir=Path(server.upload_directory.name)
    try:
        with patch.object(ui.assistant,"generate",injected_generate),patch.object(ui.assistant,"load_asr",side_effect=injected_load_asr) as lazy,patch.object(ui.assistant,"transcribe",injected_transcribe):
            session=server.operation("/api/session",{})["session"]
            image=server.operation("/api/upload",dict(payloads["PNG"],session=session))["asset"]
            voice=server.operation("/api/upload",dict(audio_payload(1600),session=session))["asset"]
            assert lazy.call_count==0
            server.operation("/api/chat",{"session":session,"prompt":"第一問","image":image})
            server.operation("/api/transcribe",{"session":session,"audio":voice})
            assert len(seen)==1 and lazy.call_count==1
            submitted=server.operation("/api/chat",{"session":session,"prompt":"我不吃辣","speech":voice})
            assert submitted["asr"]=={"transcript":"我吃辣","submitted_text":"我不吃辣","corrected":True}
            assert any(p.get("type")=="image" for turn in seen[1]["history"] for p in turn["content"])
            assert len(list(upload_dir.iterdir()))==2
            server.operation("/api/reset",{"session":session})
            assert server.sessions[session]=={"history":[],"assets":{},"transcriptions":{}}
            assert not list(upload_dir.iterdir())
            # A later upload is also deleted by actual server_close().
            server.operation("/api/upload",dict(payloads["PNG"],session=session))
    finally:server.server_close()
    assert not upload_dir.exists()
    results["injected_runner_state_check"]={"generated_calls":len(seen),"prior_image_in_second_history":True,"asr_lazy_load_calls":1,"transcribe_did_not_generate":True,"edited_text":submitted["asr"],"reset_removed_two_assets_and_history":True,"close_removed_upload_directory":True,"model_quality_verified":False}

trial=json.loads((E/"actual-ui/report.json").read_text())
browser=json.loads((E/"actual-ui/browser/report.json").read_text())
cache=json.loads((ROOT/"docs/natural-assistant/evidence/v4-research/student-local-cache-prepare/cache-prepare-receipt.json").read_text())
files=[f for snapshot in cache["prepared_snapshots"].values() for f in snapshot["files"]]
snapshot_bytes=sum(f["bytes"] for f in files)
assert len(files)==23 and snapshot_bytes==5_889_111_977
observer=[json.loads(line) for line in (E/"actual-ui/observer.jsonl").read_text().splitlines()]
server_resources=next(x for x in observer if x["kind"]=="process_resources")
assert server_resources["maximum_resident_set_kib"]==trial["server_cpu_resources"]["maximum_resident_set_kib"]==13_496_104
chat=[r for r in browser["requests"] if r["route"]=="/api/chat"]
asr=[r for r in browser["requests"] if r["route"]=="/api/transcribe"]
assert len(chat)==4 and len(asr)==1
assert chat[2]["response"]["asr"]["corrected"]
assert trial["torch_threads"]==5 and trial["torch_interop_threads"]==1
values={"public_files":len(manifest["files"]),"public_bytes":sum(x["bytes"] for x in manifest["files"]),"snapshot_files":len(files),"snapshot_bytes":snapshot_bytes,"snapshot_GiB":round(snapshot_bytes/1024**3,4),"server_peak_KiB":server_resources["maximum_resident_set_kib"],"server_peak_GiB":round(server_resources["maximum_resident_set_kib"]/1024**2,4),"ready_seconds":round(trial["ready_seconds"],2),"chat_seconds":[round(x,2) for x in browser["chat_seconds"]],"first_asr_seconds":round(browser["asr_seconds_including_load"],2),"workflow_seconds":round(trial["wall_seconds"],2),"8_MiB_bytes":8*1024**2,"8_MiB_decimal_MB":round(8*1024**2/10**6,2),"float32_element_bits":torch.tensor([1.0],dtype=torch.float32).element_size()*8,"float16_element_bits":torch.tensor([1.0],dtype=torch.float16).element_size()*8}
assert values=={"public_files":2,"public_bytes":9262,"snapshot_files":23,"snapshot_bytes":5889111977,"snapshot_GiB":5.4847,"server_peak_KiB":13496104,"server_peak_GiB":12.8709,"ready_seconds":24.24,"chat_seconds":[3.42,42.87,36.58,37.11],"first_asr_seconds":18.03,"workflow_seconds":170.33,"8_MiB_bytes":8388608,"8_MiB_decimal_MB":8.39,"float32_element_bits":32,"float16_element_bits":16}
results["record_audit_recalculation"]=values
results["empirical_denominators"]={"cpu_browser_workflows":1,"public_files":2,"official_snapshots":2,"snapshot_files":23,"chat_requests":4,"transcribe_requests":1,"input_photos":1,"input_recordings":1,"reset_requests":1,"core_loads":1,"asr_loads":1,"seed":42,"warmup_runs":0,"gpu_runs_by_this_reviewer":0,"real_model_inference_runs_by_this_reviewer":0}
results["status"]="passed"
print(json.dumps(results,ensure_ascii=False,indent=2))
