import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.architecture import _clone_config, _router_gradients
from scripts.course_experiments.common import new_lm, records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.modern import MoEFFN

torch.set_num_threads(1)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-7/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads())},
    "toy_cases": [],
}
for expert in (2.0, 1.0):
    for renormalize in (False, True):
        scores = torch.tensor([math.log(2), 0.0], dtype=torch.float64, requires_grad=True)
        probability = scores.softmax(-1)
        selected = probability.topk(1)
        p = selected.values[0]
        gate = p / p if renormalize else p
        output = gate * expert
        loss = output.square()
        loss.backward()
        expected = 0.0 if renormalize else 8 * expert**2 / 27
        torch.testing.assert_close(scores.grad, torch.tensor([expected, -expected], dtype=torch.float64), atol=1e-14, rtol=1e-14)
        result["toy_cases"].append({"expert_output": expert, "renormalize": renormalize, "probabilities": probability.detach().tolist(), "selected_index": selected.indices.item(), "indices_requires_grad": selected.indices.requires_grad, "selected_value_requires_grad": selected.values.requires_grad, "output": output.item(), "loss": loss.item(), "gradient": scores.grad.tolist(), "exact_gradient_magnitude": "0" if renormalize else ("32/27" if expert == 2 else "8/27")})

scores = torch.tensor([math.log(2), 0.0], dtype=torch.float64, requires_grad=True)
p = scores.softmax(-1)[0]
(2*p).square().backward()
updated = (scores - 0.01*scores.grad).detach().softmax(-1)
result["small_gradient_step"] = {"learning_rate": 0.01, "selected_probability_before": p.item(), "selected_probability_after": updated[0].item()}
assert updated[0] < p

a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
normalized = a/(a+b)
ga, gb = torch.autograd.grad(normalized, (a,b))
result["top2"] = {"weights_before": [0.7/0.9, 0.2/0.9], "weights_after": [0.8/1.0, 0.2/1.0], "partial_a": ga.item(), "partial_b": gb.item(), "derivation": "d(a/(a+b))/da=b/(a+b)^2，b>0 時非零；若共同倍率同時放大 a、b，比例不變，不能聲稱所有方向都有非零梯度。"}
assert ga.item() > 0
result["arithmetic"] = {"log_2": math.log(2), "exp_log_2": math.exp(math.log(2)), "exp_0": math.exp(0), "norm_3_4": math.hypot(3,4), "derivation": "p=2/3，dp/ds=(2/9,-2/9)；L=(E*p)^2，dL/ds=2*E^2*p*dp/ds=(8*E^2/27,-8*E^2/27)。E=2 得 32/27；E=1 得 8/27，恰為四分之一。p/p=1 在 p>0 時導數為零。"}

module_results=[]
for k in (1,2):
    torch.manual_seed(7)
    layer=MoEFFN(3,experts=4,top_k=k)
    x=torch.tensor([[[0.4,-0.2,0.7]]])
    output,aux,chosen=layer(x)
    grad=torch.autograd.grad(output.square().sum(),layer.router.weight)[0]
    module_results.append({"top_k": k,"selected":chosen.tolist(),"task_gradient_norm":grad.norm().item(),"all_finite":bool(torch.isfinite(grad).all())})
    assert grad.norm() > 0
result["repository_router_paths"]=module_results

experiment_path=ROOT/"docs/course-experiments/results/moe.json"
report=json.loads(experiment_path.read_text())
variant=report["results"]["variants"]["top1_aux0"]
result["recorded_experiment"]={"path":str(experiment_path.relative_to(ROOT)),"sha256":hashlib.sha256(experiment_path.read_bytes()).hexdigest(),"environment":{key:report[key] for key in ("revision","device","seed","torch_version","python_version","gpu","step_scale","evidence_status")},"model":variant["model"],"dataset":report["results"]["dataset"],"router_gradients_before":variant["router_gradients_before"],"router_gradients_after":variant["router_gradients_after"],"training":{key:variant["training"][key] for key in ("steps","optimizer_updates","skipped_updates","auxiliary_weight","batch_size","learning_rate","status")}}
assert variant["training"]["optimizer_updates"] == 180
assert variant["training"]["auxiliary_weight"] == 0
assert variant["router_gradients_before"]["effective_targets"] == variant["router_gradients_after"]["effective_targets"] == 1004
assert [round(v,5) for v in variant["router_gradients_before"]["task_gradient_norms"]] == [0.00668,0.00658]
assert [round(v,5) for v in variant["router_gradients_after"]["task_gradient_norms"]] == [0.02446,0.01400]
result["weighted_auxiliary_norms"]={str(weight):[weight*n for n in variant["router_gradients_after"]["unweighted_auxiliary_gradient_norms"]] for weight in (0,0.01)}

result["code_provenance"]=[]
for name in ("tiny_perceptron/modern.py","scripts/course_experiments/architecture.py","scripts/course_experiments/common.py","tiny_perceptron/data.py","tiny_perceptron/model.py"):
    p=subprocess.run(["git","show",report["revision"]+":"+name],capture_output=True)
    raw=(ROOT/name).read_bytes()
    result["code_provenance"].append({"path":name,"current_sha256":hashlib.sha256(raw).hexdigest(),"recorded_revision_available":p.returncode==0,"git_show_stderr":p.stderr.decode().strip(),"identical_to_recorded_revision":p.stdout==raw if p.returncode==0 else None})

asset=ROOT/"assets/training/tinystories-v1.tar.gz"
raw=asset.read_bytes()
dataset={"local_asset_sha256":hashlib.sha256(raw).hexdigest(),"local_asset_is_lfs_pointer":raw.startswith(b"version https://git-lfs.github.com/spec/v1")}
if dataset["local_asset_is_lfs_pointer"]:
    pointer=dict(line.split(" ",1) for line in raw.decode().splitlines())
    dataset["lfs_oid"]=pointer["oid"]
    dataset["lfs_size"]=int(pointer["size"])
    url="https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/assets/training/tinystories-v1.tar.gz"
    response=subprocess.run(["curl","-fsSL","--max-time","45",url],capture_output=True)
    dataset["fetch_command"]="curl -fsSL --max-time 45 "+url
    dataset["fetch_exit_code"]=response.returncode
    dataset["fetch_stderr"]=response.stderr.decode()
    if response.returncode==0:
        raw=response.stdout
        dataset["download_sha256"]=hashlib.sha256(raw).hexdigest()
        assert "sha256:"+dataset["download_sha256"] == pointer["oid"]
        assert len(raw) == int(pointer["size"])
try:
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
        names=[m for m in archive.getmembers() if m.name.endswith("tinystories-train-512.jsonl")]
        assert len(names)==1
        content=archive.extractfile(names[0]).read()
    dataset["jsonl_sha256"]=hashlib.sha256(content).hexdigest()
    records=[json.loads(line) for line in content.decode().splitlines() if line.strip()]
    for record in records:
        record["family"]=record.get("text_sha256",records_sha256([{"text":record["text"]}]))
    data=split_records(records,42)
    dataset["split_report"]={name:{"records":len(rows),"sha256":records_sha256(rows)} for name,rows in data.items()}
    assert dataset["split_report"]==report["results"]["dataset"]
    examples=text_examples(data["validation"],max_length=128)
    x,y,valid=pad_batch(examples[:8])
    dataset["probe"]={"example_lengths":[len(pair[0]) for pair in examples[:8]],"input_shape":list(x.shape),"effective_targets":int((y!=IGNORE).sum()),"valid_inputs":int(valid.sum()),"ignored_padding_targets":int((y==IGNORE).sum()),"tokenizer_test":ByteTokenizer().encode("A中"),"expected_utf8_bytes":list("A中".encode()),"byte_token_id_offset":8}
    assert dataset["probe"]["effective_targets"]==1004
    ctx=SimpleNamespace(seed=42,device="cpu")
    initial=new_lm(ctx,heads=4)
    top1,_=_clone_config(initial,ctx,experts=4,top_k=1)
    dataset["cpu_before_probe"]=_router_gradients(top1,examples,ctx)
    for observed,expected in zip(dataset["cpu_before_probe"]["task_gradient_norms"],variant["router_gradients_before"]["task_gradient_norms"],strict=True):
        assert math.isclose(observed,expected,abs_tol=1e-7,rel_tol=1e-4)
except tarfile.ReadError as exc:
    dataset["not_reproduced_reason"]=str(exc)
result["dataset_verification"]=dataset
result["limitations"]=["沒有重新執行 CUDA 訓練，180 次更新後的梯度以保存的原始實驗 JSON 與現行量測程式核對。","實驗記錄中的舊 commit 是否能取得，見 code_provenance；現行方法與舊執行程式逐位元相同與否不能由結果 JSON 證明。","非零梯度只表明當次任務目標存在學習訊號；不證明離散索引可微分、每個方向梯度非零或路由最優。"]
output=Path(__file__).with_name("verification.json")
with output.open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write("\n")
print(json.dumps({"saved":str(output),"toy_cases":len(result["toy_cases"]),"dataset_verification":dataset,"recorded_norms_after":variant["router_gradients_after"]},ensure_ascii=False))
