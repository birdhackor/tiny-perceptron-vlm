import json
import sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from tiny_perceptron.model import TinyLM, ModelConfig

output = Path(sys.argv[1])
torch.set_num_threads(1)
model = TinyLM(ModelConfig(width=8)).eval()
model_identity = id(model)
records = []
inputs = [("original", torch.tensor([[1, 2, 3]])), ("twelve_ids", torch.arange(1,13)[None])]
with torch.no_grad():
    for round_no in range(1,4):
        for label, ids in inputs:
            for _ in range(3):
                model(ids)
            with profile(activities=[ProfilerActivity.CPU]) as p:
                for _ in range(5):
                    result = model(ids)
            print("輸入", label, "輪次", round_no, "同一模型", id(model) == model_identity)
            print("分數形狀", tuple(result["logits"].shape))
            print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
            weight_bytes = sum(v.numel() * v.element_size() for v in model.parameters())
            print("權重bytes", weight_bytes)
            rows = {event.key: {"self_cpu_time_total_us": event.self_cpu_time_total, "cpu_time_total_us": event.cpu_time_total, "calls": event.count} for event in p.key_averages()}
            records.append({"input": label, "round": round_no, "model_identity": model_identity, "same_model": id(model) == model_identity, "input_shape": list(ids.shape), "logits_shape": list(result["logits"].shape), "weight_bytes": weight_bytes, "rows": rows})
receipt = {"execution_scope": "current exercise in one CPU process; same model retained across all six profiles; original and twelve_ids paired for each round", "environment": {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(), "model_device": str(next(model.parameters()).device), "profile_activity": "CPU"}, "runs": records}
(output / "exercise-observations.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
