"""Execute actual project loader/routing control flow with provider calls mocked.

This verifies wiring, not pretrained model API loading or inference/ASR quality.
No Transformers/PEFT installation, model download, or audio/GPU inference.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import json
import platform
import types
from unittest.mock import patch
import torch
import tiny_perceptron.natural_assistant as core

trace = []
class Model(torch.nn.Module):
    def __init__(self, role):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(2,2))
        self.role = role
        self.config = types.SimpleNamespace(use_cache=True)
    def requires_grad_(self, state=True):
        trace.append({"call":"requires_grad_", "role":self.role, "state":state})
        return super().requires_grad_(state)
    def gradient_checkpointing_enable(self, **kwargs):
        trace.append({"call":"gradient_checkpointing_enable", "kwargs":kwargs})

def provider(role):
    class Provider:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            trace.append({"call":"from_pretrained", "role":role, "model_id":model_id,
                          "kwargs":{k:str(v) for k,v in kwargs.items()}})
            return types.SimpleNamespace(role=role) if "processor" in role else Model(role)
    return Provider

class LoraConfig:
    def __init__(self, **kwargs):
        self.kwargs=kwargs
        trace.append({"call":"LoraConfig", "kwargs":kwargs})
def add_lora(model, config):
    model.lora_A=torch.nn.Parameter(torch.ones(1,2))
    trace.append({"call":"get_peft_model", "base_frozen":not model.base_weight.requires_grad})
    return model
class PeftModel:
    @staticmethod
    def from_pretrained(model, adapter, is_trainable):
        trace.append({"call":"PeftModel.from_pretrained", "adapter":adapter,"is_trainable":is_trainable,
                      "base_frozen":not model.base_weight.requires_grad})
        model.lora_A=torch.nn.Parameter(torch.ones(1,2),requires_grad=is_trainable)
        return model

transformers = types.SimpleNamespace(AutoProcessor=provider("vision_text_processor"),
                                    Qwen3VLForConditionalGeneration=provider("vision_text_core"),
                                    WhisperProcessor=provider("ASR_processor"),
                                    WhisperForConditionalGeneration=provider("ASR_model"))
peft=types.SimpleNamespace(LoraConfig=LoraConfig,TaskType=types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
                           get_peft_model=add_lora,PeftModel=PeftModel)
options=types.SimpleNamespace(model=core.MODEL_ID,model_revision=core.MODEL_REVISION,
    asr_model=core.ASR_VARIANTS["turbo"][0],asr_revision=core.ASR_VARIANTS["turbo"][1],
    device="cpu",dtype="float32",cache_dir="unused-no-download",local_files_only=True,
    min_pixels=65536,max_pixels=524288,lora_rank=8,adapter=None,max_seconds=2,
    manifest="intercepted",data_root="intercepted",output="intercepted",split="validation")
with patch.dict(sys.modules,{"transformers":transformers,"peft":peft}):
    base, processor=core.load_core(options)
    candidate, _=core.load_core(options,train=True)
    restored, _=core.load_core(options,adapter="explicit-candidate-path",train=False)
    asr, asrprocessor=core.load_asr(options)
assert not base.base_weight.requires_grad
assert not candidate.base_weight.requires_grad and candidate.lora_A.requires_grad
assert candidate.config.use_cache is False
assert not restored.lora_A.requires_grad
lc=next(r for r in trace if r["call"]=="LoraConfig")
assert lc["kwargs"]["target_modules"] == core.LORA_TARGETS
assert lc["kwargs"]["r"]==8 and lc["kwargs"]["lora_alpha"]==16

row={"id":"software-route-witness", "split":"validation", "task":"speech_chat",
     "audio":"intercepted", "user":"reference-only-text"}
manifest={"rows":[],"audio_rows":[row]}
calls=[]
def generated(model, proc, actual, *_):
    calls.append({"same_core":model is base,"same_processor":proc is processor,
                  "task":actual["task"],"user":actual["user"]})
    return {"id":actual["id"],"task":actual["task"]}
def transcript(model,proc,path):
    assert model is asr and proc is asrprocessor
    return {"transcript":"observed-hypothesis-only", "truncated":False,"completion_unknown":False,
            "stop_reason":"eos","generated_token_count":4}
patches={"load_manifest":lambda *_:(manifest,Path(".")),
         "load_core":lambda *_args,**_kwargs:(base,processor),
         "load_asr":lambda *_:(asr,asrprocessor),
         "transcribe":transcript,"asset_path":lambda *_:Path("intercepted"),
         "provenance":lambda *_:{},"parameter_counts":lambda *_:{},
         "write_json":lambda *_:None,"evaluate_rows":lambda *_:[],
         "generate":generated,"summarize":lambda *_:{}}
with patch.multiple(core,**patches):
    result=core.run_evaluate(options,baseline=True)
assert result["status"]=="completed"
assert calls==[{"same_core":True,"same_processor":True,"task":"speech_chat","user":"observed-hypothesis-only"},
               {"same_core":True,"same_processor":True,"task":"typed_chat","user":"reference-only-text"}]
print(json.dumps({"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},
                  "actual_functions_executed":["load_core(base)","load_core(train=True)",
                      "load_core(adapter=explicit-path)","load_asr","run_evaluate(baseline=True)"],
                  "provider_trace":trace,"generation_route_trace":calls,
                  "result_status":result["status"],
                  "scope":"Actual current project loader/routing control flow; external provider/model loading, transcribe/generate/IO intercepted. No installed Transformers/PEFT, model/ASR inference, audio decode or quality claim."},indent=2))
