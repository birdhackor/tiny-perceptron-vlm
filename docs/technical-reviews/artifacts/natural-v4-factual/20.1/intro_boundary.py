import json,platform,socket
import torch
from tiny_perceptron.capstone import CapstoneModel,default_config
from tiny_perceptron.natural_assistant import messages_for
from pathlib import Path
original=socket.socket.connect
attempts=[]
def denied(self,address):attempts.append(str(address));raise RuntimeError('No downloads permitted')
socket.socket.connect=denied
try:
 torch.manual_seed(42)
 model=CapstoneModel(default_config())
 assert model.config.experts==4
 description=model.description()
 assert len(model.language.blocks)==2 and all(len(block.ffn.experts)==4 for block in model.language.blocks)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','config_experts':model.config.experts,'randomly_initialized_small_model':True,'image_projector_present':hasattr(model,'image_projector'),'audio_projector_present':hasattr(model,'audio_projector'),'parameters':description['parameters'],'network_attempts':attempts,'scope':'Bounded constructor inspection only; no capstone training or heldout quality replication'},indent=2))
finally:socket.socket.connect=original
