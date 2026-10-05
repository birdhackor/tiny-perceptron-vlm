"""T.1: opening a file is distinct from meeting an input contract; bounded CPU only."""
import hashlib,json,platform,sys,tempfile
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import soundfile as sf
import torch
from tiny_perceptron.modal_data import modal_example

HERE=Path(__file__).resolve().parent
assert torch.version.cuda is None
print(json.dumps({'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu','Pillow':PIL.__version__,'soundfile':sf.__version__,'numpy':np.__version__,'platform':platform.platform()},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
with tempfile.TemporaryDirectory(dir=HERE) as temp:
    p=Path(temp)
    Image.new('RGB',(7,9),'red').save(p/'tiny.png')
    with Image.open(p/'tiny.png') as im: assert im.size==(7,9)
    _,labels,image,_=modal_example({'image':'tiny.png','question':'color?','answer':'red'},p,'vision')
    assert list(image.shape)==[3,16,16] and (labels!=-100).sum().item()==4
    sf.write(p/'tiny.wav',np.zeros(32,dtype=np.float32),8000)
    wave,rate=sf.read(p/'tiny.wav');assert rate==8000 and len(wave)==32
    try:modal_example({'audio':'tiny.wav','question':'pitch?','answer':'low'},p,'audio')
    except ValueError as e: rejected=str(e)
    else:raise AssertionError('file readable, but wrong sampling rate must fail')
    sf.write(p/'tiny-16k.wav',np.zeros(32,dtype=np.float32),16000)
    _,answer,_,waveform=modal_example({'audio':'tiny-16k.wav','question':'pitch?','answer':'low'},p,'audio')
    assert list(waveform.shape)==[32] and (answer!=-100).sum().item()==4
    print(json.dumps({'original_image_dimensions':[7,9],'model_image_shape':list(image.shape),'audio_read_success':True,'read_sample_rate':rate,'wrong_rate_error':rejected,'accepted_rate':16000,'accepted_waveform_shape':list(waveform.shape),'answer_target_positions_including_eos':4,'scope':'Input-contract demonstration only; no model or real-world label evaluation.'},ensure_ascii=False))
