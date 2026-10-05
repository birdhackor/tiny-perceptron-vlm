from pathlib import Path
import sys,json,math,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from tiny_perceptron.multimodal import tone,mel_filter_bank,log_mel

torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None
out=Path(__file__).resolve().parent
facts={'environment':{'python':sys.version,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda)},'scope':'Bounded deterministic feature arithmetic; no data/model access, training, optimizer, or checkpoints.'}
power=torch.tensor([0.0,1e-10,1e-4,1.0],dtype=torch.float64)
for floor in [1e-8,1e-12]:
 compressed=power.clamp(min=floor).log()
 scaled=(4*power).clamp(min=floor).log()
 expected=torch.tensor([math.log(max(float(x),floor)) for x in power],dtype=torch.float64)
 assert torch.allclose(compressed,expected,rtol=0,atol=1e-12)
 assert compressed.isfinite().all()
 facts[str(floor)]={'power':power.tolist(),'log_power':compressed.tolist(),'rounded_log':[round(x,2) for x in compressed.tolist()],'scaled_difference':(scaled-compressed).tolist(),'all_finite':bool(compressed.isfinite().all())}
assert facts['1e-08']['rounded_log']==[-18.42,-18.42,-9.21,0.0]
assert facts['1e-12']['rounded_log']==[-27.63,-23.03,-9.21,0.0]
assert abs(facts['1e-08']['scaled_difference'][2]-math.log(4))<1e-12
assert facts['1e-08']['scaled_difference'][:2]==[0.0,0.0]
unclamped=power.log()
facts['no_floor']={'first_value_is_negative_infinity':bool(torch.isneginf(unclamped[0])),'finite_flags':unclamped.isfinite().tolist()}
assert facts['no_floor']['finite_flags']==[False,True,True,True]
facts['same_input_different_floor']={'input_power':1e-10,'floor_1e8':math.log(max(1e-10,1e-8)),'floor_1e12':math.log(max(1e-10,1e-12)),'difference':math.log(1e-8)-math.log(1e-10)}
crossing=torch.tensor([0,2e-9,5e-9,1e-8,1e-4,1.0],dtype=torch.float64)
diff=(4*crossing).clamp(min=1e-8).log()-crossing.clamp(min=1e-8).log()
facts['floor_crossing']={'power':crossing.tolist(),'log_delta':diff.tolist(),'scope':'ln(4) identity only when both original and scaled powers are above the floor; crossing it gives a smaller increase.'}
assert abs(float(diff[2])-math.log(2))<1e-12
facts['display_arithmetic']={'linear_8bit_brightness':(power*255).round().tolist(),'log_display_normalized':((power.clamp(min=1e-8).log()-math.log(1e-8))/(-math.log(1e-8))).tolist(),'scope':'Illustrative linear 8-bit display and separately normalized log display; not an experiment or actual figure.'}
with torch.no_grad():
 wave=torch.cat([tone(440,seconds=0.05),tone(880,seconds=0.05)]).to(torch.float64)
 window=torch.hann_window(400,dtype=wave.dtype)
 spectrum=torch.stft(wave,400,160,window=window,return_complex=True)
 double_spectrum=torch.stft(2*wave,400,160,window=window,return_complex=True)
 p1=spectrum.abs().square();p2=double_spectrum.abs().square()
 power_scaling_error=float((p2-4*p1).abs().max())
 assert torch.allclose(p2,4*p1,atol=1e-10,rtol=1e-12)
 bank=mel_filter_bank().to(wave.dtype)
 mel=bank@p1
 actual=log_mel(wave)
 explicit=mel.clamp(min=1e-8).log()
 assert actual.shape==(16,11)
 assert torch.equal(actual,explicit)
 assert torch.equal(actual,log_mel(wave))
 silent=log_mel(torch.zeros(1600,dtype=torch.float64))
 assert torch.equal(silent,torch.full((16,11),math.log(1e-8),dtype=torch.float64))
 features2=log_mel(2*wave);above=mel>=1e-8
 ln4_error=float(((features2-actual)[above]-math.log(4)).abs().max())
 assert ln4_error<1e-10
 temporal_change=float((actual[:,2]-actual[:,8]).abs().max())
 assert temporal_change>1
 facts['audio_contract']={'wave_samples':wave.numel(),'sample_rate_hz':16000,'n_fft_samples':400,'hop_samples':160,'spectral_axes':['frequency','time'],'spectral_shape':list(p1.shape),'filter_shape':list(bank.shape),'mel_axes':['mel_band','time'],'mel_shape':list(mel.shape),'log_mel_shape':list(actual.shape),'feature_formula_exact_match':True,'repeated_formula_exact_match':True,'amplitude_factor':2,'power_factor':4,'max_power_scaling_error':power_scaling_error,'above_floor_elements':int(above.sum()),'ln4_max_error':ln4_error,'silent_log_values':silent.unique().tolist(),'max_difference_of_two_time_frames':temporal_change,'output_requires_grad':actual.requires_grad}
for path in ['tiny_perceptron/multimodal.py','docs/review-tools/section_facts.py']:
 facts.setdefault('repository_sha256',{})[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
(out/'variant-results.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False))
print('All bounded assertions passed.')
