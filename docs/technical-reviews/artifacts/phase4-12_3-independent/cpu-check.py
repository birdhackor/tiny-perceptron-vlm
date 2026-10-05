import torch, sys, json, math
from tiny_perceptron.multimodal import tone
torch.set_num_threads(1)
w = tone()
N, L, H = w.numel(), 400, 160
f = w.unfold(0,L,H)
starts = list(range(0,N-L+1,H))
assert tuple(f.shape) == (8,400)
assert torch.equal(f, torch.stack([w[s:s+L] for s in starts]))
assert torch.equal(f[0,H:], f[1,:L-H])
nonoverlap=w.unfold(0,L,L)
assert tuple(nonoverlap.shape)==(4,400)
assert torch.equal(nonoverlap.flatten(),w)
last = starts[-1] + L - 1
assert last == 1519 and N-1-last == 80
assert len(starts)==1+(N-L)//H
assert N-(starts[-1]+H)==320 and N-(starts[-1]+H)<L
window=torch.hann_window(L)
centered=torch.stft(w,n_fft=L,hop_length=H,window=window,center=True,return_complex=True)
uncentered=torch.stft(w,n_fft=L,hop_length=H,window=window,center=False,return_complex=True)
assert tuple(centered.shape)==(201,11)
assert tuple(uncentered.shape)==(201,8)
print(json.dumps({'environment':{'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':str(w.device),'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'threads':str(torch.get_num_threads())},'wave':{'points':N,'sample_rate_Hz':16000,'duration_seconds':N/16000},'frames':{'shape':list(f.shape),'starts':starts,'last_inclusive':last,'uncovered_tail':N-1-last,'next_start':starts[-1]+H,'points_after_next_start':N-(starts[-1]+H),'frame_seconds':L/16000,'hop_seconds':H/16000,'overlap_points':L-H,'exact_equal_to_source_slices':True,'overlap_equal':True},'step400':{'shape':list(nonoverlap.shape),'starts':list(range(0,N-L+1,L)),'overlap':0,'reconstructs_exact_wave':True},'stft_boundary_variant':{'n_fft':L,'hop_length':H,'centered_shape':list(centered.shape),'uncentered_shape':list(uncentered.shape),'boundary_padding':'reflect, n_fft//2 each side when center=True'}},indent=2))
