from pathlib import Path
import os,sys,hashlib,json,torch
root=Path.cwd();base=root/'docs/technical-reviews/artifacts/phase4-19_6-independent'
sys.path.insert(0,str(root))
torch.set_num_threads(1);torch.manual_seed(42);torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
fence=base/'inputs/fence-1.py';raw=fence.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f62a3848030d0a23d92d0029bc9ffeca223f978e9a32d3d8b2f42e735c595c8d'
exec(compile(raw,str(fence),'exec'),{'__name__':'__main__'})
print('ENVIRONMENT',json.dumps({'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'seed':42,'threads':torch.get_num_threads(),'no_training':True},ensure_ascii=False))
