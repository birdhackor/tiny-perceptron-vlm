import struct,json,math
from pathlib import Path
for name,count in [("MoE",5447107),("Dense",2288067)]:
 b=count*2;print(name,"ideal2B",b,"MiB",round(b/1024**2,2));assert b*2==count*4
p=Path("outputs/p7-technical-e-cache/moe-joint/model.safetensors")
with p.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(n))
t={k:v for k,v in header.items() if k!="__metadata__"}
assert {v["dtype"] for v in t.values()}=={"F32"}
payload=sum(math.prod(v["shape"])*4 for v in t.values())
assert p.stat().st_size==payload+n+8
print(json.dumps({"actual_moe_file_bytes":p.stat().st_size,"tensor_payload":payload,"header_plus_length":n+8,"all_dtypes":"F32","unique_param_bytes":5447107*4,"extra_saved_tensor_bytes":payload-5447107*4,"tied_table_bytes":550*256*4},ensure_ascii=False))
assert payload-5447107*4==550*256*4
print("file_layout_not_peak_RAM_or_converted_2B_model")
