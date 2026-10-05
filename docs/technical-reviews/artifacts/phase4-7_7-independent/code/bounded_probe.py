"""Independent CPU probe for 7.7; no optimizer, datasets, weights, or network."""
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer, CharTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss, loss_sum

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=10, width=8))

def digest_parameters():
    return hashlib.sha256(b"".join(p.detach().contiguous().numpy().tobytes() for p in model.parameters())).hexdigest()

before = digest_parameters()
seq = torch.tensor([[1, 2, 3]])
ids = torch.tensor([[0, 0, 1, 2, 3]])
valid = ids != 0
positions = torch.tensor([[0, 0, 0, 1, 2]])
base = model(seq)["logits"]
padded = model(ids, valid=valid, positions=positions)["logits"]
no_valid = model(ids, positions=positions)["logits"][:, 2:]
no_positions = model(ids, valid=valid)["logits"][:, 2:]
equivalence = {
    "base_shape": list(base.shape), "padded_shape": list(padded.shape),
    "actual_shape": list(padded[:, 2:].shape), "dtype": str(base.dtype),
    "valid": valid.tolist(), "positions": positions.tolist(),
    "correct_max_abs": (base - padded[:, 2:]).abs().max().item(),
    "without_valid_max_abs": (base - no_valid).abs().max().item(),
    "without_positions_max_abs": (base - no_positions).abs().max().item(),
    "without_valid_allclose": torch.allclose(base, no_valid, atol=1e-6),
    "without_positions_allclose": torch.allclose(base, no_positions, atol=1e-6),
    "pad_logits_max_abs": padded[:, :2].abs().max().item(),
    "pad_embedding_norm": model.embedding.weight[0].norm().item(),
    "model_embedding_padding_idx": str(model.embedding.padding_idx),
}
assert equivalence["correct_max_abs"] <= 1e-6
assert not equivalence["without_valid_allclose"] and not equivalence["without_positions_allclose"]

# Empty queries are rows, padded keys are columns, and softmax is over keys.
allowed = attention_mask(torch.arange(5), torch.arange(5), valid)
q = torch.zeros(1, 1, 5, 2)
k = torch.zeros(1, 1, 5, 2)
v = torch.tensor([[[[100.0, 100.0], [200.0, 200.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]])
attended, weights = manual_attention(q, k, v, allowed)
raw_softmax = torch.full((1, 1, 5, 5), float("-inf")).softmax(-1)
axes = {
    "axes": ["batch", "head", "query", "key"],
    "allowed_shape": list(allowed.shape), "allowed": allowed[0, 0].tolist(),
    "weights": weights[0, 0].tolist(), "weights_key_row_sums": weights.sum(-1).tolist(),
    "attention_output": attended[0, 0].tolist(),
    "allowed_key_counts": allowed.sum(-1).tolist(),
    "raw_all_negative_infinity_softmax_all_nan": bool(torch.isnan(raw_softmax).all()),
    "safe_output_finite": bool(torch.isfinite(attended).all()),
}
assert attended[0, 0, :2].eq(0).all()
assert weights[0, 0, :2].eq(0).all()
assert weights[0, 0, :, :2].eq(0).all()
assert torch.allclose(attended[0, 0, 2:], torch.tensor([[1., 1.], [1.5, 1.5], [2., 2.]]), rtol=0, atol=1e-7)
expected_allowed = [[False]*5, [False]*5, [False,False,True,False,False], [False,False,True,True,False], [False,False,True,True,True]]
assert allowed[0, 0].tolist() == expected_allowed

# Unequal scores check sqrt(head dimension), denominator and mask exclusion.
q2 = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
k2 = torch.tensor([[[[0., 0.], [math.sqrt(2), 0.], [100., 0.]]]], dtype=torch.float64)
v2 = torch.tensor([[[[10.], [20.], [999.]]]], dtype=torch.float64)
allowed2 = torch.tensor([[[[True, True, False]]]])
out2, w2 = manual_attention(q2, k2, v2, allowed2)
denom = 1 + math.exp(1)
expected_w = torch.tensor([1/denom, math.exp(1)/denom, 0.], dtype=torch.float64)
assert torch.allclose(w2[0,0,0], expected_w, rtol=0, atol=1e-12)
scaled = {"head_dim": 2, "divisor": math.sqrt(2), "unmasked_scores": [0., 1.], "denominator": denom,
          "weights": w2.tolist(), "output": out2.item(), "expected_output": 10/denom+20*math.exp(1)/denom,
          "probability_units": "dimensionless", "output_units": "same as v", "tolerance": "absolute 1e-12 float64"}

# valid is a key mask, not a query mask. A right PAD query reads earlier real keys.
rv = torch.tensor([[True,True,True,False,False]])
ra = attention_mask(torch.arange(5), torch.arange(5), rv)
axes["right_padding_invalid_query_has_allowed_key"] = bool(ra[0,0,3].any())
axes["right_padding_allowed"] = ra[0,0].tolist()

# PAD position IDs still index the table; they are irrelevant only to retained real output.
changed = positions.clone(); changed[0, :2] = torch.tensor([8, 9])
changed_logits = model(ids, valid=valid, positions=changed)["logits"]
equivalence["changed_pad_position_real_max_abs"] = (base-changed_logits[:,2:]).abs().max().item()
assert equivalence["changed_pad_position_real_max_abs"] <= 1e-6

# Short unequal-length batches, both padding sides; use one fixed model.
batch_checks = []
sequences = [[1,2,3], [4,5]]
for side in ("left", "right"):
    rows, validity, position_rows = [], [], []
    for tokens in sequences:
        n = len(tokens); p = 5-n
        rows.append([0]*p+tokens if side=="left" else tokens+[0]*p)
        validity.append([False]*p+[True]*n if side=="left" else [True]*n+[False]*p)
        position_rows.append([0]*p+list(range(n)) if side=="left" else list(range(n))+[0]*p)
    bt, bv, bp = torch.tensor(rows), torch.tensor(validity), torch.tensor(position_rows)
    got = model(bt, valid=bv, positions=bp)["logits"]
    for row, tokens in enumerate(sequences):
        ref = model(torch.tensor([tokens]))["logits"][0]
        real = got[row,bv[row]]
        d=(ref-real).abs().max().item()
        assert d <= 1e-6
        batch_checks.append({"side":side,"row":row,"real_tokens":len(tokens),"physical_length":5,"max_abs":d})

right_ids = torch.tensor([[1,2,3,0,0]])
right_ref = model(right_ids)["logits"][:,:3]
equivalence["right_padding_without_valid_or_custom_positions_max_abs"] = (base-right_ref).abs().max().item()
assert equivalence["right_padding_without_valid_or_custom_positions_max_abs"] <= 1e-6

# A -100 target controls loss, not token embedding or attention visibility.
labels = torch.tensor([[2,3,4]])
plabels = torch.tensor([[-100,-100,2,3,4]])
lbase = masked_loss(base, labels)
lpadded = masked_loss(padded, plabels)
lunmasked = masked_loss(model(ids, positions=positions)["logits"], plabels)
count = loss_sum(padded, plabels)[1]
loss_check = {"valid_target_denominator":count.item(), "base_mean":lbase.item(), "padded_mean":lpadded.item(),
    "padded_ignore_labels_without_valid_mean":lunmasked.item(),"labels":plabels.tolist()}
assert count.item() == 3 and torch.allclose(lbase,lpadded,rtol=0,atol=1e-6)
px,py,pv = pad_batch([(torch.tensor([1,2,3]), torch.tensor([2,3,4])), (torch.tensor([4,5]), torch.tensor([5,6]))])
batch_contract={"ids":px.tolist(),"labels":py.tolist(),"valid":pv.tolist(),
    "byte_tokenizer_pad_id":ByteTokenizer.pad_id,
    "char_tokenizer_unknown_is_zero":CharTokenizer("A").encode("Z"),
    "pad_batch_source": "tiny_perceptron/data.py:71-86; validity determined by length, not token value"}

# allclose includes relative tolerance unless explicitly disabled.
ta=torch.tensor([1.],dtype=torch.float64)
tb=torch.tensor([1.000002],dtype=torch.float64)
tolerance={"input":ta.tolist(),"other":tb.tolist(),"difference":(ta-tb).abs().item(),
    "default_rtol":1e-5,"atol_argument":1e-6,"mixed_allowed_difference":1e-6+1e-5*tb.item(),
    "original_allclose_result":torch.allclose(ta,tb,atol=1e-6),
    "absolute_only_allclose_result":torch.allclose(ta,tb,atol=1e-6,rtol=0.0)}
assert tolerance["difference"] > 1e-6 and tolerance["original_allclose_result"] and not tolerance["absolute_only_allclose_result"]
assert torch.allclose(base,padded[:,2:],atol=1e-6,rtol=0.0)
after = digest_parameters()
assert before == after and all(p.grad is None for p in model.parameters())
result={"environment":{"python":sys.version,"torch":str(torch.__version__),"torch_git_version":str(torch.version.git_version),
    "cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),"device":"cpu", "threads":torch.get_num_threads()},
    "equivalence":equivalence,"mask_axis_and_empty_queries":axes,"scaled_softmax_calculation":scaled,
    "unequal_length_batch":batch_checks,"loss_attention_separation":loss_check,"batch_helper_contract":batch_contract,
    "allclose_tolerance_counterexample":tolerance,"model_parameters_before_sha256":before,"model_parameters_after_sha256":after,
    "parameters_updated":False,"backward_called":False,"optimizer_created":False,"existing_model_loaded":False}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
