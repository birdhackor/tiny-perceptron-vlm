"""Bounded CPU checks of the two-candidate demonstration and its stated exercise."""
import json
import math
import torch
import torch.nn.functional as F

torch.set_default_device('cpu')
torch.set_num_threads(1)
q = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
records = []
for magnitude in (100.0, 0.1):
    k = torch.tensor([[1.0, 0.0], [0.0, magnitude]], dtype=torch.float64)
    raw = q @ k.T
    unit_q = F.normalize(q, dim=-1)
    unit_k = F.normalize(k, dim=-1)
    unit = unit_q @ unit_k.T
    expected_raw = [1.0, magnitude]
    shift = max(expected_raw)
    exps = [math.exp(x - shift) for x in expected_raw]
    hand_weights = [x / sum(exps) for x in exps]
    angles = [math.degrees(math.acos(float(unit_q[0] @ row))) for row in unit_k]
    assert raw.shape == unit.shape == (1, 2)
    assert k.T.shape == (2, 2)
    torch.testing.assert_close(raw, torch.tensor([expected_raw], dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(raw.softmax(-1), torch.tensor([hand_weights], dtype=torch.float64), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(unit, torch.tensor([[1/math.sqrt(2), 1/math.sqrt(2)]], dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(unit.softmax(-1), torch.tensor([[0.5,0.5]], dtype=torch.float64), rtol=0, atol=1e-14)
    assert all(abs(angle - 45) < 1e-12 for angle in angles)
    records.append({'k_vertical_magnitude': magnitude, 'q_shape': list(q.shape), 'k_shape': list(k.shape), 'raw_shape': list(raw.shape), 'unit_shape': list(unit.shape), 'angles_degrees': angles, 'raw_scores': raw.tolist(), 'raw_weights': raw.softmax(-1).tolist(), 'unit_q': unit_q.tolist(), 'unit_k': unit_k.tolist(), 'normalized_scores': unit.tolist(), 'normalized_weights': unit.softmax(-1).tolist(), 'independent_math_softmax': hand_weights})

# A nontrivial axis perturbation: two queries and two non-orthogonal candidates.
axis_q = torch.tensor([[1.,1.],[2.,1.]], dtype=torch.float64)
axis_k = torch.tensor([[1.,1.],[1.,100.]], dtype=torch.float64)
per_vector = F.normalize(axis_q,dim=-1) @ F.normalize(axis_k,dim=-1).T
per_candidate_axis = F.normalize(axis_q,dim=0) @ F.normalize(axis_k,dim=0).T
assert not torch.allclose(per_vector,per_candidate_axis)
assert not torch.allclose(per_vector.softmax(-1),per_vector.softmax(0))

scores = torch.tensor([1.,0.],dtype=torch.float64)
scale_records=[]
for scale in (0.5,1.,4.):
    tau = 1/scale
    observed=(scores*scale).softmax(-1)
    expected=[1/(1+math.exp(-scale)),1/(1+math.exp(scale))]
    torch.testing.assert_close(observed,torch.tensor(expected,dtype=torch.float64),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(observed,(scores/tau).softmax(-1),rtol=0,atol=0)
    scale_records.append({'scale':scale,'positive_temperature':tau,'weights':observed.tolist()})
assert scale_records[0]['weights'][0] < scale_records[1]['weights'][0] < scale_records[2]['weights'][0]
print(json.dumps({'tolerance':'float64 atol=1e-14 (angles 1e-12 degrees); original displayed 4 decimals: 5e-5','candidates':records,'axis_perturbation':{'q':axis_q.tolist(),'k':axis_k.tolist(),'per_vector_scores':per_vector.tolist(),'candidate_axis_normalized_scores':per_candidate_axis.tolist(),'candidate_softmax':per_vector.softmax(-1).tolist(),'query_softmax':per_vector.softmax(0).tolist()},'scale_temperature':scale_records,'checks':'all assertions passed; forward calculation only; no gradients, parameter updates or model evaluation'},indent=2))
