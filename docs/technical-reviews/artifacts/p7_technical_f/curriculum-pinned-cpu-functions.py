"""Necessary original source function excerpts; synthetic CPU inputs only."""

import json,math,torch

from torch.nn import functional as F

from types import SimpleNamespace

# Original outputs/p7_technical_f/source-cache/minimind-v-sources/dataset/lm_dataset.py; lines75–91; SHA256:3b7816ffb72d7d6232a54bb776ff2ed73a8fc1fcdcd67501480f7e00be5e3d24

def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels

# Original outputs/p7_technical_f/source-cache/minimind-sources/trainer/train_dpo.py; lines34–50; SHA256:a25c707baf672fff28d8a2776a0c3b621fe4a4bff1ad26eea340f204546d9286

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

# Original outputs/p7_technical_f/source-cache/minimind-sources/trainer/train_distillation.py; lines25–36; SHA256:3f41c9c5caf90e90e04bc104b196cef5bceda981510e78b1beecc33b216b23a7

def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl

ids=[2,3,11,12,8,9,13,14,5]
labels=generate_labels(SimpleNamespace(bos_id=[11,12],eos_id=[13,14],max_length=20),ids)
r=torch.zeros(2,2);mask=torch.ones(2,2)
base=dpo_loss(r,r,mask,.1).item();policy=r.clone();policy[0]=.2
preferred=dpo_loss(r,policy,mask,.1).item()
s=torch.tensor([[1.,0.]],requires_grad=True);t=torch.tensor([[1.,0.]],requires_grad=True)
k=distillation_loss(s,t,temperature=2.);k.backward()
print(json.dumps({'input_ids':ids,'labels':labels,'dpo_equal':base,'dpo_preferred':preferred,'expected_ln2':math.log(2),'distill_identical':k.item(),'teacher_grad':None if t.grad is None else t.grad.tolist(),'student_grad':s.grad.tolist()}))
