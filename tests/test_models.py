"""檢查能揭露資料洩漏、mask錯位、梯度與數值方向的核心性質。"""

import copy

import pytest
import torch
from torch import nn

from tiny_perceptron.alignment import LoRALinear, distillation_kl, dpo_loss
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.modern import MoEFFN
from tiny_perceptron.multimodal import expand_modalities, log_mel, patchify, tone, unpatchify
from tiny_perceptron.quantization import QuantizedLinear, pack_int4, unpack_int4
from tiny_perceptron.training import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_causal_future_perturbation():
    model = TinyLM(ModelConfig(vocab_size=20, width=16, heads=2, layers=2))
    x = torch.tensor([[1, 2, 3, 4]])
    y = x.clone()
    y[:, 3] = 9
    assert torch.allclose(model(x)["logits"][:, :3], model(y)["logits"][:, :3], atol=1e-6)


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cache_full_prefix_equivalence(rotary, kv_heads):
    model = TinyLM(ModelConfig(vocab_size=20, width=16, heads=2, layers=2, kv_heads=kv_heads, rotary=rotary))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    full = model(ids)["logits"]
    prefix = model(ids[:, :3])
    pieces, cache = [prefix["logits"]], prefix["cache"]
    for position in (3, 4):
        result = model(ids[:, position : position + 1], cache=cache)
        pieces.append(result["logits"])
        cache = result["cache"]
    assert torch.allclose(torch.cat(pieces, 1), full, atol=2e-6)
    assert cache[0][0].shape == (1, kv_heads, 5, 8)


def test_sdpa_forward_and_gradient_equivalence():
    manual = TinyLM(ModelConfig(vocab_size=20, width=16, heads=2))
    sdpa = copy.deepcopy(manual)
    for block in sdpa.blocks:
        block.attention.backend = "sdpa"
    ids = torch.tensor([[1, 2, 3], [0, 1, 2]])
    valid = torch.tensor([[True, True, True], [False, True, True]])
    a, b = manual(ids, valid=valid)["logits"], sdpa(ids, valid=valid)["logits"]
    assert torch.allclose(a, b, atol=2e-6)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(manual.parameters(), sdpa.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-4, rtol=1e-4)


def test_chat_first_answer_and_untrusted_marker():
    tok = ByteTokenizer()
    x, y = render_chat([{"role": "user", "content": "<assistant>"}, {"role": "assistant", "content": "答"}])
    first = (y != IGNORE).nonzero()[0].item()
    assert x[first] == tok.assistant_id
    assert y[first] == tok.encode("答")[0]
    assert (x == tok.assistant_id).sum() == 1
    assert y[-1] == tok.eos_id


def test_padding_loss_and_left_padding_invariance():
    model = TinyLM(ModelConfig(vocab_size=20, width=16))
    x = torch.tensor([[1, 2, 3]])
    baseline = model(x)["logits"]
    padded = torch.tensor([[0, 0, 1, 2, 3]])
    valid = torch.tensor([[False, False, True, True, True]])
    positions = torch.tensor([[0, 0, 0, 1, 2]])
    actual = model(padded, valid=valid, positions=positions)["logits"][:, 2:]
    assert torch.allclose(actual, baseline, atol=2e-6)
    labels = torch.tensor([[2, 3, 4]])
    extra_logits = torch.cat((baseline, torch.randn(1, 4, 20)), 1)
    extra_labels = torch.cat((labels, torch.full((1, 4), IGNORE)), 1)
    assert torch.allclose(masked_loss(baseline, labels), masked_loss(extra_logits, extra_labels))
    with pytest.raises(ValueError, match="所有 labels"):
        masked_loss(baseline, torch.full_like(labels, IGNORE))


def test_truncation_rejects_empty_supervision():
    example = render_chat([{"role": "user", "content": "很長的問題"}, {"role": "assistant", "content": "是"}])
    with pytest.raises(ValueError, match="截斷"):
        pad_batch([example], max_length=2)


def test_checkpoint_next_update_matches(tmp_path):
    model = TinyLM(ModelConfig(vocab_size=20, width=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids, labels = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    masked_loss(model(ids)["logits"], labels).backward()
    optimizer.step()
    optimizer.zero_grad()
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, step=1)
    other, payload = load_checkpoint(path, restore_rng=True)
    resumed = torch.optim.AdamW(other.parameters(), lr=0.001)
    resumed.load_state_dict(payload["optimizer"])
    for current, opt in ((model, optimizer), (other, resumed)):
        masked_loss(current(ids)["logits"], labels).backward()
        opt.step()
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(p, q)


def test_top1_router_receives_task_gradient():
    layer = MoEFFN(8, experts=3, top_k=1)
    output, auxiliary, selected = layer(torch.randn(2, 4, 8))
    output.square().sum().backward()
    assert layer.router.weight.grad.norm() > 0
    assert selected.shape == (8, 1) and auxiliary.isfinite()


def test_lora_zero_initialization_and_merge():
    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4)
    x = torch.randn(2, 4)
    assert torch.equal(layer(x), layer.base(x))
    with torch.no_grad():
        layer.b.fill_(0.1)
    expected = x @ layer.merged_weight().T + layer.base.bias
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_int4_signed_odd_length_roundtrip():
    q = torch.tensor([[-8, -7, -1, 0, 1, 6, 7]], dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.numel() == 4
    assert torch.equal(unpack_int4(packed, q.shape), q)
    layer = nn.Linear(64, 16)
    quantized = QuantizedLinear(layer, bits=4)
    assert quantized.storage_bytes() < sum(p.numel() * p.element_size() for p in layer.parameters())
    assert quantized(torch.randn(2, 64)).shape == (2, 16)


def test_dpo_gradient_direction_and_teacher_detach():
    chosen, rejected = torch.tensor([-2.0], requires_grad=True), torch.tensor([-3.0], requires_grad=True)
    ref = torch.tensor([-2.0], requires_grad=True)
    dpo_loss(chosen, rejected, ref, ref).backward()
    assert chosen.grad < 0 and rejected.grad > 0 and ref.grad is None
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[1, IGNORE, 3]])
    loss = distillation_kl(student, teacher, labels)
    loss.backward()
    assert teacher.grad is None and student.grad[:, 1].abs().sum() == 0
    assert distillation_kl(student.detach(), student.detach(), labels).abs() < 1e-5


def test_patch_and_modal_label_alignment():
    image = torch.rand(2, 3, 8, 12)
    assert torch.equal(unpatchify(patchify(image), 3, 8, 12), image)
    tok, embedding = ByteTokenizer(), nn.Embedding(264, 8)
    ids = torch.tensor([tok.bos_id, tok.image_id, tok.assistant_id, 10, tok.eos_id])
    labels = torch.tensor([IGNORE, IGNORE, IGNORE, 10, tok.eos_id])
    features = {tok.image_id: torch.randn(3, 8, requires_grad=True)}
    x, y = expand_modalities(ids, labels, embedding, features, {tok.image_id})
    assert x.shape == (1, 6, 8)
    assert torch.equal(y, torch.tensor([[IGNORE, IGNORE, IGNORE, IGNORE, 10, tok.eos_id]]))


def test_log_mel_short_waveform_and_frequency():
    for wave in (torch.zeros(30), tone(440.0)):
        features = log_mel(wave)
        assert features.shape[0] == 16 and features.isfinite().all()
