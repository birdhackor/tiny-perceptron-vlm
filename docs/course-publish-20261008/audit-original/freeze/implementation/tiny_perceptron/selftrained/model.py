"""Finite multimodal assistant; every neural module starts at random initialization.

Reuse TinyLM, Block, RMSNorm, RoPE, GQA and SDPA without modifying the course model.
Image slots retain public geometry, OCR retains columns, and audio retains time.
Perception heads are observable intermediate predictions, never an answer lookup.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import Block, ModelConfig, TinyLM, loss_sum
from tiny_perceptron.modern import DenseFFN, MoEFFN, RMSNorm
from tiny_perceptron.selftrained.tokenizer import OCR_CHARACTERS

MODEL_VERSION = "selftrained_multimodal_v1"
MODALITY_TOKENS = {"image": 2, "ocr": 32, "audio": 16}
MODALITY_IDS = {"image": 7, "ocr": 8, "audio": 9}


@dataclass
class SelftrainedConfig:
    vocab_size: int = 256
    architecture: str = "moe"
    width: int = 128
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    ffn_hidden: int = 256
    experts: int = 4
    top_k: int = 1
    max_length: int = 512
    backend: str = "sdpa"
    tied: bool = True
    auxiliary_weight: float = 0.01
    version: str = MODEL_VERSION

    def __post_init__(self):
        if self.version != MODEL_VERSION or self.architecture not in ("moe", "dense"):
            raise ValueError("Unknown selftrained model version or architecture")
        if min(self.width, self.layers, self.ffn_hidden, self.max_length, self.heads, self.kv_heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.vocab_size < 11 or self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("Invalid vocabulary, attention or GQA dimensions")
        if (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.architecture == "moe" and not 1 <= self.top_k <= self.experts:
            raise ValueError("top_k must be between one and the number of experts")
        if self.auxiliary_weight < 0:
            raise ValueError("Auxiliary weight cannot be negative")


class PaddingSafeMoE(MoEFFN):
    """Dropless routing on valid rows only; padding contributes no load or importance."""

    def forward(self, x, valid=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        mask = torch.ones(flat.shape[0], dtype=torch.bool, device=x.device) if valid is None else valid.reshape(-1)
        active_rows = mask.nonzero(as_tuple=False).flatten()
        output = torch.zeros_like(flat)
        chosen_all = torch.full((flat.shape[0], self.top_k), -1, dtype=torch.long, device=x.device)
        count = len(self.experts)
        if not len(active_rows):
            return (
                output.view(shape),
                x.new_zeros(()),
                {
                    "counts": torch.zeros(count, dtype=torch.long, device=x.device),
                    "importance": x.new_zeros(count),
                    "valid_tokens": 0,
                    "chosen": chosen_all.view(*shape[:-1], self.top_k),
                },
            )
        active = flat[active_rows]
        probabilities = self.router(active).float().softmax(-1)
        weights, chosen = probabilities.topk(self.top_k, dim=-1)
        # Top-1 keeps its gate probability so task loss also trains the router.
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        active_output = torch.zeros_like(active)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(chosen == index)
            if len(rows):
                contribution = expert(active[rows]) * weights[rows, slots, None].to(active.dtype)
                active_output.index_add_(0, rows, contribution)
        output[active_rows] = active_output
        chosen_all[active_rows] = chosen
        counts = torch.bincount(chosen.flatten(), minlength=count)
        load = counts.float() / (len(active_rows) * self.top_k)
        importance = probabilities.mean(0)
        auxiliary = count * (load.detach() * importance).sum()
        return (
            output.view(shape),
            auxiliary,
            {
                "counts": counts.detach(),
                "importance": importance.detach(),
                "valid_tokens": len(active_rows),
                "chosen": chosen_all.view(*shape[:-1], self.top_k).detach(),
            },
        )


class MaskedBlock(Block):
    def __init__(self, language_config, config):
        # The reused block supplies the same attention, norms and residual structure.
        super().__init__(language_config)
        self.ffn = (
            PaddingSafeMoE(config.width, config.experts, config.top_k, config.ffn_hidden)
            if config.architecture == "moe"
            else DenseFFN(config.width, config.ffn_hidden)
        )

    def forward(self, x, valid=None, positions=None, segments=None, cache=None, routing_valid=None):
        attended, state = self.attention(self.norm1(x), valid, positions, segments, cache)
        x = x + attended
        h = self.norm2(x)
        auxiliary, routing = x.new_zeros(()), None
        if isinstance(self.ffn, PaddingSafeMoE):
            h, auxiliary, routing = self.ffn(h, routing_valid)
        else:
            h = self.ffn(h)
        x = x + h
        if routing_valid is not None:
            x = x.masked_fill(~routing_valid[..., None], 0)
        return x, state, auxiliary, routing


class SelftrainedLanguageModel(TinyLM):
    def __init__(self, config):
        language_config = ModelConfig(
            vocab_size=config.vocab_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            max_length=config.max_length,
            norm="rms",
            rotary=True,
            tied=config.tied,
            kv_heads=config.kv_heads,
            backend=config.backend,
            experts=0,
            top_k=config.top_k,
        )
        super().__init__(language_config)
        self.blocks = nn.ModuleList([MaskedBlock(language_config, config) for _ in range(config.layers)])
        self.config.experts = config.experts if config.architecture == "moe" else 0

    def forward(self, embeddings, attention_mask=None, positions=None, segments=None, cache=None):
        b, t, _ = embeddings.shape
        offset = 0 if cache is None else cache[0][0].shape[2]
        if offset + t > self.config.max_length:
            raise ValueError("Sequence exceeds max_length; no silent history truncation")
        valid = (
            torch.ones((b, t), dtype=torch.bool, device=embeddings.device) if attention_mask is None else attention_mask
        )
        if valid.shape != (b, t) or valid.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean [batch, time] tensor")
        if cache is not None and (attention_mask is not None or segments is not None):
            raise ValueError("Cache supports unpadded, unpacked sequences only")
        if positions is None:
            positions = (
                torch.arange(offset, offset + t, device=embeddings.device)
                if attention_mask is None
                else (valid.long().cumsum(-1) - 1).clamp_min(0)
            )
        x = embeddings.masked_fill(~valid[..., None], 0)
        states, routing, auxiliary = [], [], x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x, state, aux, route = block(
                x, attention_mask, positions, segments, None if cache is None else cache[i], valid
            )
            states.append(state)
            routing.append(route)
            auxiliary = auxiliary + aux
        return {
            "logits": self.output(self.final_norm(x)),
            "cache": states,
            "aux_loss": auxiliary / len(self.blocks),
            "routing": routing,
        }


class VisionEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.projection = nn.Linear(64 * 2 * 2, width)
        self.head = nn.Linear(width, 3)
        self.symbol_projection = nn.Linear(3, width, bias=False)
        self.coordinates = nn.Linear(2, width, bias=False)
        self.slot_position = nn.Embedding(2, width)
        self.norm = RMSNorm(width)

    def forward(self, images, coordinates):
        if images.shape != (2, 1, 32, 32) or coordinates.shape != (2, 2):
            raise ValueError("Image payload needs [2,1,32,32] cells and [2,2] public x,y centers")
        features = F.gelu(self.projection(self.cnn(images).flatten(1)))
        logits = self.head(features)
        tokens = features + self.symbol_projection(logits.softmax(-1))
        tokens = tokens + self.coordinates(coordinates) + self.slot_position.weight
        return self.norm(tokens), logits


class OCREncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
        )
        self.column_projection = nn.Linear(64 * 8, 64)
        self.sequence = nn.GRU(64, 48, batch_first=True, bidirectional=True)
        self.head = nn.Linear(96, len(OCR_CHARACTERS) + 1)
        self.projection = nn.Linear(96, width)
        self.symbol_projection = nn.Linear(len(OCR_CHARACTERS) + 1, width, bias=False)
        self.column_position = nn.Embedding(32, width)
        self.norm = RMSNorm(width)

    def forward(self, image):
        if image.shape != (1, 32, 128):
            raise ValueError("OCR payload must be an explicit ROI crop [1,32,128]")
        features = self.cnn(image[None]).permute(0, 3, 1, 2).flatten(2)
        sequence, _ = self.sequence(F.gelu(self.column_projection(features)))
        sequence = sequence[0]
        logits = self.head(sequence)
        tokens = self.projection(sequence) + self.symbol_projection(logits.softmax(-1)) + self.column_position.weight
        return self.norm(tokens), logits


class AudioEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(40, 48, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(48, 64, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.sequence = nn.GRU(64, 48, batch_first=True, bidirectional=True)
        self.head = nn.Linear(96, 3)
        self.projection = nn.Linear(96, width)
        self.symbol_projection = nn.Linear(3, width, bias=False)
        self.time_position = nn.Embedding(16, width)
        self.norm = RMSNorm(width)

    def forward(self, frames, valid=None):
        if frames.ndim != 2 or frames.shape[1] != 40 or frames.shape[0] < 1:
            raise ValueError("Audio payload must contain the complete [time,40] log-mel sequence")
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != frames.shape[:1]:
                raise ValueError("Audio valid mask must be boolean [time]")
            length = int(valid.sum())
            expected = torch.arange(len(valid), device=valid.device) < length
            if length == 0 or not torch.equal(valid, expected):
                raise ValueError("Audio padding must follow a nonempty, contiguous valid prefix")
            # Trim only declared padding, never truncate real sound for a duration limit.
            frames = frames[:length]
        features = self.temporal(frames.transpose(0, 1)[None]).transpose(1, 2)
        sequence, _ = self.sequence(features)
        sequence = sequence[0]
        logits = self.head(sequence.mean(0))
        # Sixteen ordered temporal intervals, rather than one frequency-average token.
        temporal = F.adaptive_avg_pool1d(sequence.transpose(0, 1)[None], 16)[0].transpose(0, 1)
        tokens = self.projection(temporal) + self.symbol_projection(logits.softmax(-1))[None]
        return self.norm(tokens + self.time_position.weight), logits


def decode_ocr(logits):
    """Greedy CTC collapse: blank is 0; repeated glyphs require an intervening blank."""
    previous, output = -1, []
    for index in logits.argmax(-1).detach().cpu().tolist():
        if index != previous and index != 0:
            output.append(OCR_CHARACTERS[index - 1])
        previous = index
    return "".join(output)


class LimitedAssistant(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SelftrainedConfig()
        self.lm = SelftrainedLanguageModel(self.config)
        self.vision_encoder = VisionEncoder(self.config.width)
        self.ocr_encoder = OCREncoder(self.config.width)
        self.audio_encoder = AudioEncoder(self.config.width)

    @staticmethod
    def modality_token_count(kind):
        return MODALITY_TOKENS[kind]

    def _embeddings(self, input_ids, attention_mask, modalities):
        embeddings = self.lm.embedding(input_ids)
        modalities = [[] for _ in range(len(input_ids))] if modalities is None else modalities
        if len(modalities) != len(input_ids):
            raise ValueError("Modalities must have one ordered payload list per batch row")
        perception = []
        for row, payloads in enumerate(modalities):
            locations = {kind: (input_ids[row] == token).nonzero().flatten() for kind, token in MODALITY_IDS.items()}
            expected = {
                kind: sum(p.get("kind") == kind for p in payloads) * count for kind, count in MODALITY_TOKENS.items()
            }
            if any(len(locations[kind]) != expected[kind] for kind in locations):
                raise ValueError(
                    "Reserved modality slots must exactly match the payloads; no missing or gold-filled slots"
                )
            consumed = dict.fromkeys(MODALITY_TOKENS, 0)
            for index, payload in enumerate(payloads):
                if set(payload) - {"kind", "values", "valid", "coordinates"}:
                    raise ValueError("Model payloads accept pixels/features and public geometry only")
                kind = payload.get("kind")
                if kind not in MODALITY_TOKENS:
                    raise ValueError("Unknown modality payload")
                values = torch.as_tensor(payload["values"], device=embeddings.device).to(embeddings.dtype)
                if not torch.isfinite(values).all():
                    raise ValueError("Modality input contains nonfinite values")
                if kind == "image":
                    if "coordinates" not in payload:
                        raise ValueError("Image payload must supply public slot coordinates")
                    coordinates = torch.as_tensor(payload["coordinates"], device=embeddings.device).to(embeddings.dtype)
                    if not torch.isfinite(coordinates).all() or not ((coordinates >= 0) & (coordinates <= 1)).all():
                        raise ValueError("Image coordinates must be finite normalized x,y centers")
                    tokens, logits = self.vision_encoder(values, coordinates)
                elif kind == "ocr":
                    tokens, logits = self.ocr_encoder(values)
                else:
                    valid = payload.get("valid")
                    valid = None if valid is None else torch.as_tensor(valid, device=embeddings.device)
                    tokens, logits = self.audio_encoder(values, valid)
                start, count = consumed[kind], MODALITY_TOKENS[kind]
                slots = locations[kind][start : start + count]
                if attention_mask is not None and not attention_mask[row, slots].all():
                    raise ValueError("A modality slot cannot be padding")
                if len(slots) > 1 and not torch.equal(slots[1:], slots[:-1] + 1):
                    raise ValueError("Each modality payload must occupy one contiguous reserved span")
                embeddings[row, slots] = tokens
                consumed[kind] += count
                perception.append({"batch_index": row, "modality_index": index, "kind": kind, "logits": logits})
        return embeddings, perception

    def forward(
        self, input_ids, attention_mask=None, labels=None, modalities=None, *, positions=None, segments=None, cache=None
    ):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a long [batch,time] tensor")
        if cache is None:
            embeddings, perception = self._embeddings(input_ids, attention_mask, modalities)
        else:
            if modalities is not None:
                raise ValueError("Cached continuation uses the encoded prefix; do not resend modalities")
            if any((input_ids == token).any() for token in MODALITY_IDS.values()):
                raise ValueError("A cached continuation cannot introduce unencoded modality slots")
            embeddings, perception = self.lm.embedding(input_ids), []
        result = self.lm(embeddings, attention_mask, positions, segments, cache)
        result["perception"] = perception
        result["loss"] = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("Labels must already be shifted and align with the reserved modality slots")
            if attention_mask is not None and (labels[~attention_mask] != IGNORE).any():
                raise ValueError("Padding cannot carry supervised labels")
            total, count = loss_sum(result["logits"], labels)
            result["language_loss"] = total / count
            result["supervised_tokens"] = count
            result["loss"] = result["language_loss"] + self.config.auxiliary_weight * result["aux_loss"]
        return result

    def generate(self, input_ids, modalities=None, max_new_tokens=96, eos_id=2, use_cache=True, temperature=0.0):
        """Return newly generated tokens only; one unpadded conversation per call."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or (input_ids == 0).any():
            raise ValueError("Generation accepts one unpadded conversation")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        was_training = self.training
        self.eval()
        sequence, state, generated = input_ids.clone(), None, []
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if sequence.shape[1] >= self.config.max_length:
                        break
                    current = sequence if state is None else sequence[:, -1:]
                    result = self(
                        current,
                        modalities=modalities if state is None else None,
                        cache=state if use_cache else None,
                    )
                    scores = result["logits"][:, -1].clone()
                    # Control markers belong to the renderer. EOS and visible UNK remain available.
                    scores[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]] = float("-inf")
                    next_id = (
                        scores.argmax(-1, keepdim=True)
                        if temperature <= 0
                        else torch.multinomial((scores / temperature).softmax(-1), 1)
                    )
                    generated.append(next_id)
                    sequence = torch.cat((sequence, next_id), 1)
                    state = result["cache"] if use_cache else None
                    if next_id.item() == eos_id:
                        break
        finally:
            self.train(was_training)
        return torch.cat(generated, 1) if generated else input_ids.new_empty((1, 0))

    def description(self):
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters())

        language = count(self.lm)
        perception = {
            "image": count(self.vision_encoder),
            "ocr": count(self.ocr_encoder),
            "audio": count(self.audio_encoder),
        }
        expert_parameters, router_parameters = 0, 0
        for block in self.lm.blocks:
            if isinstance(block.ffn, PaddingSafeMoE):
                expert_parameters += count(block.ffn.experts)
                router_parameters += count(block.ffn.router)
        active = language
        if self.config.architecture == "moe":
            active -= expert_parameters * (self.config.experts - self.config.top_k) // self.config.experts
        return {
            "version": MODEL_VERSION,
            "config": asdict(self.config),
            "parameters": count(self),
            "language_parameters": language,
            "perception_parameters": perception,
            "expert_parameters": expert_parameters,
            "router_parameters": router_parameters,
            "active_language_parameters_per_token": active,
            "active_with_all_perception_parameters": active + sum(perception.values()),
            "active_count_convention": "All shared LM weights plus selected FFNs; includes full embedding/output matrices. "
            "Perception counts are encoder weights used per input, not per language token or a FLOP estimate.",
            "initialization": "All modules initialized from scratch by PyTorch; no pretrained weights loaded.",
            "generation_policy": "Renderer-only BOS/PAD/role/modality IDs are masked during generation. "
            "EOS and visible UNK are available; no content or answer lookup is applied.",
            "dense_comparison": f"Same width/layers/attention/tokenizer/perception and one same-width FFN versus "
            f"top-{self.config.top_k} MoE. Matches backbone and FFN hidden size, not total parameters, router overhead, "
            "FLOPs or memory; the one-FFN active comparison applies to top-1 only.",
        }


def generate(model, input_ids, **kwargs):
    return model.generate(input_ids, **kwargs)
