"""Save source fingerprints and short inspected passages, never third-party full text."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EXT = Path("/workspace/work/reference/technical-glossary-own")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote(path, ranges, limit=1550):
    # Match nl/sed line numbers: PDF form-feed markers are not new source lines.
    lines = Path(path).read_text(errors="replace").split("\n")
    return "\n".join(f"{n}: {lines[n-1]}" for a, b in ranges for n in range(a, min(b, len(lines))+1))[:limit]


selected = {
    "bengio2003": ([(246, 273), (312, 322)], "§2, printed pp.1141–1142: finite vocabulary, trainable embedding rows and conditional language likelihood; no ordinal semantics for IDs."),
    "adam2014": ([(75, 98)], "Algorithm 1 and §2: per-coordinate m/v, bias correction and learning-rate-scaled update; no universal convergence claim."),
    "pytorch_crossentropy": ([(587, 621)], "CrossEntropyLoss docs 2.14: unbatched input, unreduced -log p target formula, ignore_index=-100 and none/mean/sum reductions. Averaging is a reduction, not a prerequisite for cross entropy."),
    "sklearn_leakage": ([(169, 195)], "§12.2: test data cannot make model choices; fit only on train. The course's named calibration/policy/source-holdout splits are explicit local conventions, not a claim that sklearn imposes them."),
    "gpt2_2019": ([(171, 181), (187, 204)], "§2.2: byte-level input representation and subword BPE; no claim all tokens equal one Chinese character."),
    "lora2021": ([(203, 222)], "§4.1 Eq.(3): W0 frozen, trainable B/A rank-r path added to original output; r is a chosen bottleneck and upper bound on actual matrix rank."),
    "instructgpt2022": ([(88, 99), (112, 115)], "§1 / Figure2: demonstrations→SFT, preference comparison→reward model, PPO optimization; source of reward differs from update method."),
    "dpo2023": ([(60, 67), (143, 155)], "§3 and §4: preferences directly optimize policy with a reference; avoids a separately trained reward-model/RL pipeline. No guarantee unreliable preferences become correct."),
    "attention-2017": ([(129, 143), (166, 178), (225, 260)], "§3.1–3.4 Eqs.(1),(2): residual, causal masking, Q/K/V weighting, same-source self-attention, multihead projection/concatenation, position-wise FFN and LM output. Restricted-local attention discussed in §4."),
    "rmsnorm-2019": ([(160, 190)], "§3 Eqs.(2),(3), §4 Eq.(4): LayerNorm centers and rescales; RMSNorm does not subtract mean. Learned gain/bias and numerical epsilon are omitted from the glossary's mechanism summary."),
    "vit-2021": ([(173, 191)], "§3.1 / Eq.(1): images split into patches, flattened and linearly projected; classification head maps learned image representation to classes."),
    "llava-2023": ([(242, 252), (294, 305)], "§4 Eq.(1), Stage1: linear projector maps visual features to language embedding width; image-caption likelihood aligns the two. This supports the course's simple projector, not all multimodal architectures."),
    "whisper-2022": ([(137, 145)], "§2.1: speech recognition maps audio to transcript text; published model capabilities are not reproduced in this review."),
    "whisper-audio-code": ([(12, 21), (143, 158)], "Pinned audio.py: sample rate and hop define frames, STFT magnitude squared gives power, mel filter matrix aggregates bands. Only tiny excerpt retained."),
    "librosa-mel-source": ([(225, 241)], "librosa0.11.0 filters.mel: uniformly spaced mel centers and intersected lower/upper ramps yield overlapping triangular frequency weights."),
    "mixtral-2024": ([(80, 103)], "§2: sparse softmax TopK gating, expert weighted sum and distinction total versus active parameters; inactive weights remain part of stored model."),
    "switch-2021": ([(396, 421)], "§2.2 auxiliary load-balancing loss encourages uniform expert routing; the glossary does not claim perfect balance or universal speedup."),
    "mqa-2019": ([(153, 167), (182, 194)], "§2.4 incremental self-attention caches previous K/V; dense full causal attention still reads prefix K/V. Does not cover sliding-window cache eviction."),
    "flashattention-2022": ([(243, 250), (277, 285), (317, 321)], "§3.1 / Algorithm1 / Theorem1: tiling plus statistics combine exact softmax attention mathematically, with reduced HBM intermediates; no bitwise equality or local GPU benchmark asserted."),
    "rope-2021": ([(210, 237)], "§3.1–3.2 Eq.(11),(12), later Eqs.(14)–(16): pairwise rotations encode position/relative dot products. No long-context quality guarantee."),
    "quantization-jacob-2018": ([(132, 155)], "§2.1 Eq.(1): r=S(q-Z); adjacent integer levels separated by positive S. This is affine scalar quantization, not every quantizer."),
    "kd-2015": ([(44, 48), (75, 86)], "§1–2: trained teacher supplies soft targets to student; teaching data/soft distributions do not imply the student inherits arbitrary teacher capabilities."),
    "rag-2020": ([(119, 128)], "§2: retrieve documents z for x and condition generation on x,z. Inference retrieval is context input; paper also trains RAG models, so no claim RAG training is impossible."),
    "passk-codex-2021": ([(150, 176)], "§2.1 Eq.(1): at least one passing candidate among k; unbiased finite-pool estimator samples n>=k and counts c. Tests define the success criterion."),
    "reinforce-1992": ([(240, 270)], "§4 Theorem1: (r-baseline) times gradient log probability; output-independent baseline condition; expected gradient, not each individual sampled step's guarantee."),
    "guo2017": ([(96, 109), (290, 297), (433, 443)], "§2 Eq.(1), §4 assumptions and §4.2 Eq.(9): calibration, held-out T>0 and logits/T; source assumes same distribution and cannot promise source-transfer calibration."),
}
bank = {}
for relative in ["docs/technical-sources/text-behavior.json", "docs/technical-sources/systems-modal.json"]:
    for item in json.loads((ROOT / relative).read_text())["sources"]:
        bank[item["id"]] = item

sources = []
receipts = []
for sid, (ranges, note) in selected.items():
    item = bank[sid]
    lc = item.get("local_cache", {})
    text_path = ROOT / (item.get("evidence_local_path") or lc.get("text") or lc.get("original"))
    raw_path = ROOT / (item.get("artifacts", {}).get("original_local_path") or lc.get("original") or str(text_path.relative_to(ROOT)))
    kind = "official_source" if text_path.suffix == ".py" else "official_docs" if sid.startswith("pytorch") or sid == "sklearn_leakage" else "paper"
    source = {"id": sid, "kind": kind, "title": item["title"], "url": item["url"], "version": item.get("version") or item.get("retrieved_version"), "verified": True, "checked_original": True, "accessed_on": "2026-10-29", "authority_reason": "Original authors/publisher or maintained official implementation; locally cached original was directly inspected by this reviewer.", "inspection_note": note}
    sources.append(source)
    receipts.append({"source_id": sid, "raw_cache_path": str(raw_path), "raw_sha256": sha(raw_path), "text_cache_path": str(text_path), "text_sha256": sha(text_path), "inspected_locator": note, "small_original_excerpt": quote(text_path, ranges)})

custom = [
    ("utf8", "official_docs", "RFC 3629: UTF-8", "https://www.rfc-editor.org/rfc/rfc3629", "RFC3629, November2003", EXT/"utf8-rfc3629.txt", [(192, 201), (230, 235)], "§3 character table and surrogate exclusion: 1–4 octets, U+D800–DFFF not permitted."),
    ("json", "official_docs", "RFC 8259: JSON", "https://www.rfc-editor.org/rfc/rfc8259", "RFC8259, December2017", ROOT/"outputs/technical-sources/7.13-fresh/rfc8259.txt", [(149, 166), (248, 257)], "§1 and §2: six value types; serialized JSON text need not be an object. The glossary's brace syntax is an object example, not a complete JSON grammar."),
    ("ppo", "paper", "Proximal Policy Optimization Algorithms", "https://arxiv.org/pdf/1707.06347v2", "1707.06347v2", EXT/"ppo-v2.txt", [(104, 132)], "§3 Eq.(7): minimum of ratio*advantage and clipped-ratio*advantage; clipping objective does not hard-bound every policy change."),
    ("mtp", "paper", "Better & Faster Large Language Models via Multi-token Prediction", "https://arxiv.org/pdf/2404.19737v1", "2404.19737v1", EXT/"mtp-v1.txt", [(64, 86)], "§2: one context produces n future-token predictions through shared trunk, independent Transformer heads, shared unembedding."),
    ("pi", "paper", "Extending Context Window of Large Language Models via Positional Interpolation", "https://arxiv.org/pdf/2306.15595v2", "2306.15595v2", EXT/"pi-v2.txt", [(236, 253)], "§2.3 Eq.(4): position m is mapped to mL/Lprime before RoPE; coordinate formula does not establish task quality."),
    ("loop", "paper", "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", "https://arxiv.org/pdf/2502.05171v2", "2502.05171v2", EXT/"loop-v2.txt", [(126, 137), (152, 166)], "§3.1 Figure2: shared recurrent core R repeatedly consumes previous latent state; more iterations perform more core computation without duplicating R parameters."),
    ("spec", "paper", "Fast Inference from Transformers via Speculative Decoding", "https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf", "PMLR202 (2023), original proceedings PDF", EXT/"speculative.txt", [(114, 132)], "§2.3 and Algorithm1: proposal q, acceptance p/q and corrected residual distribution preserve target distribution; simple greedy agreement is a restricted case."),
    ("vqvae", "paper", "Neural Discrete Representation Learning", "https://arxiv.org/pdf/1711.00937v1", "1711.00937v1", Path('/workspace/work/reference/vqvae-1711.00937v1.txt'), [(136, 165), (183, 204)], "§3.1–3.2 Eqs.(1)–(3): nearest code selection, straight-through encoder reconstruction gradient, codebook versus commitment stop-gradient paths."),
    ("encodec", "paper", "High Fidelity Neural Audio Compression", "https://arxiv.org/pdf/2210.13438v1", "2210.13438v1", Path('/workspace/work/reference/encodec-2210.13438v1.txt'), [(176, 195)], "§3.1: residuals quantized by successive codebooks; selected code vectors are summed; [B,Nq,T] separates codebook and time axes."),
    ("jev", "official_docs", "TypeSafe System One API", "https://docs.typesafe.ai/api.md", "OpenAPI3.1.0 info.version0.2.0; fetched2026-10-29; SHA receipt retained", EXT/"jev-api.md", [(361, 380), (413, 424), (626, 642), (676, 689), (719, 731), (785, 794)], "Question.oneOf and Answer.oneOf enumerate Noul,Choice,Score,Box (Box gated by image danger-zone header); Noul is yes-probability, Choice/Score probability maps. Thus current official API does not have exactly three total types."),
    ("jev-confidence", "official_source", "TypeSafe reference adapter choice_confidence", "https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/src/system_one_adapter/_utils/confidence_metrics.py", "tagv0.2.1 commit e1d4cc938204b22fc5a3c3aca7044072fe3f712d", EXT/"_utils_confidence_metrics.py", [(17, 24)], "choice_confidence normalizes the supplied probabilities then computes (pmax-1/n)/(1-1/n) for n>1; computed concentration, not observed correctness frequency."),
    ("kev", "official_source", "Kev PointerHead and hybrid branch caches", "https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py", "2074cc44256200bb37540103fa4572901faa10f3", Path('/workspace/work/reference/kev/kev/model.py'), [(205, 219), (342, 359)], "PointerHead q/k projected dot-product; _rows_hidden forks hybrid conv/recurrent states and KV cache. Score formula alone does not establish candidate-order equivariance."),
    ("kev-readme", "official_docs", "Kev original README", "https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/README.md", "2074cc44256200bb37540103fa4572901faa10f3", Path('/workspace/work/reference/kev/README.md'), [(11, 19)], "Project declares independent family of Jev-like local decision models; public code is not proof of private commercial Jev training details."),
    ("qwen3vl-processor", "official_source", "Qwen3VLProcessor official source", "https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py", "transformers v4.57.6", ROOT/"outputs/natural-v4/factual-research/training-v4/transformers-qwen3vl-processor-v4.57.6.py", [(68, 84), (150, 164)], "Processor wraps image processor, tokenizer, video processor and chat template; image inputs produce pixel values. No local pretrained model was loaded."),
]
for sid, kind, title, url, version, path, ranges, note in custom:
    source = {"id": sid, "kind": kind, "title": title, "url": url, "version": version, "verified": True, "checked_original": True, "accessed_on": "2026-10-29", "authority_reason": "Original paper, standards body or official maintained source, directly inspected.", "inspection_note": note}
    sources.append(source)
    raw = path.with_suffix('.pdf') if path.with_suffix('.pdf').is_file() else path
    receipts.append({"source_id": sid, "raw_cache_path": str(raw), "raw_sha256": sha(raw), "text_cache_path": str(path), "text_sha256": sha(path), "inspected_locator": note, "small_original_excerpt": quote(path, ranges)})

for sid, file, url, version, phrases in [
    ("q25", "qwen25-omni-report-v1", "https://arxiv.org/html/2503.20215v1", "2503.20215v1", ["Qwen2.5-Omni employs Thinker-Talker", "TMRoPE encodes", "Talker receives both"]),
    ("q3", "qwen3-omni-report-v1", "https://arxiv.org/html/2509.17765v1", "2509.17765v1", ["Talker no longer consumes", "The Talker adopts a multi-codebook", "TM-RoPE factorizes", "For speech synthesis in multi-turn", "left context only multi-codebook"]),
]:
    text_path = EXT/(file+'.txt')
    text = ' '.join(text_path.read_text().split())
    excerpts = []
    for phrase in phrases:
        index = text.find(phrase)
        if index < 0:
            raise RuntimeError(f'Original text phrase absent: {sid} {phrase}')
        excerpts.append(text[index:index+500])
    sources.append({"id": sid, "kind": "paper", "title": "Qwen2.5-Omni Technical Report" if sid=='q25' else "Qwen3-Omni Technical Report", "url": url, "version": version, "verified": True, "checked_original": True, "accessed_on": "2026-10-29", "authority_reason": "Original Qwen technical report.", "inspection_note": "Directly read HTML-derived original §2.1–2.5; checked Thinker/Talker, text/multimodal conditioning, TMRoPE and frame/codebook versus token-time MTP. No reported benchmark was transferred to this project."})
    raw = Path('/workspace/work/reference')/(file+'.html')
    receipts.append({"source_id": sid, "raw_cache_path": str(raw), "raw_sha256": sha(raw), "text_cache_path": str(text_path), "text_sha256": sha(text_path), "inspected_locator": "§2.1–2.5", "small_original_excerpts": excerpts})

uax = ' '.join((EXT/'unicode-uax29-r47.txt').read_text().split())
index = uax.index('user-perceived characters')
sources.append({"id": "uax29", "kind": "official_docs", "title": "Unicode Standard Annex29: Unicode Text Segmentation", "url": "https://www.unicode.org/reports/tr29/tr29-47.html", "version": "Unicode17.0.0 revision47, 2025-08-17", "verified": True, "checked_original": True, "accessed_on": "2026-10-29", "authority_reason": "Unicode Consortium normative text segmentation annex.", "inspection_note": "Summary and grapheme-cluster treatment distinguish user-perceived characters from code points; course example thumbs-up plus skin-tone is separately verified as two Python codepoints."})
receipts.append({"source_id": "uax29", "raw_cache_path": str(EXT/'unicode-uax29-r47.html'), "raw_sha256": sha(EXT/'unicode-uax29-r47.html'), "inspected_locator": "Summary; §3 Grapheme Cluster Boundaries", "small_original_excerpt": uax[index-100:index+400]})
sources.append({"id": "torch", "kind": "official_source", "title": "Installed PyTorch official docstrings and source", "url": "https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070/torch", "version": "PyTorch2.14.1+cpu, git5c4886908584029761b579af026dcfb627c84070", "verified": True, "checked_original": True, "accessed_on": "2026-10-29", "authority_reason": "Installed official PyTorch source matching reported CPU runtime commit.", "inspection_note": "Directly inspected Linear, ReLU, GELU, Embedding, LayerNorm, F.normalize, softmax, cross_entropy, Tensor.backward, Optimizer.zero_grad/state_dict; per-file hashes and line numbers retained in verification-output.json. Includes epsilon and reduction boundaries."})
sources.append({"id": "math", "kind": "derivation", "title": "Transparent glossary derivations", "verified": True, "details": "ID lookup uses dog0/look1/cat2. Eight binary digits have2^8=256 values 0..255. softmax(2,1,0) divides exp scores by exp2+exp1+1. With signal energy fixed E>0, SNR=10log10(E/error_energy) increases when positive error_energy decreases; zero error gives positive infinity. KL(p||p)=sum p(logp-logp)=0 and entropy=-sum p logp with0log0=0; KL requires q positive wherever p positive. Positive T preserves within-row logit inequalities but two unrelated rows need not preserve maximum-softmax order, as shown by executed counterexample. Coverage=A/N, accepted risk=E/A for A>0; A=0 is undefined. Unit-length L2=v/||v|| assumes v nonzero and no epsilon-clamping; default torch normalize leaves zero vector zero."})
sources.append({"id": "cpu", "kind": "execution", "title": "Independent deterministic CPU examples", "verified": True, "artifact_id": "cpu-output"})
result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_glossary", "sources": sources, "original_receipts": receipts, "retention": "Only short passages and fingerprints retained here. Third-party full texts remain in ignored outputs/ or /workspace/work/reference outside Git. No other reviewer report or sidecar inspected."}
(OUT/'original-evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({"source_count": len(sources), "receipt_count": len(receipts), "bytes": (OUT/'original-evidence.json').stat().st_size, "sources": [{"id": s['id'], "inspection_note": s.get('inspection_note','')} for s in sources]}, ensure_ascii=False, indent=2))
