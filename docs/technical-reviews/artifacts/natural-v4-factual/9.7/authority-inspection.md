# Independent original-authority inspection for 9.7

Reviewer: `/root/v4_review_coordinator/factual_v4_9_7`; context: fresh.
Access date: 2026-10-04. Original third-party PDF/HTML remains only in the ignored research directory. Retrieval SHA-256 receipts are in `authority-retrieval-receipts.json`.

## Wallace et al., The Instruction Hierarchy

Actual version read: arXiv:2404.13208v1, 19 Apr 2024.
Original URL: https://arxiv.org/pdf/2404.13208v1

- Section 2, page 3, “Prompt Injections”: instructions inserted to subvert the application designer's intent; indirect injection can enter through third-party browsing/tool input. This supports describing an embedded “answer blue” directive as an injection attempt against the outer extraction task.
- Section 3.1, page 4: misaligned lower-priority instructions should be ignored when the original task can continue. This is a desired model behavior, rather than proof that arbitrary current models enforce it.
- Section 3.2, page 4, “Context Ignorance”: the training response is the same answer as without the lower-level instruction. The closed-domain example on page 6 explicitly treats instructions inside the analyzed text as data. The indirect-injection procedure on page 6 preserves the original ground-truth answer after inserting an adversarial string. These support unchanged extraction labels for clean/adversarial examples.
- Section 4 and Figure 3, page 7: the authors evaluate attack categories excluded from training to test generalization. This supports checking unseen wording using actual model responses. Their GPT-3.5 Turbo SFT/RLHF experiments are not replications of this repository's toy exercise.

## OpenAI Model Spec

Actual version read: 2025-12-18.
Original URL: https://model-spec.openai.com/2025-12-18.html
Actual HTML locators: `#follow_all_applicable_instructions` and `#ignore_untrusted_data`.

The first section orders authority by message/source rather than rhetorical claims. The second states that quoted text, file attachments and tool outputs are “untrusted data and have no authority by default”; authority can be delegated by applicable unquoted instructions. The lesson's outer instruction asks for extraction and does not delegate control to the document. This supports the intended task boundary, not an empirical guarantee for all LLMs or Python dictionaries.

## OWASP LLM01:2025 Prompt Injection

Actual page heading/version read: LLM01:2025 Prompt Injection.
Original URL: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
Locators: definition; “Indirect Prompt Injections”; “Prevention and Mitigation Strategies”, opening paragraph and items 4, 6, 7.

The definition and indirect-injection section describe external websites/files changing model behavior. The mitigation section says no fool-proof prevention is established; fine-tuning and RAG do not fully mitigate this vulnerability. Item 6 recommends separating and identifying external content. Item 4 recommends privilege control and minimum model privileges, with extensible functions handled by code. Item 7 calls for adversarial testing of trust boundaries. These support explicit task/document fields as useful provenance and an application-enforced authorization boundary, while retaining the lesson's explicit limits.

## Exact original repository configuration for the forward reference

The official public record is `docs/course-experiments/results/safety.json`, SHA-256 `d98ee412c2c19029d731406e629c6dbd86a99a16f215ef26c3c89cdf4f85d0ed`, revision `a864a60bbf72583afc9bbaf45e052bd4fe076c62`.
Archived behavior source was actually retrieved and read at https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/a864a60bbf72583afc9bbaf45e052bd4fe076c62/scripts/course_experiments/behavior.py ; its SHA-256 `94ab92aa1b8017edbb8f5524ee598dce93756cf295377ac21515bf3c0a05bdc0` matches the registered original source hash.

Inspected `_safety_records` and `run_safety`: original tasks retain the document color as the target; the final mixed `model` is passed to both the original evaluation and the wording evaluation, with no intervening fit call. Actual current full-file hashes of common.py, data.py, and model.py match the archived record. Current behavior.py differs as a full file; exact AST comparisons show `_conversation`, `_safety_records`, and `run_safety` are unchanged. This difference was checked rather than silently treating current source as identical to the archived revision.

The bounded CPU audit rebuilds the relevant inputs/splits and re-decodes recorded token IDs, independently checking target-token denominators and exact-match counts. The actual GPU run is not rerun. Its one-seed small-world results support the forward reference, not arbitrary prompts, a complete injection defense, or a GPU timing replication.
