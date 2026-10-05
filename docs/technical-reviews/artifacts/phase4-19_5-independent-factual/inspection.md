# 19.5 independent factual inspection

Reviewer task: `/root/phase4_factual_coordinator/factual_19_5`.
This reviewer personally read the current lesson, sources and raw records. No old technical/reader review conclusions or author outcome summaries were read. Filename searches showed old artifact paths, but only original PDF locator records were used. The PDF contents were then independently inspected.

## Frozen input and actual reading scope

- Initial 19.5 original UTF-8 bytes: source lines 217–258; SHA-256 `2198766b2260e04563c02849c36a5b38042e398a63a2212e97551849b99d3f21`. One Python fence, no figures. No rendering is applicable to this lesson.
- Complete chapter frozen bytes are saved only as input provenance in `inputs/course/chapters/19.md`; SHA-256 `3a008303054604279f49a1c00b70792db43bc4136e5dba9380990f04cced3537`. This is the chapter version at extraction, not a claim about later chapter edits. Only 19.5 and the necessary 19.12 cross-reference context were read.
- 19.12 context: heading and tables, source-section lines 14–43 and 50–end, to inspect the claimed final capability-table reference. Initial extraction of the final section encountered an IndexError from assuming a following header; corrected extraction uses EOF. No source file changed. The actual selected content was read after correction. All context belongs to the lesson manuscript, not an author review summary.
- Read required factual instructions, technical checker schema, section-facts implementation, clear-tutorial skill and review protocol. Read cloud setup skill and onboarding reference; the existing `.venv` sufficed and no environment configuration change was needed.

## Repository original method

AST listed function/class names and source line spans before selective source reads. In `tiny_perceptron/capstone.py`, inspected imports/constants lines 1–28, `_row` 124–136, `build_dataset` 139–282, `prompt_ids` 299–305, `encode_record` 308–318, `prepare_batch` 321–338, `generate_trace` 379–424, `generate_traces` 428–496, `evaluate_rows` 499–547, `parse_action` 550–561, `expected_final` 601–605 and `export_inference` 655–694. In `tiny_perceptron/data.py`, inspected ByteTokenizer 14–28. In `scripts/course_experiments/capstone.py`, inspected training prerequisites/provenance lines 70–187 and objective/update/validation code 193–244. The training functions were inspected, never executed. The student experiment source was located by AST and frozen, but its body was not read or used as evidence.

The build_dataset manifest's majority baseline definition and synthetic/narrow-template limitations describe methods and scope, not previously measured wins or review conclusions. Ordinary comments about split design and DPO replay describe the original recipe and were treated only as methods. `validation_summary` in the training receipts was not inspected.

## Raw existing results and provenance

Before raw value reads, listed top-level keys/types in each specified validation/data/train-report JSON and immediate record/trace/manifest shapes. Full untouched originals and hashes are saved. The named JSON pointers actually consumed are recorded in `verification.json#/actual_json_pointers`. These include raw task IDs, input mapping, token IDs, generated strings, EOS, expected values, action/final outcomes and training stage/code/checkpoint provenance. No notes/review/scope-correction fields were accessed.

All ten raw GitHub files were fetched over HTTPS at commit `1df335318bda03fd771807f66976953231d5a00b`; TLS verification remained enabled. Every fetched SHA matched its current local original. URLs, returned HTTP status 200, date and hashes are recorded in `sources/pinned-source-provenance.json`.

The historical receipts identify seed 42, SFT 1400 updates, joint 600 updates and DPO 100 updates, with completed schedules on CUDA. That CUDA training happened in the archived experiment; the present reviewer only verified raw receipts on CPU. Parent checkpoint hashes form the SFT → joint → DPO chain. The relevant code SHA values match the independently inspected raw code. No weights were downloaded, loaded or newly saved.

## Official originals personally read

- Ouyang et al., **Training language models to follow instructions with human feedback**, arXiv `2203.02155v1`, 4 March 2022, original author PDF at `https://arxiv.org/pdf/2203.02155v1`. Reused an exact original PDF located by the locator-only index, verified its raw SHA, copied it intact, and personally verified first-page title/authors/arXiv version. Read abstract and introduction, §3.1 Step 1 (desired demonstration data), §3.5 SFT, §3.6 behavior evaluation/truthfulness/harm proxies and §5.3 model limitations. Accessed the actual saved original on 2026-10-05, not a paper summary.
- Hugging Face Transformers official source documentation, tag `v4.57.1`, `docs/source/en/chat_templating.md`, fetched over HTTPS from its official repository. Personally read lines 22–27, 78–84, and 194–224: chat is a formatted token sequence; system role gives behavior directives; training contains full assistant messages with end markers. This documentation supports the general message/role method only; the course uses its own ByteTokenizer and does not call Transformers.

## Executions and conclusions

1. `.venv/bin/python docs/review-tools/section_facts.py course/chapters/19.md#19.5 --output /tmp/phase4-19_5-independent-factual-run --execute --timeout 60`: exit 0, attempted fence [1], no guard events. Permanent copies of the fence, bootstrap, environment, stdout/stderr and command/result JSON are in `fence-run/`. Only regular files were copied, not the transient workspace or symlinks. Output prints four authors' training targets and constructs no model.
2. Offline CPU command recorded in `commands.json` executes `verify_19_5.py`, reads the existing raw result files, reconstructs their input/token mappings and counts, and checks short tokenizer/action-parser variants. Exit 0; stdout and verification JSON are saved. This is not new model evaluation, training, or data preparation.
3. SFT four task scores are style 3/3, missing 3/3, unavailable 10/10, safety 3/3, all with genuine EOS ID 2. Remaining text tasks are calculator 10, tool_return 5, concept 5 and rag 3, totaling 23/23. Total text is 42/42 for all three stages. The three rag truth labels are exactly `DIRECT:書櫃`. Per-task constant payloads also achieve missing 3/3, unavailable 10/10, safety 3/3. These are ceilings in this fixed set and do not establish general safety, calibration or varied-source QA.
4. Prompt IDs match source rows exactly; generated IDs decode to the saved raw strings. ID `691c9de656c01fa2df60` has original question 「照抄數字15，只要答案。」, generic system 「計算器=開；風格=短。」 and genuine SFT output `DIRECT:15`. The generic system does not contain per-question answer values.
5. Encoding variants verify that assistant targets include EOS while the input prefix is ignored in the SFT loss. Removing EOS makes the parser reject the otherwise correct string. A same-question availability counterfactual changes generic system availability and switches its target from TOOL to ASK. These verify data and grammar, not learned model generalization.
6. Joint to DPO correctness changes comprise exactly four joint-task records; all other task scores stay unchanged. Joint goes from 18/18 to 14/18. Saved exact changed raw outputs replace blue with red; pitch stays as originally generated. This supports only this fixed-set regression.
7. Hand example: requesting bag and boot from [bag, boot, trousers] gives two selected items, each on one line. Listing all three violates scope; adding an introduction violates the requested two-line format. A unit price of 14 gives totals 14 and 28 for quantities 1 and 2, so no unique total follows without quantity. These are problem-specification calculations, not trained model scores.
8. Future list changes, revised history, correct content/range/format/end and answerable/insufficient pairs are proposed acceptance criteria. No new mainline product capability was asserted or tested.

## Initial revision finding

The final sentence claims 「最終文字能力與各項分母統一見19.12的能力表」. In the frozen 19.12, the main table is a future acceptance matrix and the existing capability table expressly lists only four selected old tasks (calculator, image color, image shape and audio). It does not list the four text behaviors or their final text test denominators. The reference therefore promises evidence outside the linked table's actual scope and risks mixing this lesson's validation evidence with final test evidence. Request changing the reference to the capability definitions/criteria that 19.12 actually contains, leaving the current validation-result links explicit. Initial verdict: revise. No content or figure was edited by this reviewer.
