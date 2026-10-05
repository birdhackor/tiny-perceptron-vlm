# Review identity and actual scope

Reviewer: /root/phase4_factual_coordinator/factual_7_8. Fresh independent technical reviewer for 7.8 only. No old technical or reader review contents or other reviewers' judgments were read. An existing report, if present, was copied as opaque bytes into history without parsing its contents. The repository-wide checker may load report metadata to check reviewer identity uniqueness; its verdict is not used to decide technical truth.

Read current 7.8 in full, original UTF-8 bytes and original Python fence; read 7.4 and 7.7 as necessary prerequisites; read `TinyLM.forward`, `generate`, attention masking/forward/manual attention, the data shifting/chat/padding contracts, `pyproject.toml`, `scripts/build_course.py:BOOTSTRAP`, and both mandated review instructions and checking helpers. This is not the first numbered chapter section; chapter introduction review is not applicable here. No GPU work, existing weight evaluation, complete training, new course dataset preparation, paid compute or model/data download occurred. The extra TinyLM instance is newly initialized random width-8, vocab-10 and runs inference only.

# Original authority inspection

All primary files were acquired directly over certificate-verified HTTPS on 2026-10-05, with exact URL, status, byte count and SHA in download receipts. Read original lines, not a search summary or candidate-source judgment. The official release snapshots are PyTorch v2.9.0, NumPy v2.1.0 and Transformers v4.56.2; the local execution environment is PyTorch 2.14.1+cpu, Python 3.13.5. The report does not equate these versions: the primary snapshots establish the documented API convention and each relevant operation was exercised on the installed CPU version.

- PyTorch `_torch_docs.py`: lines 1504-1524 logical NOT on Boolean input; 4573-4612 gather axis equation, same rank, index/output shape and no input/index broadcast; 6742-6785 amax values and reduction; 9320-9371 half-open arange range and integer dtype; 9474-9506 reshape same values/number of elements.
- PyTorch `_tensor_docs.py`: lines 3180-3195 masked_fill_ replaces positions where mask=True with a broadcastable Boolean mask; 6609-6615 masked_fill is out-of-place; 6280-6328 expand and expand_as shape equivalence.
- PyTorch `tensor_view.rst`: lines 88-92 explicitly state indexing follows NumPy behavior and link the indexing documentation.
- NumPy `basics.indexing.rst`: lines 43-61 zero-based/negative indexing; 250-260 None/newaxis adds a dimension; 297-340 advanced integer index arrays, negative values, joint broadcast and paired iteration; 389-396 output concatenates advanced-index shape with unused dimension shape.
- Transformers `modeling_outputs.py`: CausalLMOutput and CausalLMOutputWithPast lines 629-675 define next-token loss and [batch_size,sequence_length,vocab_size] logits before SoftMax.
- Transformers `generation/utils.py`: lines 2305-2318 left-padding convention and warning about right padding in decoder-only generation; lines 2881-2923 read the last sequence position, then independently process scores and sample/argmax, then append IDs. This source's generator assumes its own left-padding convention. It does not claim arbitrary padding is accepted merely by choosing one logit slice.

# Actual visual inspection

7.8 itself contains zero SVG references. Prerequisite figures `rewrite-07-07-padding-positions.svg` and `rewrite-07-04-answer-alignment.svg` were copied byte-for-byte, rendered to PNG using actual Inkscape 1.4 commands with a 30-second timeout (exit 0), then both PNGs were opened via view_image. The first shows physical columns 0..4, valid False,False,True,True,True, model positions 0..2 for true tokens. The second shows assistant at input position 4 predicting A at target position 4, then A predicting EOS. These support physical-position and next-token interpretation used in 7.8. Render stderr contains nonfatal Pango/Gtk initialization warnings; the viewed raster images are complete and legible. Browser desktop/mobile page rendering was not part of this technical figure check and is not claimed.

# Decisions

No material false or unresolved substantive claim found in the current 7.8. The explicit empty-row warning and valid/model-position limitation match the tests and repository contracts. Original and exercise integers/axes are exact. The extra exhaustive mask and model checks establish their narrowly recorded mechanics only; there is no empirical model-score claim to rerun.
