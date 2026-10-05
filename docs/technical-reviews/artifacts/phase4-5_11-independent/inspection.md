# 5.11 independent original-source inspection

Reviewer: /root/phase4_factual_coordinator/factual_5_11, fresh context; 2026-10-05.
No previous technical or reader report was read. No child agent was used.

Read the whole original 5.11 section (course/chapters/05.md lines 393–427),
T.4 (course/training.md lines 123–216), the immediately preceding closing
boilerplate and the first explanatory example of 5.12 (lines 428–440) while
locating the section boundary, assets/training/README.md, the full factual
reviewer instructions, the full reader review protocol, section_facts.py and
the checker schema. This is not a chapter-first section; no chapter-intro
review is claimed. No earlier reports or their verdicts were consulted.

## Primary sources personally read

- Python 3.13 official built-in types documentation: str.split (text snapshot
  lines 3500–3568), str.join (3116–3130), str.encode (2554–2565).
  split without sep groups all consecutive Unicode whitespace and also
  discards leading/trailing empty fields; join uses the receiver as separator;
  encode defaults to UTF-8. The lesson's three inputs have no edge whitespace.
  This is a content-comparison convention, not a semantic-equivalence test.
- Python 3.13 official hashlib documentation: Hash algorithms (286–303),
  hash.digest_size (534–540), hash.digest and hash.hexdigest (586–602).
  sha256 consumes bytes; the hexadecimal text is twice the byte digest size.
  Documentation snapshot last updated Oct 01, 2026; execution Python is
  3.13.5, torch 2.14.1+cpu. No installed/current-major API mismatch occurs.
- NIST FIPS PUB 180-4, August 2015: introductory Explanation item 3
  (snapshot lines 74–87) specifies secure digest, hard preimage/collision
  search and a changed digest with high probability for changed data.
  Section 1 Introduction and Figure 1 (lines 227–261, printed page 2)
  call these one-way functions and specify SHA-256's 256-bit digest.
  Fixed-size output over many possible inputs cannot be reversible storage;
  a shorter prefix has fewer possible identities. These are identity/security
  properties, not similarity measurements. No attack-resistance benchmark or
  universal no-collision guarantee is claimed.
- Lee et al., Deduplicating Training Data Makes Language Models Better,
  ACL 2022 long paper 577, pp. 8424–8445, downloaded directly from the
  publisher. Personally read abstract and Introduction pp. 8424–8425
  (snapshot lines 10–84), and Section 4 p. 8426 (167–179).
  The paper documents train/validation overlap and explains that exact full
  string comparison misses partial and approximate duplication. This supports
  leakage risk and the lesson's scope limitation. Its reported model scores,
  percentages or training benefits are not asserted for these toy inputs or
  the T.4 pilot and are not used as evidence for the local no-deletion claim.
- Python 3.13 language reference, Sections 2.1.8–2.1.9, text lines 437–441
  and 518–525: indentation determines statement grouping and whitespace
  inside literals is a special case. This directly supports the caution
  about whitespace having meaning. Poetic formatting is an application
  choice: the lesson says it *may* matter, not that a particular metric was
  measured for poetry.

Raw fetched sources, extracted texts, official HTTPS URLs, returned status,
hashes, byte counts and PDF conversion commands are in sources/.

## Code and original-data inspection

Read tiny_perceptron/data.py fingerprint lines 102–104 and ByteTokenizer
lines 14–27; scripts/course_experiments/text.py _deduplicate_text lines 91–99,
_save_splits lines 46–61 and run_real_text lines 317–345; common.py
split_records lines 50–70 and text_examples lines 77–97. Cleaning uses the
normalized full string as the actual seen key, retains the first entire row
including original text, assigns full SHA-256 family, and only then splits
groups. text_examples reads the original text and encodes UTF-8 bytes.

Original real_text report revision is 5581462ef01959636425eb7795ae3153142dbfeb.
The retained local historical text.py/common.py from git show match the report's
code_sha256 exactly, and both match this review's frozen current snapshots.
The probe executes those exact historical function source spans. No original
GPU recipe or training loop was executed; the original GPU environment is
recorded as historical identity, distinct from this CPU audit.

The original report and the existing raw 512/365 JSONLs are permanently
copied under inputs/. Each JSONL byte size and SHA-256 matches the fixed asset
manifest. Recomputing normalized full-string identities, cleaning, seed=42
split serialization, counts and all six complete JSONL hashes reproduces
the original report exactly. Both sets have zero deletions at this step.
All raw text values remain exactly equal. There are 491 stories and all 365
poems containing newlines. One newline-containing record from each source was
also passed through the original text_examples implementation: every target
equals the original UTF-8 byte sequence followed by EOS (700 and 171 targets).
No model score, inference, optimizer, training or near-duplicate detection was
used. No weights were read or copied.

## Executions and limitations

The original fence was executed unchanged by section_facts.py with a 45-second
timeout in the existing .venv CPU environment, exit 0. Original stdout and
environment are under original/. Exact-fence variations confirm replacing
the second text with 鳥 看狗 changes unique hashes from 2 to 3 and fails the
old equality assertion; the inequality assertion succeeds. Added whitespace
keeps two identities. Extra edge cases cover Unicode whitespace, empty inputs,
and word-order/character changes. A one-hex-character prefix collision is
explicitly found between document 0 and document 5, whose full digests differ;
it demonstrates information loss only. The one-character example changes 125
of 256 bits, an illustration rather than an avalanche guarantee.

Uniform record sampling gives the duplicated content probability 2/3 versus
1/2 after deduplication. This is an exact sampling calculation, not a model
quality result, and its scope assumes records are sampled uniformly without
duplicate-adjusted weights. It explains the local distribution claim.

The first private whitespace-syntax probe mistakenly assumed a one-statement
suite would stop compiling when flattened; Python permits a one-line suite.
It failed before the data audit. initial-probe.py/stdout/stderr retain that
attempt. The corrected demonstration includes a following statement, whose
flattened form fails parsing. This was a reviewer-probe issue, not a lesson
defect. The final probe passed all assertions after adding actual raw text
tokenization and group-descendant checks.

5.11 references no image or SVG, confirmed both in source text and extraction
metadata. There is therefore no original diagram to render or view. The
figure-consistency check is not applicable. No browser, desktop/mobile page
render or full-course visual verification is claimed. The normalized strings
and fingerprints in textual stdout are sufficient for this section's concrete
identity-comparison task.

Verdict: pass. No substantive unresolved factual issue was found. The preserved
evidence supports only the current section's claims and the documented
full-content no-deletion/split identities; it does not establish that the
corpora contain no similar or rewritten stories, nor any model-quality effect.
