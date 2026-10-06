# 7.5 actual reinspection by its original technical owner

Date: 2026-10-06. Reviewer: `/root/phase4_factual_coordinator/factual_7_5`.
I read the entire current 7.5 directly and compared it with my own frozen original section.
I preserved my prior canonical report as opaque bytes, without loading its judgment into this reinspection.
I did not read author repair answers, another review, or a coordinator gate judgment.

## Actual changes and their scope

Current section SHA-256: `80eca03ddf1c11687dcaf4bca50ddab789573362023c4edb6702935252c93f11`.
Prior frozen section SHA-256: `96c80abf3f99d994fce39e78762439f394bef5cef93cbcc2a635a7ae01dc639f`.
The full exact diff is saved in `own-original-to-current.diff`.

The substantive code change is `logits.grad[0,0]` to `logits.grad[0,2]`.
The first print label changes from first position to question position; the second label now says question embedding, rather than the fixed spelling Q.
The accompanying explanation and Z exercise consistently describe the question's position, not the first/BOS position.
There is also a wording correction in the opening gradient sentence.
This moves the displayed direct score derivative to the same physical token position as the input embedding row under discussion.
No change was made to target alignment, tokenizer, seed, model width, forward pass, retain_grad, loss or backward.

## Current execution and independent controls

I executed the entire modified fence through the CPU/offline section helper, not the old fence.
Its actual bytes have SHA-256 `ce8105e8b2babd7ca071a37a03d8451e2d9fe5316fca7b0830fbeee0583319db`; the old fence SHA is `8698a96fa19ef3d2b4d6310b31c412152541aee65104aa15cf0ad1d376d248e1`.
The helper exited 0 and printed question-position logit-gradient norm 0.0 and question-embedding norm 0.00571822514757514.
The original current section, fence, bootstrap, exact command, stdout, stderr and CPU/imported-module environment are permanently saved under `current/`.

`current_controls.py` independently executes those actual current bytes and saves an actual Z-only variant.
Both questions remain at X index 2; Q ID=89, Z ID=98; labels remain [-100,-100,-100,-100,73,2].
Q and Z question-logit gradient norms are exactly 0; embedding norms are respectively 0.00571822514757514 and 0.004994203336536884.
All actual fence model parameters equal a separate model initialized with the same seed, confirming backward has not changed parameter values.

The independently derived finite, unweighted integer-class CE has logits axes [batch=1, position=6, candidate=264] and effective target positions 4 and 5.
The denominator is 2, not six positions or one conversation. The derivative is M_t/N*(softmax(z_t)-onehot(Y_t)).
Manual float64 CE=5.618060945615395 versus observed 5.618060945615394; absolute difference 8.88e-16. Maximum analytic-gradient difference is 2.60e-18, both below 1e-14.
As a loss-only control, keeping the same logits but adding target A73 at physical question position 2 increases the denominator to 3 and makes that direct logit gradient nonzero (norm 0.33264447956724363).
That altered label is only a mechanism check; it is not represented as a correct dialogue training example.
All-ignored labels still raise the explicit repository ValueError before attempting an empty average.

Blocking Q as an attention key in the unchanged single-layer untied model makes its embedding-row derivative exactly zero.
Detaching the lookup output leaves every forward logit equal and the output table trainable, but embedding.weight.grad is None.
The mask axes are [batch, head, query, key], shape [1,1,6,6]. Query4 can read question key2 and cannot read future answer key5.
Changing future A->B changes query4 logits by exactly zero; changing Q->Z changes them by 0.042634427547454834.
These controls preserve the explanation's distinction between loss selection and visibility. They do not guarantee nonzero derivatives in every parameter state.

## Personally checked reuse

`scope-and-reuse.json` records exact SHA comparisons against my original snapshots for model/data/attention/FFN/build/bootstrap/helper/checker contracts.
All eight executable or protocol snapshots match. The factual-reviewer method changed and was read again in its current full form; its current bytes are saved.
All 16 previously obtained original authority snapshots match their original download-provenance SHA values. No new paper or official-document fetch was needed.
Their original authority, version and locator evidence remains applicable to the unchanged mathematical operation and implementation.
I additionally reread the saved original Transformer PDF §3.1/§3.2.1 Eq.(1), original PyTorch CrossEntropyLoss class-index/ignore/shape equations, Autograd chain-rule explanation, Tensor.is_leaf/retain_grad, and torch.norm vector behavior at the prior precise locators.
The docs are PyTorch 2.9; the actual CPU execution remains 2.14.1+cpu, git `5c4886908584029761b579af026dcfb627c84070`. The installed-commit original functional/_tensor source snapshots remain unchanged and support the same API contracts.
The unmodified one-dimensional norm semantics and nonleaf-retention controls remain supported by the prior personal original-code execution evidence whose code/stdout hashes are checked in the reinspection receipt.

## Figure and supplied preview

Current 7.5 still references no image; figure consistency is not applicable to this section.
The original contextual 7.4 alignment SVG is byte-identical to my previously rendered/viewed copy. Its exact Q index2/ID89 and answer positions remain applicable; it was not presented as a newly executed rendering.
The supplied page returned HTTP200. Exactly one rendered pre block matches the current fence's text, including `[0, 2]`.
I used existing system Chromium through Playwright to capture desktop 1280×800 and mobile 390×844 pages and code crops, and actually viewed the two current-code crops.
Chinese labels are readable; the modified index is visible in the desktop crop. The mobile code crop has horizontal overflow; exact code completeness is checked by DOM text, not by pretending the entire long line fits the screenshot.
An initial default Playwright launch lacked its managed browser. That attempt's stderr is retained; using existing `/usr/bin/chromium` succeeded without installing anything.
The HTTP receipt's earlier statement that no browser visual check was claimed records that stage. The later browser receipt and actual image views extend that evidence; full-page screenshot files are saved, but only the code crops were visually inspected.

## Result and limits

The changed code and explanation are consistent. Current verdict: pass; no unresolved substantive claim.
The entire current fence and necessary Z/loss-axis/denominator/attention/gradient controls were rerun on CPU.
Unchanged authority and implementation evidence is explicitly reused after fingerprint checks.
No whole-chapter rerun, training, existing-model-weight evaluation, data/model download, GPU work, paid compute, textbook edit or commit occurred.
