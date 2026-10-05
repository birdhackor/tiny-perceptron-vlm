# Independent phase 4 factual review, lesson 1.2

Reviewer: /root/phase4_factual_coordinator/factual_1_2, fresh context. Reviewed on 2026-10-05. No old technical report or report history body/conclusion was read. Assessment scope: all current section 1.2 bytes, source lines 50–86. Context read: chapter introduction and 1.1; the initial chapter read also extended through the start of 1.6. These contextual sections are not reviewed here. Actual implementation read: tiny_perceptron/data.py lines 1–145, especially split_documents lines 89–99. Read scripts/build_course.py BOOTSTRAP and the entire reviewer instructions, checker schema, and section_facts helper. Section 1.2 references zero images, so no rendering task applies.

## Original authority inspection

- scikit-learn official Cross-validation guide, downloaded page identifying itself as 1.9.1. Personally read introductory paragraphs (plain text lines 9–20 and 154–177): testing on learned examples can reward label repetition; validation is used when choosing hyperparameters; final evaluation remains on a held-out test set. Personally read grouped-data sections 3.1.2.1 and 3.1.2.4 (text lines 861–878, 1653–1680): dependent samples must be grouped, with a source group absent from both paired training and validation folds. This supports the method, not any accuracy of this toy model.
- Lee et al., Deduplicating Training Data Makes Language Models Better, original ACL 2022 publisher PDF, pages 8424–8445. Personally read abstract/introduction pp. 8424–8425, exact-substring method §4.1 p. 8426, approximate document method §4.2 p. 8427, cross-split treatment §5 p. 8428 and Train/Test Set Leakage §5.3 p. 8429. §5.3 expressly says duplicate validation examples can inflate metrics for models better at memorizing train sets. The paper addresses sufficiently long repeated substrings and near-duplicate documents; its experimental 50-token threshold is not a claimed threshold for the course's three-character, same-source illustration. No paper metric is imported as a result of the course.
- Python official 3.13 documentation, fetched pages identifying documentation patch version 3.13.16. Personally read random.seed, random.shuffle and Notes on Reproducibility: shuffle operates in place, fixed seeds reproduce runs under the same environment, while most algorithms can change between Python versions. Read Built-in Types entries dict.fromkeys, dictionary insertion order, dict.items/views, set.intersection / &, Truth Value Testing, and Boolean not. Installed Python is 3.13.5, explicitly not the documentation patch release. The APIs under review agree with executed behavior; no cross-version shuffle guarantee is claimed.
- Official PyTorch tutorial source pinned at pytorch/tutorials commit 4d66ab3f9226b7e712b725f9cf0f03fe5b0f1531. Personally read source lines 15–19 and 129–142, plus the train-loop implementation lines 153–176: training optimizes model parameters; optimizer.step adjusts parameters. Used only for the training-parameter concept, not its mixed validation/test wording. Documentation HTML endpoint returned 403, so the author's official repository source was fetched at the file's last-change commit obtained from GitHub. The full tutorial was not run: it would download FashionMNIST and train. The section's code itself does neither.

All fetched original URLs, sizes, hashes and access date are in download-receipt.json. Original HTML/PDF/source snapshots are retained. HTML-to-text/PDF-to-text copies are convenience locators, never substitutes for source provenance.

## Independent numeric derivations and executed results

For n=4 unique documents: initial a=max(1,int(4*0.8))=3, b=max(2,int(4*0.9))=3. Then b=min(3,3)=3, a=min(3,2)=2. The slices docs[:2], docs[2:3], docs[3:] have lengths 2,1,1. Expected counts were derived from the real helper before checking output.

Original fence execution: exit 0, no helper guard events. Exact stdout:

    train 2 ['鳥看魚。', '狗看貓。']
    validation 1 ['魚看鳥。']
    test 1 ['貓看狗。']

Independent bounded probe, exit 0: all three pairwise intersections are empty; repeated seed with the same input reproduces the split; original input is unmodified; adding another exact duplicate leaves the total 4 and the split unchanged; adding 鳥看貓。 increases unique total to 5 (helper yields 3,1,1). Checked n=3..20 distinct dummy strings with added duplicates for nonempty pairwise disjoint groups. These checks establish the helper's behavior, not language-model generalization.

The source 貓看狗。狗看貓。 has first two length-four sliding windows 貓看狗。 and 看狗。狗. The suffix/prefix shared by those adjacent windows is 看狗。, exactly three characters. They originate in the same source; if put into training and held-out partitions, the supposedly new window contains content already available in training. This is an illustration of provenance leakage, with no quantified score effect.

## Explicit limits checked rather than assumed

The helper has no source-family identifier or approximate matching. A bounded witness with seed=0 and [貓看狗。, 貓看狗！, 狗看貓。, 鳥看魚。] puts 貓看狗。 in train and 貓看狗！ in validation. The section's original train/validation-only assertion also passes an artificial split whose train and test both contain 貓看狗。. These are limitations already stated in current text lines 74 and 83; they are not evidence of a contradictory course promise. The section's actual 2/1/1 sample is exact-string disjoint, and its raw fence and helper contain no model, gradient, optimizer or training operation. “New material” is supported only conditional on later respecting the source-group separation and training only on the assigned train group. This review does not certify source independence of arbitrary input corpora or unseen pretraining data.

Shared vocabulary is expected to be reusable. The executed example has train 狗看貓。 and test 貓看狗。, which share every character while remaining unequal documents. The authorities' leakage criteria refer to dependent source groups, duplicate documents and substantial matching content, not a prohibition on sharing individual characters. The current caution about same-source copies/high-overlap content is consistent with that scope.

No unresolved substantive claim found. No model training, GPU work, training-data/model download, upload, paid job, textbook edit, figure edit or commit performed.
