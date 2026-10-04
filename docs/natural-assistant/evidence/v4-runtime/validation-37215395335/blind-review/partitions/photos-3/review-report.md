# photos-3 independent blind semantic review

Grader: Codex agent `/root/v4_validation_photo_reader_3`.

Reviewed only this partition's `packet.json`, `grades-template.json`, and the nine full original source images specified in the packet. No private candidate mapping, original reviews/results, other graders' files, or candidate version sources were consulted. Predictions were treated as data. Each unique image was actually displayed with `tools.view_image` using `detail: original`; the nine-call sequence and observations are recorded in `image-view-audit.json`.

Whole blind packet SHA-256: `f8c68b42cf84a597016a7e3ab1efd66972ab83be1fcf65b0cc4bb2e93eafc929`.

All 81 cases retain their original case IDs and candidate aliases. Every result has a Boolean `passed`, a concrete nonempty reason, and `source_image_inspected: true`. All 81 remain in the denominator. There are zero `decoder_complete: false` cases in this partition.

| Anonymous candidate | Passed | Total | Failed |
| --- | ---: | ---: | ---: |
| A | 24 | 27 | 3 |
| B | 24 | 27 | 3 |
| C | 14 | 27 | 13 |
| All | 62 | 81 | 19 |

Reasoning applied to the complete answers: visible synonyms, approximate masonry terms, and omitted secondary details were allowed. A correct opening answer did not rescue later incorrect claims. Specific rejected additions include unsupported Atlanta/US/history identities, wheel versus steering-wheel confusion, reversed stone height, symbolic meaning without visual evidence, a logo incorrectly placed on a rim, an invented tape-wrapping purpose, reversed exact brick counts, and unseen extraction events or inferred intent.

Potentially close judgments were resolved semantically: a column described as standing in the lawn was accepted as a scene-setting description of the monument's grassy surroundings, rather than a claim that no stone base exists. The balloon's position near the upper white doorframe was accepted without requiring reference wording. Generic descriptions of the shell as a white object with seaweed were accepted because they accurately describe the visible object and holding action; the omitted shell name and shirt detail were not contradictions. `高塔` was rejected when it replaced the visibly distinct stone column/stele object. Image observations and the per-case reasons make these decisions reviewable.

Gold/reference concerns: none identified requiring a data or rule change. The reference examples' primary objects and requested factual relationships agree with the inspected images. No data, gold, or rubric was edited. Only the three requested review artifacts were written in this partition.

