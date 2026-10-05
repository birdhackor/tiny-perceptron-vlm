# Independent bounded calculation

An answer is correct only if its payload equals the truth for the current input.
Consider one sound question whose correct pitch is low. A procedure that always
answers high, without examining the sound, gets 0/1. Replace only the waveform by
a high sound and update the truth to high. The same constant procedure now gets
1/1. Its score changed, but its output never depended on the sound. Therefore a
score change alone is insufficient evidence that a model recognized the changed
sound correctly. A paired test must check each current answer against its current
truth while holding the image and question fixed.

The frozen pure-audio test has three low and three high truths: the constant high
payload is correct on exactly 3 of 6 rows, or 0.5. The image-color test has nine
green truths and the image-shape test has nine square truths: their respective
constant payloads are each correct on 9 of 9 rows, or 1.0. These are arithmetic
baselines with denominators in questions, not measured model predictions.

Exact overlap checks test equality of the chosen input representation. Empty
overlap says nothing about shared templates or semantics. In this dataset the
same two calculator question templates occur in all three splits despite empty
family/input intersections. The test therefore cannot establish understanding
of previously unseen Chinese phrasing.

All calculations are exact integer counts. No learned parameters, model
inference, model scores, or training were used for this review.
