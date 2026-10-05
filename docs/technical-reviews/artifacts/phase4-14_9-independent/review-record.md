# 14.9 independent factual review: actual scope and support

Reviewer task: `/root/phase4_factual_coordinator/factual_14_9`. Date: 2026-10-05.

## Inputs actually read

Read the complete original 14.9 section (chapter source lines 307–333), its only referenced SVG, and the necessary original sections 14.1 and 14.8. Read the factual-reviewer instructions, checker schema, section_facts helper, clear-tutorial SKILL.md and its review protocol. Did not read old technical/reader reports, course source notes, author review notes, or revision summaries. The locator JSONs were read only for top-level keys/types and primary-paper locator metadata. No required paper was present there.

The complete raw chapter is preserved as the initial frozen input in `chapter14-frozen-input.md`; its SHA is 292b95b0fdd08459dde76bc6c237da58129852b35f7aabd8bd5f6ee7f88b37a3. This is a historical input snapshot, not an assertion about a later whole-chapter version or about reviewing all sections. The introduction was not reviewed. The section source and the figure were separately frozen from the extraction's original bytes.

There are zero original Python or shell fences in 14.9. There are no reported model measurements, sample denominators, training runs, or real-model configuration numbers to verify in this section. The 60° and 15° rates are explicitly artificial examples. No training, GPU work, dataset/model download, full-model evaluation, or engineering setup was performed.

## Primary sources personally checked

RoFormer was acquired from https://arxiv.org/pdf/2104.09864v5 with TLS verification. The first-page stamp states arXiv:2104.09864v5, 8 Nov 2023, with a November 9, 2023 PDF header. Personally read §3.2.1–3.2.2 on original pp.4–5, Eqs.12–16: rotate Q/K after their projections; a 2D block uses cos(mθ) and sin(mθ); the even-dimensional space has d/2 paired subspaces; Θ gives different predetermined rates; orthogonal rotation preserves vector length; the Q/K inner product uses n−m. This supports the two-coordinate arrow explanation and frequency-dependent phase differences. It does not establish a language model's quality after modifying frequencies.

YaRN was acquired from https://arxiv.org/pdf/2309.00071v3 with TLS verification. The first-page stamp states arXiv:2309.00071v3, 6 Feb 2026. Personally read §2.1–2.3 (original pp.2–3) and §3.1–3.2 (pp.4–5). Eq.8 defines wavelength λ=2π/θ, the token distance for one full turn. §3.1 discusses uniform scaling reducing high frequencies, with increasing extension causing greater distortion. §3.2 explains how altering short wavelengths affects nearby relative distances and motivates frequency-dependent scaling: leave short wavelengths largely unscaled, interpolate long wavelengths, blend between them. The section's prose states a design direction and a possible need to adapt weights; it reports no YaRN capability result and no universal guarantee.

Source limitation: the v3 PDF actually prints `g(m)=s·m` in §2.3 Eq.9 while defining `s=L′/L`. I rendered and viewed original p.3 to rule out text-extraction ambiguity. I do not use Eq.9 as evidence for coordinate compression. The explicit operations in this section, division by 2 and 4, are verified independently from RoFormer's phase `mθ`; YaRN is cited for the tradeoff and the frequency-dependent design only. This paper-printing discrepancy does not change those cited paragraphs or the section's arithmetic.

## Substantive coverage and independent checks

1. A four-coordinate feature can be represented as two 2D arrows, with position-dependent rotations of the Q/K feature pairs. Confirmed against RoFormer Eqs.12–16; the metaphor is not a claim about clocks inside cards.
2. Different pair rates imply different periods. For the hand-selected examples, 360/60=6 positions and 360/15=24 positions. The unit is degrees per position, not degrees per second.
3. With Δm=1, replacing m by m/2 gives Δangle=θ/2 for every pair: 30° and 7.5°. With m/4 it gives 15° and 3.75°. CPU stdlib checks also translate both positions from 0/1 to 10/11 and preserve the same gap-dependent dot product; angle arithmetic is exact here and trig uses absolute tolerance 1e-12.
4. Smaller phase gaps do not delete tokens or force different content vectors to become identical. Rotation is orthogonal; a bounded variant verifies that two distinct content vectors retain their squared distance 2 under a common rotation. This is a mechanism check, not an assertion that a model successfully distinguishes every pair of positions.
5. Uniform compression changes nearby relations as well as distant ones; a trained network can need adaptation. Confirmed as a conditional mechanism/tradeoff against YaRN §3.1–3.2. No measured failure rate or guaranteed training requirement is inferred from the toy calculation.
6. A single rate is periodic and cannot uniquely encode every arbitrarily long distance. A variant at position 6 returns the fast 60° pair to [1,0] while the 15° pair is [0,1]. Different rates and content can contribute other information, without a claim that two selected rates remove all aliasing. In fact, this artificial pair of rates jointly repeats after 24 positions. The text makes no universal uniqueness promise.
7. The original SVG's four orange arrows, in its read order, encode 60°, 30°, 15°, and 7.5°. The screen y-axis is downward, so geometric angle is computed using y1−y2. All errors are less than 0.0001°; lengths are 64 px within 0.0001 px. Blue dashed baselines point right from the same centers. Labels, row/column order, and the 0.5 gap match the section.

## Real visual checks and failures

Inkscape 1.4 actually rendered the original SVG to `figure-render.png`; I viewed the resulting image. It shows all four clocks, dashed baselines, orange hands, angle labels and the footer without clipping. The short 7.5° hand separation is small as expected and its explicit number is visible. Inkscape emitted Pango/GTK wrapping warnings; it still returned 0 and produced the actual image. The image was viewed through view_image.

Chromium 151 attempts were also made for a standalone responsive figure wrapper at 1280×800 and 390×844. They did not produce either requested screenshot. The initial desktop process was sent TERM; it exited while the first script then continued to its mobile call. The initial script and logs captured at that point are retained under `*.initial.*`. A second script used a separate profile, disabled background networking and dev-shm usage, and requested a 30-second timeout; it exited 124 without a screenshot. The first remaining mobile process group was terminated explicitly, yielding exit 137 for the initial script. The cleanup command reported that the already-finished second process group no longer existed. The combined `render.stderr.txt` honestly retains interleaved old/new Chromium output, including NUL gaps from two writers; it was not cleaned to look successful. `render.initial.stderr.txt` preserves the first capture before this overlap.

The two failed view_image requests for nonexistent desktop/mobile files are retained in the rendering event record. No browser page or mobile-layout success is claimed. The completed visual scope is the original diagram's actual Inkscape rendering, which satisfies the diagram's factual consistency check. The standalone wrapper was only a figure presentation probe, not a built course-site screenshot.

## Outcome

No unresolved substantive factual issue was found in 14.9. The section appropriately limits the example to hand-picked rates, keeps content and coordinate compression distinct, states the local-distance tradeoff conditionally, and introduces frequency-dependent scaling without claiming model ability. The report's pass applies only to 14.9 and its referenced diagram, with the visual and source limitations above.
