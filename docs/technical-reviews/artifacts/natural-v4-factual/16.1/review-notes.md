Independent factual review of course/chapters/16.md#16.1

Reviewer: /root/v4_review_coordinator/factual_v4_16_1, fresh task.
The complete first-section introduction and complete raw section were read and
saved before probing. Their SHA-256 values match the dispatch. The previous
assigned report was copied without opening or interpreting it.

Read prerequisites: 4.5 (model block), 15.11 (operation counts versus time),
4.2/4.3/4.4 (residual, normalization, positionwise FFN), 15.6/15.10
(dispatch/combine and stored versus active parameters), and complete T.8
(the original measurement scopes). This is dependency inspection, not a
separate pass verdict on every factual claim in those prerequisites.

Figures personally viewed after rendering with installed Inkscape:

- foundations_residual.svg: x=[1,2] splits into an upper 2x branch [2,4]
  and a lower preserved [1,2] branch. Both arrows meet the plus node and
  the output arrow reaches [3,6]. Bottom sensitivity 1+2=3 agrees with
  the derivative of x+2x. The positions and arrows show a sum, not overwrite.
- architecture_dispatch_example.svg: upper and lower left contributions
  labelled row 2, [1,1] and [3,3], both point to bottom right row 2 [4,4].
  Middle left row 0 [2,2] points to top right row 0 [2,2]. Middle right
  row 1 [0,0] has no incoming arrow. Caption explicitly calls missing row 1
  work deliberate. Crossing arrows preserve row identity, not completion order.

Actual render commands:
inkscape course/figures/foundations_residual.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/natural-v4-factual/16.1/prerequisite-residual.png
inkscape course/figures/architecture_dispatch_example.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/natural-v4-factual/16.1/prerequisite-dispatch.png
Both returned exit 0 and produced PNGs. Inkscape emitted Pango/Gtk wrapping
warnings; the personally viewed PNGs rendered the Chinese labels and arrows.
There is no SVG in the assigned section or introduction. The report's figure
map nevertheless includes the two necessary prerequisite figures actually viewed.

Parameter derivation (default width 8, one block/head, max_length 128,
vocab 264, separate output weight, GELU FFN hidden width 32):
embedding 264*8=2112; position 128*8=1024;
three LayerNorm affine pairs 3*(8+8)=48;
Q/K/V/output attention projections 4*(8*8)=256;
FFN up 8*32+32=288, down 32*8+8=264;
language output 8*264=2112.
Sum 2112+1024+48+256+288+264+2112=6104.
FP32 parameter payload 6104*4=24416 bytes. Inputs of length 3 or 12 do
not resize any parameter table; the logits' position axis changes accordingly.
These are payload bytes, not object size, allocation peak, or model RAM usage.

Median derivation: for sorted nine values the central rank is 5 (zero-based
index 4); [1,2,3,20] has median (2+3)/2=2.5, but arithmetic mean 26/4=6.5.
Binary MiB uses 2^20=1048576 bytes. No timing value is mathematically fixed by
these identities.

CPU probe actually executed the lesson's complete code block (seed 42 only
added before initialization for a reproducible fixture), then profiled the
same model at lengths 3 and 12 for three rounds each. Each measurement has
three unrecorded warmups and five recorded forwards, CPU activities only,
one thread, FP32, eval and no_grad. Calls per five forwards: index_select 10,
GELU 5, _softmax 5, linear 35. Separate diagnostic module annotations place
the two index_select calls under embedding/position, GELU under FFN,
_softmax under attention, and matrix operations under multiple modules.
For every nested CPU event the observed self time equals total minus direct
children to absolute tolerance 1e-9 microseconds. Adding total rows duplicates
nested durations. The full averages' total/self sums demonstrate that overlap.

In the three rounds, _softmax self times for the five calls were
9.977/10.557/11.525 microseconds at length 3 and
23.427/21.857/24.047 microseconds at length 12. This identifies a longer-input
work item worth inspecting in this run; it proves neither a general bottleneck
nor a machine-independent timing ratio. Other operator times fluctuated.

Separate interleaved timing executed nine samples per mode, thirty identical
forwards per sample, three warmups per mode. The perf_counter interval excludes
profiler initialization/shutdown/export. Plain median was 0.2607985666 ms per
forward, versus 0.4765443333 ms with CPU profiling active. This establishes
observed instrumentation overhead here, not a fixed slowdown across environments.

Original-source inspection: PyTorch 2.14.1 official documentation-source files
and implementation were retrieved from the v2.14.1 tag on its official GitHub
repository. Public docs.pytorch.org URLs and NIST page returned 403. Original
source retrievals succeeded; the report cites those genuinely read files and
exact APIs/line locators. Full third-party raw files remain in ignored research;
the durable authority retrieval file preserves original URLs, hashes and outcomes.
The empty APIs have unspecified useful contents in the default state used here;
the deterministic fill override documented by torch.empty/empty_like is an
explicit scope limit, and empty payload values were intentionally not read.
View claims assume valid shapes/strides; as_strided bounds/overlap restrictions
and view contiguity conditions are not erased by the small alias probe.

Original GPU records: efficiency and precision use revision
48a4f3e912b483d70aee57c42c2aac226534a9a6; Flash uses revision
382604d17d91cfe9e0a58a7de997486e3a0ccafa. Reading architecture.py from
those existing Git objects produces the exact code_sha256 in each saved run.
This is read-only source inspection; no Git state was changed.
The first revision's _benchmark at lines 494-510 performs three warmups,
an initial synchronization and a synchronization after each measured call,
then median(samples). The ending barrier for one call is the starting barrier
for the next because no GPU work is queued between them. The Flash revision's
_flash_measure at lines 1176-1226 additionally synchronizes before each sample.
Both revisions' _train synchronize on each side of the update interval and
take median(latencies[3:] or latencies); actual reported schedules exceed
three updates. The update interval includes input preparation/transfers,
forward, loss, backward, finite-gradient checks, clipping and optimizer step;
it is different from no_grad forward/decode/component measurement.

The CPU audit recomputed medians of all 21 timing groups with saved raw
samples: efficiency 10, Flash 8, precision 3. Each has 3 warmups, 9 measured
calls and synchronized=true; every stored median equals sorted(samples)[4]
within 1e-12 seconds. Per-update training latencies are not saved in those
public exports, so their numerical medians were not recomputed. The original
run implementation supports the claimed omission of the first three updates;
no GPU timing or training was replicated by this reviewer.

Memory authority: PyTorch CUDA memory_allocated/max_memory_allocated count
tensor allocations, while memory_reserved includes the allocator's retained
unused memory. The original branch implementations reset the allocated-byte
peak and record its starting allocation. Adam's original implementation has
gradient references and lazily allocates exp_avg/exp_avg_sq tensors beyond
the parameters. Together with TinyLM's returned K/V state and activations,
this supports the section's warning that parameter payload is not a peak.

Limits: CPU only, no GPU training, no full benchmark rerun, no model/data
downloads, no environment/source/checker/Git edits, no subagents. Three
warmups are a documented measurement choice, not a proof that every possible
initialization/JIT or thermal effect has disappeared. Performance observations
are scoped to these sizes, APIs, execution modes and sampled machine state.
