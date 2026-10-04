# Actual figure inspection

19.10 contains no direct SVG. I rendered and personally viewed the five relevant prerequisite SVGs using `tools.view_image` after the actual Inkscape rendering recorded in `render-receipts.json`. Source and rendered PNG SHA-256 values are retained there. Inkscape returned zero for each render; its harmless Pango/Gtk initialization warnings are preserved in the receipt.

- `architecture_quantize_grid.svg` (17.2): Left-to-right arrows mean divide by scale, round, multiply by scale. With scale 0.5, the top row is 0.7→1.4→1→0.5 and bottom row 1.2→2.4→2→1.0. Both reconstruction errors are 0.2; the right edge does not falsely match the original value.
- `architecture_int4_packing.svg` (17.8): First-code dashed green arrows enter the right/low nibble; second-code orange arrows enter the left/high nibble. The displayed −8,−1→0,7 produce 01110000=112; 0,1→8,9 produce 10011000=152. This matches the actual +8 storage convention, not the quantizer's usable −7…7 range.
- `capstone_resources.svg` (19.2): Four experts remain stored; a token's arrows select experts 1 and 3. Shared embedding, attention, router and output rules are drawn separately. The selected pair is an example, and the next token may use another pair; the figure claims neither fixed modality expertise nor guaranteed speed/memory savings.
- `capstone_pipeline.svg` (19.4): Arrows go pretrain A→SFT B→joint C; a dashed branch goes C→DPO D. Joint is labeled the recommended branch, and students/quantization are separate deployment choices. This matches the independently audited joint/DPO teacher SHA lineage.
- `capstone_tool_loop.svg` (19.7): Downward arrows distinguish first model request TOOL:calculator:1+2, runtime checks and actual result 3, then the same model's second generation DIRECT:3. Its annotations require EOS, exact parameters and final answer, and prevent replacing model output with the runtime result. This matches the raw-record audit and the CE/KD 1+8 failure interpretation.

The figures support the section's necessary prerequisite concepts and its scoped comparisons. None depicts a GPU training result or a general speed guarantee.
