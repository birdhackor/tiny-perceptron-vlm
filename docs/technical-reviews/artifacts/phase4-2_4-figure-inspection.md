# Actual figure inspection for 2.4

Reviewer: /root/phase4_factual_coordinator/factual_2_4, 2026-10-05.

I read the original SVG (course/figures/rewrite-02-nonlinearity.svg), exported it to phase4-2_4-nonlinearity.png with Inkscape 1.4 (e7c3feb100, 2024-10-09), and actually opened the PNG using view_image. The final saved render was viewed again after the archival command. Dimensions: 640 by 990 pixels.

The title says the same weights differ only by the presence of an intermediate ReLU. The upper panel has input labels -2, 0, 2, output labels all 0, an output axis pointing upward, an input axis pointing right, and three points on the horizontal zero line. The lower panel has the same input labels, corresponding output labels 2, 0, 2, and the V-shaped segment descends to the zero point then rises. The output-axis labels 0 and 2 and the input labels identify the numeric values; the claimed slopes are in these numerical coordinates, not in uncalibrated image pixels. The bottom caption says these points come from the hand-filled formulas and are not training results. All text is readable in the inspected render, and the displayed points, axes, captions and line directions match the raw SVG, current table, forward outputs and my algebra.

Renderer limitation: Inkscape emitted PangoFT2FontMap and GtkRecentManager wrapping warnings, preserved in phase4-2_4-render-stderr.txt, but exited 0 and produced the image viewed above. This was an actual SVG-to-PNG review; no Chromium/browser rendering claim is made, and no browser timeout was observed in this task. There was no offline-helper guard event during original execution.
