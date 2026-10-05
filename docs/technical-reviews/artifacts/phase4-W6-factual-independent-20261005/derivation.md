# W.6 independent derivation

Reviewed by `/root/phase4_factual_coordinator/factual_w_6`, 2026-10-05.

The example uses abstract knob-position units, not a calibrated temperature law.
The vertical quantity is squared error in the corresponding squared abstract units.
Let L(w)=(w-3)^2=w^2-6w+9. Then L'(w)=2w-6.

For nonzero h, [L(1+h)-L(1)]/h = [(-2+h)^2-4]/h = -4+h.
As h approaches zero from either side the difference quotient approaches -4.
At h=0.1 the quotient is -3.9, L(1.1)=3.61, and the decrease is 0.39.
The derivative at w=1 is exactly -4, while at w=1.4 it is exactly -3.2.
The sign says increasing w locally decreases L, and |L'| scales a small input change.

One update from w=1 using rate alpha gives w_new=1-alpha*(-4)=1+4*alpha.
L(w_new)=(-2+4*alpha)^2=4(1-2*alpha)^2.

| alpha | new w | new L | derivative at new w |
|---|---|---|---|
| 0.1 | 1.4 | 2.56 | -3.2 |
| 0.6 | 3.4 | 0.16 | 0.8 |
| 1.1 | 5.4 | 5.76 | 4.8 |

Thus the 0.6 step overshoots the minimum but improves the loss; the 1.1 step overshoots and worsens it. For this starting point and quadratic alone the loss strictly decreases for 0<alpha<1. No general optimizer threshold follows from that example.

The squared error is nonnegative and reaches its unique minimum zero at w=3.
The figure's coordinate transform is x=75+123*w and y=430-34*L. The blue current point is (198,294), orange update is (247.2,342.96), and blue minimum is (444,430). SVG y increases downward, so a right-down arrow represents increased w and decreased L. Its arrow joins the displayed results; it is not a tangent. The 161 rounded curve vertices match this transform within 0.03 pixel (observed maximum 0.00933542203711113).

Exact Fraction calculations in `verify_w6.py` independently check the arithmetic. Executed float32 comparisons tolerate absolute error below 1e-6; this is appropriate for the demonstrated single-step numbers, not a promise for arbitrary floating-point calculations.
