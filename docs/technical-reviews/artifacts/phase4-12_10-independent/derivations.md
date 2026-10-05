# Personal arithmetic and interpretation, 12.10

For the stipulated rule y(f)=high iff f>300 Hz, otherwise low:
- [200,220,440,660] Hz maps to [low,low,high,high]. Constant high matches 2/4=0.5.
- f=300 Hz is low (strict inequality); f=310 Hz is high. No continuous-frequency identification performance is claimed.
- The recorded test set is 7 frequencies × 2 amplitude/duration settings =14 cases, with 8 low and 6 high. Constant low is 8/14=4/7; an exercise with 9 low out of 10 gives 9/10=0.9.
- Old-target consistency compares g(x') with y(x); new-input accuracy compares g(x') with y(x'). These can disagree whenever y(x') differs from y(x).
- The recorded donor is (i+7)%14. Rows 0 and 7 have old/new low labels. The other 12 old/new labels differ. Saved answers give 3/14 old-target matches and 11/14 donor-target matches, both independently recomputed from raw IDs and donors.
- A zero waveform has no identifiable single tone carrier: for every frequency f, amplitude 0 gives 0*sin(2*pi*f*t)=0. Its comparison with original tone labels is an intervention control, not a frequency classification truth label for silence.
- Raw result recomputation verifies historical bytes and counting, not the recorded GPU model or its training. Scores concern synthetic sinusoidal audio and this particular fitted model on 14 cases; they do not establish real-person speech intent recognition or statistical certainty.
