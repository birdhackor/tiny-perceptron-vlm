# Independent hand calculation for lesson 17.14

For exact decimal input x=(0.1,0.4,0.9), maxabs=0.9, signed symmetric four-bit maximum positive code is 2^(4-1)-1=7. s=0.9/7=0.12857142857142857. Normalized x/s=(7/9,28/9,7)=(0.777777...,3.111111...,7); round-to-nearest gives q=(1,3,7), already in [-7,7]. q*s=(0.12857142857142857,0.3857142857142857,0.9), which to four decimals is (0.1286,0.3857,0.9000). These are hand values, not claims that FP32 tensor arithmetic equals exact decimals.

The chosen surrogate is y=x+stopgrad(restored-x). Its reverse-mode Jacobian is I: the stopped term contributes zero, leaving the outer x. L=sum(y) has dL/dy=(1,1,1), hence the STE dL/dx=(1,1,1). For L=sum(y^2), dL/dy=2y and the STE gradient is (0.25714285714285714,0.7714285714285714,1.8), to four decimals (0.2571,0.7714,1.8000). This is the specified surrogate gradient, not the mathematical derivative of round or of discrete quantization. For ordinary round, the function is locally constant away from midpoint jumps, so its derivative is zero there and undefined at jumps; PyTorch's chosen round backward returns zero even at jumps.

This example's independent hard input fixes s=0.9/7, so the standard chain rule multiplies any downstream loss derivative by zero from round. Its sum loss thus yields (0,0,0). Changing the first loss only leaves the hard branch's sum loss unchanged.

Formal reported test NLL arithmetic: QAT packed 41.25349044799805/69=0.597876673159392; matched PTQ 28.054508209228516/69=0.40658707549606543. Subtraction is 0.19128959766332654 -> 0.1913. Raw exact counts 4/10 versus 6/10 differ by two questions (fraction -0.2). Validation QAT 18.98137855529785/36=0.5272605154249403 and matched PTQ 42.51936721801758/36=1.1810935338338215. Both raw generation exact counts are 2/5.

The model tensor accounting is 102912 bytes of retained FP32 parameters + 65824 bytes of packed values/scales/FP32 biases = 168736 bytes. Formal private deployment file: 183045=168736+14309 bytes of serialization/metadata overhead. Published sanitized QAT deployment file: 181381=168736+12645; published matched PTQ file:184341=168736+15605. The same tensor storage need not have equal serialized file length.

Review scope: hand calculation validates these inputs and these quoted report values. Actual FP32 values, gradient vectors, all effective target counts, and all per-question ID comparisons are independently executed in fact_v2_17_14_audit.py and its saved execution receipt.
