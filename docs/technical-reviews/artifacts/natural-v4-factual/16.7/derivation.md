# Independent derivations for 16.7

FP16: one sign bit, five exponent bits, ten fraction bits; normal exponent maximum 15. Maximum finite (2 - 2^-10) * 2^15 = 65504. BF16: one sign bit, eight exponent bits, seven fraction bits; maximum normal exponent 127. Maximum finite (2 - 2^-7) * 2^127 = 3.3895313892515355e38, compared with FP32 (2 - 2^-23) * 2^127 = 3.4028234663852886e38. Fraction spacing near 1 is 2^-10 versus 2^-7, while FP32 uses 2^-23. IEEE bytes are eight bits: 32/8 = 4; 16/8 = 2. The range statement concerns exponent range and approximate largest normal values; hardware may handle subnormals differently.

Exact decimal multiplication before floating rounding:
C11 = 1.001*0.5 + 2.002*1.5 = 0.5005 + 3.003 = 3.5035.
C12 = 1.001*1 + 2.002*(-1) = -1.001.
C21 = 3.003*0.5 + 4.004*1.5 = 1.5015 + 6.006 = 7.5075.
C22 = 3.003 - 4.004 = -1.001.
BF16 rounds this a to [[1,2],[3,4]], while b is exactly representable. Its multiplication gives [[3.5,-1],[7.5,-1]]. Absolute decimal differences [[0.0035,0.001],[0.0075,0.001]], maximum 0.0075. Converting these already rounded numbers to FP32 adds precision capacity but cannot recover the original values. The CPU execution additionally records exact FP32 errors.

For a multiplied by 10, ideal decimal product is [[35.035,-10.01],[75.075,-10.01]]. BF16 input rounds to [[10,20],[30,40]], product [[35,-10],[75,-10]], maximum decimal error 0.075. Actual CPU maximum 0.07500457763671875. For the all-integer input, both products are [[3.5,-1],[7.5,-1]] and CPU error is exactly zero. Neither result is a general relative-error guarantee for arbitrary inputs.

For constant scale s, d(s*L)/dw = s*dL/dw by linearity and chain rule. With L=(w-3)^2, w=1, s=1024: L=4, dL/dw=-4, scaled loss=4096, scaled gradient=-4096, unscaled gradient=-4. SGD learning rate 0.1 still gives w=1 - 0.1*(-4)=1.4; scaling has not increased the effective learning rate. Scale is a numerical-management tool, not proof of equal training trajectories after rounding.

141568 FP32 scalar parameters * 4 bytes = 566272 bytes. Memory MiB equals bytes / 1048576. Each displayed start/peak/additional entry must be rounded independently: for BF16, 68549120/1048576=65.37353515625; 79568384/1048576=75.88232421875; their byte difference 11019264/1048576=10.5087890625. These independently round to 65.374, 75.882, 10.509; subtracting displayed rounded cells need not yield the displayed delta. Baseline offsets are BF16-FP32 32.28759765625 MiB and FP16-FP32 32.146484375 MiB. Excluding reserved unused and driver/context allocations follows PyTorch CUDA allocator API scope.

The actual L4 record audit recomputes the loss denominator from each answer UTF-8 length plus one EOS: validation 36, test 69; raw ID equality counts 2/5 and 7/10 for FP32, 1/5 and 6/10 for BF16, 1/5 and 4/10 for FP16. Seed-42 sampling replay gives 22493 supervised training targets in 200 attempts of 16 records. Forward medians are recomputed from all nine saved observations. Training raw per-step times are absent from the public record, so their aggregate is checked against the original code scope rather than independently recomputed.
