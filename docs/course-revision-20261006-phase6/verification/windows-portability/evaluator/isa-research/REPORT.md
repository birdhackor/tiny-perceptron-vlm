# Native Windows illegal-instruction investigation

The grounded CI mitigation is `ONEDNN_MAX_CPU_ISA=AVX2` on the Windows pytest step, set before Python starts. This restricts oneDNN's CPU dispatcher below AMX while keeping the existing BF16 operators and all test assertions. The official documentation and the exact installed backend source support this setting; the disposable precision fixture passes with it on Linux. No CI, application, model, or test file was changed during this investigation.

## Actual repository and job evidence

Job `112332616411` terminates with `0xc000001d` in `torch.nn.Linear.forward`, called by `DenseFFN.forward` at `tiny_perceptron/modern.py:46`, `_forward` at `scripts/course_experiments/architecture.py:131`, and `_nll` during `_train`'s initial evaluation. The test is the precision parameter of `test_public_entries_honor_step_scale_and_return_partial_status`; seven earlier architecture-completion cases passed before the fatal process exit. `_nll` runs before the deadline check, so the fixture's `deadline=0.0` still performs its real initial forward pass.

The earlier job `112325891684` passed all eight architecture-completion cases. The two jobs have identical Windows image `windows-2025-vs2026` / `20260925.250.1`, Azure region `centralus`, Python `3.13.15`, pytest `9.1.1`, and Torch `2.14.1+cpu`; they have different worker IDs. Git bytes for the test, experiment implementation, model implementation, FFN implementation, and `uv.lock` are identical between `77d0589887847777d85aebdbc5b74351f12237b0` and `e03c617ace040b664f4cb722fee622ab034168b8`. Their digests and selected original log lines are saved in [comparison.json](comparison.json).

Neither native job log contains the CPU model, oneDNN's selected ISA, the failing native instruction, or the active autocast dtype. Therefore, the precise hardware defect is not proven for this particular worker. The documented Windows hosted-runner AMX defect below is a strong match, supported by unchanged relevant source/dependencies and hardware-dependent failure.

## Authoritative upstream diagnosis

The [oneDNN issue 5689](https://github.com/uxlfoundation/oneDNN/issues/5689), **Illegal instruction on EMR + Windows**, reports BF16/AMX failures on Xeon Platinum 8573C with Windows Server 2025. The investigator's [root-cause comment](https://github.com/uxlfoundation/oneDNN/issues/5689#issuecomment-5136593460) states:

> After much internal investigation, root cause is github runner `windows-2025` for 5th gen Xeon EMR platform, advertises inconsistent CPUID AMX capabilities.

> eventually calling `TILEZERO tmm0` which raises #UD.

The same comment says EMR instances sometimes report AMX absent consistently, which avoids the issue. [Runner-images issue 14483](https://github.com/actions/runner-images/issues/14483) records AMX feature bits in CPUID.7 and XCR0 while CPUID.1D reports zero tile palettes. This explains how different worker instances using the same OS image can behave differently. It is evidence for the failure class, not a CPU identification for our uninstrumented worker.

[oneDNN PR 5801](https://github.com/uxlfoundation/oneDNN/pull/5801), **cpu: x64: disable AMX when no tile palette is supported**, explicitly states:

> Some misconfigured hypervisors/VMs advertise AMX support via CPUID (AMX-TILE bit) and XCR0[18:17], so mayiuse(amx_tile) returns true, yet report zero supported tile palettes in CPUID.1D:EAX. Executing any AMX instruction (e.g. tilezero/ldtilecfg) on such a system raises #UD (illegal instruction).

The saved PR diff adds `&& x64::amx::get_max_palette() != 0` to AMX selection. The API response reports this PR closed with `merged_at: null`; this report does not claim the proposal shipped. Raw issue, comment, PR, and diff bytes are retained locally.

The installed Linux Torch `2.14.1+cpu` build identifies oneDNN `3.12.0`, commit `80afa71049cd69a3df32adcccb623b12cd7baa22`. Its exact [Windows AMX state check](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/cpu/x64/cpu_isa_traits.cpp#L374) tests OSXSAVE and XCR0[18:17]. Its [AMX selection](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/cpu/x64/cpu_isa_traits.hpp#L466) checks the AMX-TILE feature and AMX state, without the proposed nonzero-palette check. The original Windows job did not print its backend build configuration, so matching the Windows backend commit needs a native diagnostic.

## Exact supported environment setting

The official [CPU Dispatcher Control documentation](https://github.com/uxlfoundation/oneDNN/blob/8b6108adf17c1af67ef07a43577a26413df02747/doc/performance/dispatcher_control.md#L21) says:

> When the feature is enabled at build-time, the `ONEDNN_MAX_CPU_ISA` environment variable can be used to limit processor features oneDNN is able to detect to certain Instruction Set Architecture (ISA) and older instruction sets.

Its environment-variable table at line 32 explicitly lists:

| Environment variable | Value | Description |
| --- | --- | --- |
| `ONEDNN_MAX_CPU_ISA` | `AVX2` | Intel Advanced Vector Extensions 2 (Intel AVX2) |

The exact installed [environment reader](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/common/utils.cpp#L112) contains:

```cpp
for (const auto &prefix : {"ONEDNN_", "DNNL_"}) {
    std::string name_str = std::string(prefix) + std::string(name);
```

and lowercases the value with `std::transform(..., ::tolower)`. The [ISA initialization source](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/cpu/x64/cpu_isa_traits.cpp#L35) reads `getenv_string_user("MAX_CPU_ISA")` and accepts `avx2`. Thus `ONEDNN_MAX_CPU_ISA=AVX2` is the modern spelling; `DNNL_MAX_CPU_ISA=AVX2` is accepted as a compatibility spelling, with `ONEDNN_` checked first. Both need not be set.

This setting controls oneDNN, which has its own ISA detector. The local A/B trace shows ATen's reported capability remains `AVX512` when oneDNN is capped to `AVX2`; an `ATEN_CPU_CAPABILITY` setting alone would not establish a oneDNN cap.

## Actual fixture operator trace and verification

[trace_precision_fixture.py](trace_precision_fixture.py) runs the existing precision fixture with its original assertions. It wraps `_nll` with a `TorchDispatchMode` observer that forwards each operator unchanged while recording `aten.addmm.default` tensor shapes, dtypes, and strides. `ONEDNN_VERBOSE=1` records actual backend dispatch. It uses only pytest-created temporary fixture artifacts.

The FFN up projection is the dense matrix operation `[32, 8] @ [8, 32]` with bias `[32]`. In the BF16 branch, all three `aten.addmm` inputs are `torch.bfloat16`; this is observed after CPU autocast's casts. In the FP32 branch, the same inputs are `torch.float32`. PyTorch's pinned source registers CPU `linear` and `addmm` with the `lower_precision_fp` autocast policy ([raw source](pytorch-autocast.cpp), lines 350–353), flattens contiguous biased linear input to `addmm` ([raw source](pytorch-linear.cpp)), and attempts `mkldnn_bf16_gemm` in BF16 CPU BLAS ([raw source](pytorch-blas.cpp), line 364).

| Fresh-process fixture run | Observed backend | Result |
| --- | --- | --- |
| Default oneDNN ISA | `Intel AVX 10.1`; BF16 `brg_matmul:avx512_core_bf16` for `32x8:8x32` | 1 passed, 7 deselected |
| `ONEDNN_MAX_CPU_ISA=AVX2` | `Intel AVX2`; BF16 `aten.addmm` retained; default BF16 oneDNN matmul primitive absent | 1 passed, 7 deselected |

Logs are [precision-default.log](precision-default.log) and [precision-avx2.log](precision-avx2.log). The capped run demonstrates the environment variable is honored by this Torch build, and the original BF16 precision assertions still execute. Linux cannot reproduce the native Windows hypervisor defect. The Python stack by itself cannot distinguish FP32 from BF16 at the native fatal point; this report distinguishes the fixture's observed dtype/kernel path from the native log's missing dtype.

Both commands ran from `/workspace/selftrained-v2`:

```bash
ONEDNN_VERBOSE=1 /workspace/tiny-perceptron-vlm/.venv/bin/python docs/course-revision-20261006-phase6/verification/windows-portability/evaluator/isa-research/trace_precision_fixture.py
ONEDNN_VERBOSE=1 ONEDNN_MAX_CPU_ISA=AVX2 /workspace/tiny-perceptron-vlm/.venv/bin/python docs/course-revision-20261006-phase6/verification/windows-portability/evaluator/isa-research/trace_precision_fixture.py
```

Set the cap only for the native Windows pytest invocation and retain all assertions. A native rerun should record `Get-CimInstance Win32_Processor`, `torch.__config__.show()`, and oneDNN verbose information if detailed dispatch confirmation is needed. A passing native full suite with this cap is evidence for the concrete CI mitigation; it does not retroactively identify the unrecorded failing instruction. No actual course model, saved candidate record, heldout set, frozen artifact, or Modal run was used.

[source-index.json](source-index.json) records authoritative URLs, fetch status, retained raw-file status, and SHA-256 values. Only relevant successful source bytes are retained; unsuccessful and broad exploratory lookups are recorded in the index without their unrelated payload files.
