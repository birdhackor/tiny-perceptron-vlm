# PyTorch runtime 原始來源（2026-10-02）

這是來源準備，供之後的獨立技術核對使用；沒有讀取或修改課文章，也不表示任何課節或效能數字已審閱通過。原文副本位於 ignored `outputs/technical-sources/torch-runtime/`。完整 URLs、SHA-256、引文與定位收在同名 JSON。

## 實際版本與界線

- 本機：`2.14.1+cpu`，`torch.version.git_version=5c4886908584029761b579af026dcfb627c84070`。官方 `v2.14.1` tag 指向同一 SHA；5 個 installed Python source 檔逐 byte 與該固定 commit 相同。
- 已存在的正式報告 `docs/course-experiments/results/text_foundation.json` 記錄 `2.14.1+cu126`、`NVIDIA L4`。報告沒有記錄 GPU wheel 的 PyTorch git SHA／build flags，不能由 CPU wheel 推定。
- 實際下載的 API／CUDA 文件題名是 **PyTorch 2.14 documentation**；官方 compile tutorial 題名是 **PyTorch Tutorials 2.14.0+cu130 documentation**。Tutorial 的 GPU 示例不當作本專案 cu126/L4 成績。
- 所有抓取日期為 2026-10-02。本次 22 個官方 URL 全部 HTTP 200；沒有以不可訪問的來源當證據。2.14 URL 可更新，這份來源庫保留本次 HTML SHA；程式引用另外固定到官方 commit。

## 對後續核對最有用的差異

1. **SDPA 不等於必定 Flash。** 官方文件要求按輸入選 backend；固定 2.14.1 dispatcher 還包含 cuDNN。NVIDIA L4 官方 compute capability 是 8.9，滿足 NVIDIA Flash hardware gate 的 SM 範圍；但 SM8.x Flash 的 dtype gate 只有 FP16/BF16，FP32 會受限，且還有 build、head dimension、mask、dropout、stride 等 gate。Memory-efficient 在 SM8+ 的 dtype gate 包含 FP32。
2. **GQA 舊版本說法不能直接搬過來。** 2.14 API 明列 Flash/cuDNN/math，並明列 NVIDIA CUDA memory-efficient GQA；固定 commit 的 CUDA dense-input `supports_gqa=true` 可交叉查證。
3. **FP16 iteration 不一定有更新參數。** `GradScaler` 遇 inf/NaN gradient 可跳過 optimizer update 並縮小 scale；先前 loss 有限、`step()` 返回 None 或迴圈跑了一次都不能單獨證明更新成功。
4. **cold compile 與穩態時間分開。** 前幾次執行可能花編譯時間，guard failure 可再編譯。Inductor 或 CUDA graphs 的 memory/shape 條件不能由單次 speedup 範例外推。GPU 計時也須同步或 CUDA Events。
5. **allocator bytes 有明確範圍。** allocated 是 tensors bytes，reserved 是 caching allocator 管理的 bytes；peak 的 reset/起點要交代。context、cache、第三方直接配置／NCCL 等不能全部都解讀成模型 tensors bytes。

## 可直接定位的原文


### docs214-sdpa — torch.nn.functional.scaled_dot_product_attention — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.099179+00:00；HTTP 200。

原文 SHA-256：`f79084c6a533772c7874e5cb38b78b21d2d01a6f209cad49e742e1731180212f`。

SDPA 會依輸入自動選擇可用實作；fused kernel 有各自輸入限制。指定 backend 的測試應保留不適用原因，不可把呼叫 SDPA 等同 FlashAttention。

定位：scaled_dot_product_attention: implementation selection and limitations；快取文字 L535–548。

```text
The function may call optimized kernels for improved performance when using the CUDA backend.
For all other backends, the PyTorch implementation will be used.
All implementations are enabled by default. Scaled dot product attention attempts to automatically select the
most optimal implementation based on the inputs. In order to provide more fine-grained control over what implementation
is used, the following functions are provided for enabling and disabling implementations.
The context manager is the preferred mechanism:
torch.nn.attention.sdpa_kernel(): A context manager used to enable or disable any of the implementations.
torch.backends.cuda.enable_flash_sdp(): Globally enables or disables FlashAttention.
torch.backends.cuda.enable_mem_efficient_sdp(): Globally enables or disables  Memory-Efficient Attention.
torch.backends.cuda.enable_math_sdp(): Globally enables or disables  the PyTorch C++ implementation.
Each of the fused kernels has specific input limitations. If the user requires the use of a specific fused implementation,
disable the PyTorch C++ implementation using torch.nn.attention.sdpa_kernel().
In the event that a fused implementation is not available, a warning will be raised with the
reasons why the fused implementation cannot run.
```

不同 backend 的浮點結果可能不同；math backend 支援 float64，half/bfloat16 的中間值保留 float。

定位：scaled_dot_product_attention: numerical accuracy；快取文字 L549–552。

```text
Due to the nature of fusing floating point operations, the output of this function may be different
depending on what backend kernel is chosen.
The c++ implementation supports torch.float64 and can be used when higher precision is required.
For math backend, all intermediates are kept in torch.float if inputs are in torch.half or torch.bfloat16.
```

2.14 文件明列 GQA 支援 Flash、cuDNN、math，memory-efficient 在 NVIDIA CUDA 也支援；head 數仍有整除及 K/V 一致限制。

定位：scaled_dot_product_attention: Grouped Query Attention；快取文字 L554–559。

```text
Grouped Query Attention (GQA) is an experimental feature. It works with FlashAttention,
cuDNN attention, and the math kernel on CUDA tensors. Memory-efficient attention also supports
GQA on NVIDIA CUDA. GQA does not support Nested tensors.
Constraints for GQA:
number_of_heads_query % number_of_heads_key_value == 0 and,
number_of_heads_key == number_of_heads_value
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-amp — Automatic Mixed Precision package - torch.amp — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/amp.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.100311+00:00；HTTP 200。

原文 SHA-256：`13ef300cf6920386ea0964ca37b69daa25b79f7404107abca32e85907469729c`。

autocast 按 operation 選 dtype；模型及輸入不必先全面 half()，forward/loss 包在 autocast，backward 不建議包在其中。

定位：torch.autocast；快取文字 L3815–3824。

```text
allow regions of your script to run in mixed precision.
In these regions, ops run in an op-specific dtype chosen by autocast
to improve performance while maintaining accuracy.
See the Autocast Op Reference for details.
When entering an autocast-enabled region, Tensors may be any type.
You should not call half() or bfloat16() on your model(s) or inputs when using autocasting.
autocast should wrap only the forward pass(es) of your network, including the loss
computation(s).  Backward passes under autocast are not recommended.
Backward ops run in the same type that autocast used for corresponding forward ops.
Example for CUDA Devices:
```

FP16 小梯度可能 underflow；loss scaling 後須先 unscale 才更新參數。FP16 也可能 overflow，GradScaler 不保證 scale 大於 1。

定位：Gradient Scaling；快取文字 L3979–3995。

```text
If the forward pass for a particular op has float16 inputs, the backward pass for
that op will produce float16 gradients.
Gradient values with small magnitudes may not be representable in float16.
These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost.
To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and
invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are
then scaled by the same factor. In other words, gradient values have a larger magnitude,
so they don’t flush to zero.
Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer
updates the parameters, so the scale factor does not interfere with the learning rate.
Note
AMP/fp16 may not work for every model! For example, most bf16-pretrained models cannot operate in
the fp16 numerical range of max 65504 and will cause gradients to overflow instead of underflow. In
this case, the scale factor may decrease under 1 as an attempt to bring gradients to a number
representable in the fp16 dynamic range. While one may expect the scale to always be above 1, our
GradScaler does NOT make this guarantee to maintain performance. If you encounter NaNs in your loss
or gradients when running with AMP/fp16, verify your model is compatible.
```

float64/非浮點、in-place、out= 與顯式 dtype operation 不會依一般 autocast eligibility 轉型。

定位：Autocast Op Reference / Op Eligibility；快取文字 L4004–4014。

```text
Ops that run in float64 or non-floating-point dtypes are not eligible, and will
run in these types whether or not autocast is enabled.
Only out-of-place ops and Tensor methods are eligible.
In-place variants and calls that explicitly supply an out=... Tensor
are allowed in autocast-enabled regions, but won’t go through autocasting.
For example, in an autocast-enabled region a.addmm(b, c) can autocast,
but a.addmm_(b, c) and a.addmm(b, c, out=d) cannot.
For best performance and stability, prefer out-of-place ops in autocast-enabled
regions.
Ops called with an explicit dtype=... argument are not eligible,
and will produce output that respects the dtype argument.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-compile — torch.compile — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.compile.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.105052+00:00；HTTP 200。

原文 SHA-256：`473184fe5b0cb98ecd71f52b901b4d9c9735f35270ebf33ae804a321246cce67`。

torch.compile 會嘗試編譯並快取；guard failure 會再編譯，達限制後可退回 eager。不能假設固定一次編譯後所有輸入永久命中。

定位：torch.compile: description；快取文字 L1110–1122。

```text
Optimizes given model/function using TorchDynamo and specified backend.
If you are compiling an torch.nn.Module, you can also use torch.nn.Module.compile()
to compile the module inplace without changing its structure.
Concretely, for every frame executed within the compiled region, we will attempt
to compile it and cache the compiled result on the code object for future
use.  A single frame may be compiled multiple times if previous compiled
results are not applicable for subsequent calls (this is called a “guard
failure”), you can use TORCH_LOGS=guards to debug these situations.
Multiple compiled results can be associated with a frame up to
torch._dynamo.config.recompile_limit, which defaults to 8; at which
point we will fall back to eager.  Note that compile caches are per
code object, not frame; if you dynamically create multiple copies of a
function, they will all share the same code cache.
```

fullgraph=True 遇 graph break 會報錯；dynamic=False 固定 shape specialization；預設 backend 是 Inductor。

定位：torch.compile: fullgraph / dynamic / backend；快取文字 L1126–1140。

```text
in the function that it will optimize. If True, then we require that the entire function be
capturable into a single graph. If this is not possible (that is, if there are graph breaks),
then this will raise an error. This also opts into unbacked semantics, notably it will turn on
capture_scalar_outputs and capture_dynamic_output_shape_ops on by default.
dynamic (bool or None) – Use dynamic shape tracing.  When this is True, we will up-front attempt
to generate a kernel that is as dynamic as possible to avoid recompilations when
sizes change.  This may not always work as some operations/optimizations will
force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
When this is False, we will NEVER generate dynamic kernels, we will always specialize.
By default (None), we automatically detect if dynamism has occurred and compile a more
dynamic kernel upon recompile.
backend (str or Callable) –
backend to be used
”inductor” is the default backend, which is a good balance between performance and overhead
Non experimental in-tree backends can be seen with torch._dynamo.list_backends()
```

reduce-overhead/CUDA graphs 可降低 Python overhead，但可能多用 workspace 記憶體且不保證適用；文件另外說明 max-autotune 啟用 GPU CUDA graphs 的預設行為。

定位：torch.compile: mode；快取文字 L1145–1158。

```text
Can be either “default”, “reduce-overhead”, “max-autotune” or “max-autotune-no-cudagraphs”
”default” is the default mode, which is a good balance between performance and overhead
”reduce-overhead” is a mode that reduces the overhead of python with CUDA graphs,
useful for small batches.  Reduction of overhead can come at the cost of more memory
usage, as we will cache the workspace memory required for the invocation so that we
do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
There are other circumstances where CUDA graphs are not applicable; use TORCH_LOGS=perf_hints
to debug.
”max-autotune” is a mode that leverages Triton or template based matrix multiplications
on supported devices and Triton based convolutions on GPU.
It enables CUDA graphs by default on GPU.
”max-autotune-no-cudagraphs” is a mode similar to “max-autotune” but without CUDA graphs
To see the exact configs that each mode sets you can call torch._inductor.list_mode_options()
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-cuda — CUDA semantics — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/notes/cuda.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.105640+00:00；HTTP 200。

原文 SHA-256：`bff26e0d0e235b0dd16e81d9f688ac9c7418025fff9cfe13f4e74c6c983047dd`。

GPU operation 預設非同步，CPU 呼叫返回不等於 GPU 工作完成。

定位：Asynchronous execution；快取文字 L618–621。

```text
By default, GPU operations are asynchronous.  When you call a function that
uses the GPU, the operations are enqueued to the particular device, but not
necessarily executed until later.  This allows us to execute more computations
in parallel, including operations on CPU or other GPUs.
```

GPU 計時需適當 synchronization 或 CUDA Events；未同步的 wall clock 不能當 kernel 完成時間。

定位：Asynchronous execution: timing；快取文字 L632–641。

```text
A consequence of the asynchronous computation is that time measurements without
synchronizations are not accurate. To get precise measurements, one should either
call torch.cuda.synchronize() before measuring, or use torch.cuda.Event
to record times as following:
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# Run some things here
end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
```

allocated 是 tensors 使用量；reserved 是 caching allocator 管理總量。empty_cache 釋出未使用 cache，不會釋放仍被 tensor 佔用的記憶體。

定位：Memory management；快取文字 L759–769。

```text
PyTorch uses a caching memory allocator to speed up memory allocations. This
allows fast memory deallocation without device synchronizations. However, the
unused memory managed by the allocator will still show as if used in
nvidia-smi. You can use memory_allocated() and
max_memory_allocated() to monitor memory occupied by
tensors, and use memory_reserved() and
max_memory_reserved() to monitor the total amount of memory
managed by the caching allocator. Calling empty_cache()
releases all unused cached memory from PyTorch so that those can be used
by other GPU applications. However, the occupied GPU memory by tensors will not
be freed so it can not increase the amount of GPU memory available for PyTorch.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-allocated — torch.cuda.memory.memory_allocated — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_allocated.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.401202+00:00；HTTP 200。

原文 SHA-256：`992f108b5cad89a47f7b11913e100b757bca348ec1ae4b93226d9825149ad1f3`。

memory_allocated 回報選定 device 的 tensor bytes；通常不等於 nvidia-smi，cache/context 是差異來源。

定位：torch.cuda.memory.memory_allocated；快取文字 L480–492。

```text
torch.cuda.memory.memory_allocated(device=None)[source]#
Return the current GPU memory occupied by tensors in bytes for a given device.
Parameters:
device (torch.device or int, optional) – selected device. Returns
statistic for the current device, given by current_device(),
if device is None (default).
Return type:
int
Note
This is likely less than the amount shown in nvidia-smi since some
unused memory can be held by the caching allocator and some context
needs to be created on GPU. See Memory management for more
details about GPU memory management.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-reserved — torch.cuda.memory.memory_reserved — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_reserved.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:39:00.466680+00:00；HTTP 200。

原文 SHA-256：`0a5e86becb8bc80c95f0d89bee182e9b8a389836b886fcdeaa12f75935d8369d`。

memory_reserved 回報選定 device 的 caching allocator bytes，而非模型參數 bytes 或整張卡所有使用量。

定位：torch.cuda.memory.memory_reserved；快取文字 L480–485。

```text
torch.cuda.memory.memory_reserved(device=None)[source]#
Return the current GPU memory managed by the caching allocator in bytes for a given device.
Parameters:
device (torch.device or int, optional) – selected device. Returns
statistic for the current device, given by current_device(),
if device is None (default).
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### official-tag — PyTorch official v2.14.1 tag identity

原始 URL：https://api.github.com/repos/pytorch/pytorch/git/ref/tags/v2.14.1

版本：PyTorch v2.14.1 identity。抓取：2026-10-02T19:39:00.566655+00:00；HTTP 200。

原文 SHA-256：`46cd7504d9ac97ab4e627cafe9eb7d210d70fbd8c8dc700b4f231b8db7c5435f`。

官方 v2.14.1 tag 直接指向 commit 5c4886908584029761b579af026dcfb627c84070，與已安裝 CPU torch.version.git_version 相同。

定位： / 。

```text
{"sha": "5c4886908584029761b579af026dcfb627c84070", "type": "commit", "url": "https://api.github.com/repos/pytorch/pytorch/git/commits/5c4886908584029761b579af026dcfb627c84070"}
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### official-runtime-commit — PyTorch installed CPU runtime commit identity

原始 URL：https://api.github.com/repos/pytorch/pytorch/commits/5c4886908584029761b579af026dcfb627c84070

版本：PyTorch v2.14.1 identity。抓取：2026-10-02T19:39:00.655517+00:00；HTTP 200。

原文 SHA-256：`0dd7867f1a58b3b2f385c157aa39e5fa45dc218877c36faf9f37ccd626bd1397`。

固定 commit 的 GitHub API 回應確認 SHA 與 committer date；只作版本身分證據，不將提交訊息作功能論點。

定位： / 。

```text
{"sha": "5c4886908584029761b579af026dcfb627c84070", "committer_date": "2026-09-30T01:47:31Z"}
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### compile-tutorial — Introduction to torch.compile — PyTorch Tutorials 2.14.0+cu130 documentation

原始 URL：https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

版本：PyTorch Tutorials 2.14.0+cu130 documentation snapshot (not a cu126 benchmark)。抓取：2026-10-02T19:39:00.686538+00:00；HTTP 200。

原文 SHA-256：`74168c05d31a7ed8028818a203edcca68ee2dc9ca00c96832552a9b84ef61bc1`。

官方教學明述第一次／前幾次執行包含編譯額外時間；需要分別呈現 cold compile 與後續重用 compiled code 的時間。

定位：Demonstrating Speedups；快取文字 L484–489。

```text
Notice that torch.compile appears to take a lot longer to complete
compared to eager. This is because torch.compile takes extra time to compile
the model on the first few executions.
torch.compile re-uses compiled code whever possible,
so if we run our optimized model several more times, we should
see a significant improvement compared to eager.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。 來源中的示範 speedup 數字不外推到小模型、L4 或本專案；只引用 cold/warm 差異與量測方法。

### docs214-memory-usage — Understanding CUDA Memory Usage — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:41:23.154189+00:00；HTTP 200。

原文 SHA-256：`a5896c1dad3a502974b7c7b0a983ed0109d379bfb554e28515ea3497defdbe78`。

PyTorch memory profiler 主要可見自己 allocator 管理的 CUDA device allocation；CUDA API／第三方直接配置及 NCCL allocation 可在視野之外。

定位：Understanding CUDA Memory Usage: Note；快取文字 L3769–3776。

```text
By default, the memory profiler only has visibility into CUDA device memory allocated and managed through the
PyTorch allocator (e.g., torch.empty(..., device='cuda'), :func:torch.cuda.memory.CUDAPluggableAllocator).
CPU pinned memory (host memory) can optionally be included by passing record_pinned_host_memory=True
to :func:~torch.cuda.memory._record_memory_history; see Including Pinned (Host) Memory below.
Any memory allocated directly from CUDA APIs in C++ (e.g., cudaMalloc, cuMemCreate) or via third-party
Python bindings such as cuda-python <https://github.com/NVIDIA/cuda-python>_ will not be visible in the
PyTorch memory profiler. NCCL (used for distributed communication on CUDA devices) is a common example of a
library that allocates GPU memory outside of PyTorch’s allocator.  See Identifying Non-PyTorch allocations for more info.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。 此處對外部 allocation 的明述是 profiler visibility；allocated/reserved API 範圍另由 allocator 指標來源交叉支持。

### docs214-peak-allocated — torch.cuda.memory.max_memory_allocated — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:41:23.155276+00:00；HTTP 200。

原文 SHA-256：`e889c5f3a9eadf98bc918dc92332f0cbc996cf82c874463a0528a056a38acfe3`。

max_memory_allocated 預設追蹤整個程式開始以來 peak；reset_peak_memory_stats 可重設計量起點。

定位：torch.cuda.memory.max_memory_allocated；快取文字 L481–486。

```text
Return the maximum GPU memory occupied by tensors in bytes for a given device.
By default, this returns the peak allocated memory since the beginning of
this program. reset_peak_memory_stats() can be used to
reset the starting point in tracking this metric. For example, these two
functions can measure the peak allocated memory usage of each iteration in a
training loop.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### docs214-reset-peak — torch.cuda.memory.reset_peak_memory_stats — PyTorch 2.14 documentation

原始 URL：https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_peak_memory_stats.html

版本：PyTorch 2.14 documentation snapshot。抓取：2026-10-02T19:41:23.156378+00:00；HTTP 200。

原文 SHA-256：`69259770a30fbc16a540201eb21c6d93da5d8519f37f03843e60f150dff2040e`。

reset_peak_memory_stats 重設 allocator 所追蹤 peak 統計；其本身不是釋放 tensor/cache 的操作。

定位：torch.cuda.memory.reset_peak_memory_stats；快取文字 L480–485。

```text
torch.cuda.memory.reset_peak_memory_stats(device=None)[source]#
Reset the “peak” stats tracked by the CUDA memory allocator.
See memory_stats() for details. Peak stats correspond to the
“peak” key in each individual stat dict.
Parameters:
device (torch.device or int, optional) – selected device. Returns
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### source-sdp-cuda — PyTorch CUDA SDPA eligibility and dispatcher (v2.14.1 exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.156905+00:00；HTTP 200。

原文 SHA-256：`5aebb6de086f21291b7966566d00d9581d8d547316bb4f9f6963c03aa12a30d9`。

此 commit NVIDIA CUDA Flash hardware gate 的執行條件是 SM8.0 至 SM12.1；仍須通過 build、輸入形狀、dtype 等其它 gate。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / check_flash_attention_hardware_support，原始程式 L450–465；快取文字 L450–465。

```text
      TORCH_WARN("flash attention requires a CUDA device, which is not available.");
    }
    return false;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm80, sm121>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
```

此 commit NVIDIA memory-efficient hardware gate 的執行條件是 SM5.0 至 SM12.1。上方舊註解寫 sm90；此處以執行條件為依據。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / check_mem_efficient_hardware_support，原始程式 L515–523；快取文字 L515–523。

```text
  if (!check_sm_version<sm50, sm121>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Mem Efficient Attention only supports gpu architectures in the range [sm50, sm121]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
```

NVIDIA SM8.x 的 Flash dtype gate 僅允許 half/bfloat16，FP32 Q/K/V 無法通過；SM9+且啟用 FA3 時另有 float8_e4m3fn 分支，不概括所有新版 Flash 都只有兩種 dtype。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / check_dtypes_low_precision / check_dtypes_flash_attention，原始程式 L895–915；快取文字 L895–915。

```text
  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    constexpr auto sm80_dtypes =
        std::to_array<at::ScalarType>({at::kHalf, at::kBFloat16});
    return check_tensor_dtype(params, sm80_dtypes, debug);
  } else {
    constexpr auto default_dtypes = std::to_array<at::ScalarType>({at::kHalf});
    return check_tensor_dtype(params, default_dtypes, debug);
  }
}

bool check_dtypes_flash_attention(sdp_params const& params, bool debug) {
  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 9 and at::globalContext().userEnabledFA3SDP()) {
    constexpr auto fa3_dtypes =
        std::to_array<at::ScalarType>({at::kFloat8_e4m3fn, at::kHalf, at::kBFloat16});
    return check_tensor_dtype(params, fa3_dtypes, debug);
  } else {
    return check_dtypes_low_precision(params, debug);
  }
}
```

NVIDIA Flash gate 要求 Q/K/V 最後一維相同且 head dimension <=256；程式使用 <=，不沿用附近註解的 less than。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / check_head_dim_size_flash，原始程式 L203–212；快取文字 L203–212。

```text
  // All head_dim sizes must be equal and less than 256
  const auto max_size = c10::SymInt(256);
#endif
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last <= max_size))) {
    if (debug) {
```

部分 SM8.6–8.9／SM12.0–12.1 training 在 head dim >192 時另受 dropout/head dim gate；可用硬體與 dtype 仍不足以保證可用。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120，原始程式 L542–549；快取文字 L542–549。

```text
  bool is_head_dim_gt192 = params.query.sym_size(-1) > 192;
  bool is_head_dim_lte224 = params.query.sym_size(-1) <= 224;
  bool is_dropout = params.dropout > 0.0;
  //  head_dim size  in (192, 224] is not supported on sm86 and sm89
  bool cond1 = is_head_dim_gt192 && is_head_dim_lte224;
  // head_dim size > 224 and is_dropout is not supported on sm86 and sm89
  bool cond2 = params.query.sym_size(-1) > 224 && is_dropout;
  if (input_requires_grad(params) && (is_sm86_or_sm89 || is_sm120_or_sm121) && (cond1 || cond2)) {
```

Flash eligibility 包含 USE_FLASH_ATTENTION build flag 及多個 gate；GPU 名稱並不直接證明本次輸入有使用此 backend。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / can_use_flash_attention，原始程式 L1047–1065；快取文字 L1047–1065。

```text
bool can_use_flash_attention(sdp_params const& params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  if (debug) {
    TORCH_WARN("Torch was not compiled with flash attention.");
  }
  return false;
#else // defined(USE_FLASH_ATTENTION)
  // Define gate functions that determine if a flash kernel can be ran
  constexpr auto general_constraints = std::to_array<bool (*)(sdp_params const&, bool)>({
      check_runtime_disabled_flash,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_for_attn_mask,
      check_head_dim_size_flash<false /*caller_is_meff*/>,
      check_flash_attention_hardware_support,
      check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120,
      check_flash_causal_non_square_seqlens,
      check_dtypes_flash_attention});
  for (auto& constraint : general_constraints) {
```

NVIDIA memory-efficient 在 SM8+ 的 allowed dtype 是 FP16、FP32、BF16；SM8 之前是 FP16、FP32。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / can_use_mem_efficient_attention，原始程式 L1104–1114；快取文字 L1104–1114。

```text
  constexpr auto less_than_sm80_mem_efficient_dtypes =
      std::to_array<at::ScalarType>({at::kHalf, at::kFloat});
#ifdef USE_ROCM
  constexpr auto aotriton_mem_efficient_dtypes =
      std::to_array<at::ScalarType>({at::kHalf, at::kFloat, at::kBFloat16});
  constexpr auto ck_mem_efficient_dtypes =
      std::to_array<at::ScalarType>({at::kHalf, at::kBFloat16});
#else
  constexpr auto greater_than_or_equal_sm80_mem_efficient_dtypes =
      std::to_array<at::ScalarType>({at::kHalf, at::kFloat, at::kBFloat16});
#endif
```

固定 commit 的 memory-efficient dense-input CUDA 分支 supports_gqa=true，與目前官方 GQA 文件一致；ROCm 分支不同。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / can_use_mem_efficient_attention，原始程式 L1139–1153；快取文字 L1139–1153。

```text
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
#ifdef USE_ROCM
    constexpr bool supports_gqa = false;
    constexpr bool supports_mqa = false;
#else
    constexpr bool supports_gqa = true;
    constexpr bool supports_mqa = true;
#endif
    constexpr auto dense_constraints = std::to_array<bool (*)(sdp_params const&, bool)>({
        check_nonzero_sequence_lengths_dense,
```

CUDA dispatcher 逐一檢查 cuDNN、Flash、efficient、math；可用 backend 與實際優先選中 backend 需分別辨識。

定位：aten/src/ATen/native/transformers/cuda/sdp_utils.cpp / select_sdp_backend，原始程式 L1198–1226；快取文字 L1198–1226。

```text
  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
              return SDPBackend::cudnn_attention;
        }
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      case SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP()) {
          TORCH_CHECK(false, "Invalid backend");
        }
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 NVIDIA CUDA 分支與 ROCm 分支需分開。gate/source 支持必要條件，不能取代 cu126 wheel build、實際 SDPAParams 與 profiler/backend probe。 附近註解可能落後執行程式；硬體上限與 head dimension 邊界引用實際 condition。 FlashAttention gate 包含 FA3 分支，不能把 SM8.x 的 FP16/BF16 限制擴張成所有硬體/全部版本通則。

### source-sdp-common — PyTorch shared SDPA dtype checks (v2.14.1 exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/sdp_utils_cpp.h

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.157305+00:00；HTTP 200。

原文 SHA-256：`c31711aa9f1d8e5adf33e52851fba1abcd668d52c7c19a8337a50db26c04b48c`。

fused dtype 檢查要求 Q/K/V dtype 相同且屬該 backend allowed_dtypes。

定位：aten/src/ATen/native/transformers/sdp_utils_cpp.h / check_tensor_dtype，原始程式 L100–114；快取文字 L100–114。

```text
inline bool check_tensor_dtype(
    sdp_params const& params,
    dtype_vector allowed_dtypes,
    bool debug) {
  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    if (debug) {
      TORCH_WARN(
          "Expected query, key and value to all be of dtype: {",
          c10::Join(", ", allowed_dtypes),
          "}. Got ",
          "Query dtype: ",
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### source-cuda-backends — PyTorch CUDA SDPA Python predicates (installed source and exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/backends/cuda/__init__.py

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.157831+00:00；HTTP 200。

原文 SHA-256：`aa36a537f1fbea9292069b7b8cce52a5ef4c5f354b73165448a1ca9ab116d81e`。

Installed：`/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages/torch/backends/cuda/__init__.py`；逐 byte SHA 相同：`true`。

is_flash_attention_available 檢查 build；can_use_flash_attention 要輸入 SDPAParams。debug=True 可提供不適用原因；這些 CUDA API 在 CPU build 回傳 False。

定位：torch/backends/cuda/__init__.py / is_flash_attention_available / can_use_flash_attention，原始程式 L582–612；快取文字 L582–612。

```text
def is_flash_attention_available() -> bool:
    r"""Check if PyTorch was built with FlashAttention for scaled_dot_product_attention.

    Returns:
        True if FlashAttention is built and available; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._is_flash_attention_available()


def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_flash_attention(params, debug)
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### source-autocast — PyTorch autocast implementation/docstring (installed source and exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/amp/autocast_mode.py

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.158384+00:00；HTTP 200。

原文 SHA-256：`a927f545435b1ce7cfbe2754c2f44da316a234cf7eaac555a93266708ad8db03`。

Installed：`/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages/torch/amp/autocast_mode.py`；逐 byte SHA 相同：`true`。

已安裝 2.14.1 CPU wheel 的 autocast docstring 與固定官方 commit 檔案逐 byte SHA 相同，支持 operation-specific dtype 與 forward/loss 範圍。

定位：torch/amp/autocast_mode.py / autocast，原始程式 L52–68；快取文字 L52–68。

```text
class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### source-gradscaler — PyTorch GradScaler skip/update implementation (installed source and exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/amp/grad_scaler.py

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.167518+00:00；HTTP 200。

原文 SHA-256：`97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`。

Installed：`/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages/torch/amp/grad_scaler.py`；逐 byte SHA 相同：`true`。

GradScaler 遇 inf/NaN gradient 會跳過 underlying optimizer update，update 退 scale；初期校準可能有 skipped steps。訓練 iteration 數因此不能一律視為 parameter-update 數。

定位：torch/amp/grad_scaler.py / GradScaler，原始程式 L93–105；快取文字 L93–105。

```text
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).
```

一般 optimizer 路徑實際以 found_inf_per_device 判斷是否呼叫 optimizer.step；不是單看 loss 有限或 GradScaler.step 返回 None 來判斷成功。

定位：torch/amp/grad_scaler.py / GradScaler._maybe_opt_step / GradScaler.step，原始程式 L363–386；快取文字 L363–386。

```text
    def _maybe_opt_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_state: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> float | None:
        retval: float | None = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        return retval

    def step(
        self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any
    ) -> float | None:
        """Invoke ``unscale_(optimizer)`` followed by parameter update, if gradients are not infs/NaN.

        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

```

update 使用 backoff/growth 調 scale，需在所有 optimizer 的 step 之後呼叫，scale 不保證大於 1。

定位：torch/amp/grad_scaler.py / GradScaler.update，原始程式 L484–511；快取文字 L484–511。

```text
    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """Update the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.

        .. warning::
            For performance reasons, we do not check the scale factor value to avoid synchronizations,
            so the scale factor is not guaranteed to be above 1. If the scale falls below 1 and/or
            you are seeing NaNs in your gradients or loss, something is likely wrong. For example,
            bf16-pretrained models are often incompatible with AMP/fp16 due to differing dynamic ranges.
        """
        if not self._enabled:
            return

```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 特殊 optimizer 的 _step_supports_amp_scaling 路徑可自行處理 found_inf；一般 _maybe_opt_step 的 return None 不是可靠成功/skip旗標。

### source-memory — PyTorch CUDA allocator metric implementations (installed source and exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/cuda/memory.py

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.171631+00:00；HTTP 200。

原文 SHA-256：`ed3e6068d73ef9d2165f7276476041ee5bfe8f19db863e3e11172d840812173b`。

Installed：`/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages/torch/cuda/memory.py`；逐 byte SHA 相同：`true`。

已安裝 memory_allocated/max_memory_allocated 實際讀 allocator allocated_bytes.all.current/peak；reset 起點與 device 必須在結果中交代。

定位：torch/cuda/memory.py / memory_allocated / max_memory_allocated，原始程式 L525–563；快取文字 L525–563。

```text
def memory_allocated(device: "Device" = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: "Device" = None) -> int:
```

已安裝 memory_reserved 實際讀 reserved_bytes.all.current，與 allocated 使用不同統計。

定位：torch/cuda/memory.py / memory_reserved，原始程式 L563–580；快取文字 L563–580。

```text
def memory_reserved(device: "Device" = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### source-compile — PyTorch torch.compile entry point (installed source and exact commit)

原始 URL：https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/__init__.py

版本：PyTorch v2.14.1 / commit 5c4886908584029761b579af026dcfb627c84070。抓取：2026-10-02T19:41:23.173910+00:00；HTTP 200。

原文 SHA-256：`bd4a10ff16357ed78f45aaf4b46ce5b771fdc0e8549402206e324d0eb792c672`。

Installed：`/workspace/tiny-perceptron-vlm/.venv/lib/python3.13/site-packages/torch/__init__.py`；逐 byte SHA 相同：`true`。

已安裝 torch.compile 的快取/guard/recompile 文件與固定官方 commit 相同。

定位：torch/__init__.py / compile，原始程式 L3066–3078；快取文字 L3066–3078。

```text
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure"), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

```

已安裝 torch.compile 的 Inductor default 與 reduce-overhead workspace memory 說明。

定位：torch/__init__.py / compile，原始程式 L3094–3117；快取文字 L3094–3117。

```text

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOGS=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### nvidia-cuda-gpus — NVIDIA CUDA GPU Compute Capability table

原始 URL：https://developer.nvidia.com/cuda-gpus

版本：Unversioned official hardware table snapshot, accessed 2026-10-02。抓取：2026-10-02T19:42:23.472062+00:00；HTTP 200。

原文 SHA-256：`bf1b28c46e0f1cb74f3499394d4e6f25490e0917f189e358f82e58ccdfa33697`。

NVIDIA 官方 compute-capability 表的 8.9 row 列出 L4／L40／L40S；L4 位於 Flash NVIDIA hardware gate 範圍，並不能代替實際 wheel build/input/backend probe。

定位：CUDA GPU Compute Capability: 8.9 / Data Center GPUs；快取文字 L20–24。

```text
9.0
NVIDIA GH200NVIDIA H200NVIDIA H100
8.9
NVIDIA L4NVIDIA L40NVIDIA L40S
NVIDIA RTX 6000 AdaNVIDIA RTX 5000 AdaNVIDIA RTX 4500 AdaNVIDIA RTX 4000 AdaNVIDIA RTX 4000 SFF AdaNVIDIA RTX 2000 AdaGeForce RTX 4090GeForce RTX 4080GeForce RTX 4070 TiGeForce RTX 4070GeForce RTX 4060 TiGeForce RTX 4060GeForce RTX 4050
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 文件版本固定到 2.14 major/minor URL，但網頁可更新；以抓取日期與原文 SHA 固定本次所讀內容，不聲稱專屬 2.14.1 point release。

### tutorials-main-commit — PyTorch tutorials snapshot commit identity

原始 URL：https://api.github.com/repos/pytorch/tutorials/commits/main

版本：pytorch/tutorials commit 11512db7cbbcc4b530ecfc63205d3fed13fa190d。抓取：2026-10-02T19:42:23.472829+00:00；HTTP 200。

原文 SHA-256：`58067b87c3ab291a6c98d845f77ec2cb743bf21219633e0d52f06bc5c63bb1a5`。

2026-10-02 抓取官方 tutorials main 所解析的固定 commit，後續原文 URL 已改用其 SHA。這不是 PyTorch v2.14.1 的套件 tag。

定位： / 。

```text
{"sha": "11512db7cbbcc4b530ecfc63205d3fed13fa190d", "committer_date": "2026-09-30T22:09:34Z"}
```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。

### source-compile-tutorial — PyTorch official torch.compile tutorial source (fixed tutorials commit)

原始 URL：https://raw.githubusercontent.com/pytorch/tutorials/11512db7cbbcc4b530ecfc63205d3fed13fa190d/intermediate_source/torch_compile_tutorial.py

版本：pytorch/tutorials commit 11512db7cbbcc4b530ecfc63205d3fed13fa190d。抓取：2026-10-02T19:42:24.550934+00:00；HTTP 200。

原文 SHA-256：`6cbbf9de72b1acca550416f3f5d86d689cac97c6e67b0a09b04f2393ac9a7f54`。

固定官方 tutorial commit 的計時範例用 CUDA Events 並 synchronize；相同段落明述第一次／前幾次編譯額外成本。

定位：intermediate_source/torch_compile_tutorial.py / timed / Demonstrating Speedups，原始程式 L159–180；快取文字 L159–180。

```text
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


inp = torch.randn(4096, 4096).cuda()
print("compile:", timed(lambda: opt_foo3(inp))[1])
print("eager:", timed(lambda: foo3(inp))[1])

######################################################################
# Notice that ``torch.compile`` appears to take a lot longer to complete
# compared to eager. This is because ``torch.compile`` takes extra time to compile
# the model on the first few executions.
# ``torch.compile`` re-uses compiled code whever possible,
# so if we run our optimized model several more times, we should
# see a significant improvement compared to eager.

```

範圍：僅支持列出的來源論點；未做本倉庫課節正確性審閱，也不是本倉庫速度、精度或記憶體 benchmark。 來源中的示範 speedup 數字不外推到小模型、L4 或本專案；只引用 cold/warm 差異與量測方法。

## 尚未解決


- 正式 GPU cu126 wheel 的 torch.version.git_version、CUDA runtime/driver/build flags 尚未在本次來源準備中實測。CPU SHA/tag 相同並不自動證明 GPU wheel 的編譯選項相同。 由已授權正式 GPU runner 未來記錄 version/git_version/CUDA/device capability/build info；本工作不啟動任何付費工作。
- 本倉庫每種 dtype/head dimension/mask/dropout/GQA 形狀實際可用及選中的 backend 尚未由此來源庫驗證。 實際 CUDA build 中以 SDPAParams 的 can_use_* debug、強制 sdpa_kernel、profiler/kernel 名稱交叉辨識，保留不適用原因。
- 來源不能替代實驗：cold/warm 執行時間、FP16 skipped update 數、各段 peak allocator bytes 與外部/context memory 都要明示量測範圍。 實驗報告分開 cold compile/warmup/steady-state；記錄 attempted與successful updates，以及 reset/開始/結束 allocator 量測範圍。
