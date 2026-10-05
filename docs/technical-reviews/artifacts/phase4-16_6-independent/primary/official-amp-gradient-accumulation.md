### Gradient accumulation

Gradient accumulation adds gradients over an effective batch of size `batch_per_iter * iters_to_accumulate`
(`* num_procs` if distributed). The scale should be calibrated for the effective batch, which means inf/NaN checking,
step skipping if inf/NaN grads are found, and scale updates should occur at effective-batch granularity.
Also, grads should remain scaled, and the scale factor should remain constant, while grads for a given effective
batch are accumulated. If grads are unscaled (or the scale factor changes) before accumulation is complete,
the next backward pass will add scaled grads to unscaled grads (or grads scaled by a different factor)
after which it's impossible to recover the accumulated unscaled grads {meth}`step <step>` must apply.

Therefore, if you want to {meth}`unscale_ <unscale_>` grads (e.g., to allow clipping unscaled grads),
call {meth}`unscale_ <unscale_>` just before {meth}`step <step>`, after all (scaled) grads for the upcoming
{meth}`step <step>` have been accumulated. Also, only call {meth}`update <update>` at the end of iterations
where you called {meth}`step <step>` for a full effective batch:

```python
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

```{currentmodule} torch.amp

```

