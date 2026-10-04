ORIGINAL LINES 14-35

Each `torch.Tensor` has a {class}`torch.dtype`, {class}`torch.device`, and {class}`torch.layout`.

(dtype-doc)=

## torch.dtype

```{eval-rst}
.. class:: dtype
```

A {class}`torch.dtype` is an object that represents the data type of a
{class}`torch.Tensor`. PyTorch has several different data types:

**Floating point dtypes**

| dtype | description |
| --- | --- |
| `torch.float32` or `torch.float` | 32-bit floating point, as defined in <https://en.wikipedia.org/wiki/IEEE_754> |
| `torch.float64` or `torch.double` | 64-bit floating point, as defined in <https://en.wikipedia.org/wiki/IEEE_754> |
| `torch.float16` or `torch.half` | 16-bit floating point, as defined in <https://en.wikipedia.org/wiki/IEEE_754>, S-E-M 1-5-10 |
| `torch.bfloat16` | 16-bit floating point, sometimes referred to as Brain floating point, S-E-M 1-8-7 |
