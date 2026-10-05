add_docstr_all(
    "item",
    r"""
item() -> number

Returns the value of this tensor as a standard Python number. This only works
for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

This operation is not differentiable.

Example::

    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0

""",
)

add_docstr_all(
    "round",
    r"""
round(decimals=0) -> Tensor

See :func:`torch.round`
""",
)

add_docstr_all(
    "softmax",
    r"""
softmax(dim) -> Tensor

Alias for :func:`torch.nn.functional.softmax`.
""",
)

add_docstr_all(
    "tolist",
    r"""
tolist() -> list or number

Returns the tensor as a (nested) list. For scalars, a standard
Python number is returned, just like with :meth:`~Tensor.item`.
Tensors are automatically moved to the CPU first if necessary.

This operation is not differentiable.

Examples::

    >>> a = torch.randn(2, 2)
    >>> a.tolist()
    [[0.012766935862600803, 0.5415473580360413],
     [-0.08909505605697632, 0.7729271650314331]]
    >>> a[0,0].tolist()
    0.012766935862600803
""",
)
