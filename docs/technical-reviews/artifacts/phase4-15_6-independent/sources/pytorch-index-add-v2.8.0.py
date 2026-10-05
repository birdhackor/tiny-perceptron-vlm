# Original v2.8.0 torch/_tensor_docs.py lines 2373-2423
add_docstr_all(
    "index_add_",
    r"""
index_add_(dim, index, source, *, alpha=1) -> Tensor

Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
tensor by adding to the indices in the order given in :attr:`index`. For example,
if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
``source`` is subtracted from the ``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of ``source`` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

For a 3-D tensor the output is given as::

    self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
    self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
    self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

Note:
    {forward_reproducibility_note}

Args:
    dim (int): dimension along which to index
    index (Tensor): indices of ``source`` to select from,
            should have dtype either `torch.int64` or `torch.int32`
    source (Tensor): the tensor containing values to add

Keyword args:
    alpha (Number): the scalar multiplier for ``source``

Example::

    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
    >>> x.index_add_(0, index, t, alpha=-1)
    tensor([[  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.]])
""".format(**reproducibility_notes),
)
# Original v2.8.0 torch/_tensor_docs.py lines 5510-5530
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
# Original v2.8.0 torch/_tensor_docs.py lines 6506-6513
add_docstr_all(
    "index_add",
    r"""
index_add(dim, index, source, *, alpha=1) -> Tensor

Out-of-place version of :meth:`torch.Tensor.index_add_`.
""",
)
