add_docstr(
    torch.tensor,
    r"""
tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with no autograd history (also known as a "leaf tensor", see :doc:`/notes/autograd`) by copying :attr:`data`.

.. warning::

    When working with tensors prefer using :func:`torch.Tensor.clone`,
    :func:`torch.Tensor.detach`, and :func:`torch.Tensor.requires_grad_` for
    readability. Letting `t` be a tensor, ``torch.tensor(t)`` is equivalent to
    ``t.detach().clone()``, and ``torch.tensor(t, requires_grad=True)``
    is equivalent to ``t.detach().clone().requires_grad_(True)``.

.. seealso::

    :func:`torch.as_tensor` preserves autograd history and avoids copies where possible.
    :func:`torch.from_numpy` creates a tensor that shares storage with a NumPy array.

Args:
    {data}

Keyword args:
    {dtype}
    device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
        then the device of data is used. If None and data is not a tensor then
        the result tensor is constructed on the current device.
    {requires_grad}
    {pin_memory}


Example::

    >>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[ 0.1000,  1.2000],
            [ 2.2000,  3.1000],
            [ 4.9000,  5.2000]])

    >>> torch.tensor([0, 1])  # Type inference on data
    tensor([ 0,  1])

    >>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
    ...              dtype=torch.float64,
    ...              device=torch.device('cuda:0'))  # creates a double tensor on a CUDA device
    tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

    >>> torch.tensor(3.14159)  # Create a zero-dimensional (scalar) tensor
    tensor(3.1416)

    >>> torch.tensor([])  # Create an empty tensor (of size (0,))
    tensor([])
""".format(**factory_data_common_args),
)

add_docstr(
    torch.round,
    r"""
round(input, *, decimals=0, out=None) -> Tensor

Rounds elements of :attr:`input` to the nearest integer.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.
The return type of output is same as that of input's dtype.

.. note::
    This function implements the "round half to even" to
    break ties when a number is equidistant from two
    integers (e.g. `round(2.5)` is 2).

    When the :attr:\`decimals\` argument is specified the
    algorithm used is similar to NumPy's `around`. This
    algorithm is fast but inexact and it can easily
    overflow for low precision dtypes.
    Eg. `round(tensor([10000], dtype=torch.float16), decimals=3)` is `inf`.

.. seealso::
    :func:`torch.ceil`, which rounds up.
    :func:`torch.floor`, which rounds down.
    :func:`torch.trunc`, which rounds towards zero.

Args:
    {input}
    decimals (int): Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions
        to the left of the decimal point.

Keyword args:
    {out}

Example::

    >>> torch.round(torch.tensor((4.7, -2.3, 9.1, -7.7)))
    tensor([ 5.,  -2.,  9., -8.])

    >>> # Values equidistant from two integers are rounded towards the
    >>> #   the nearest even value (zero is treated as even)
    >>> torch.round(torch.tensor([-0.5, 0.5, 1.5, 2.5]))
    tensor([-0., 0., 2., 2.])

    >>> # A positive decimals argument rounds to the to that decimal place
    >>> torch.round(torch.tensor([0.1234567]), decimals=3)
    tensor([0.1230])

    >>> # A negative decimals argument rounds to the left of the decimal
    >>> torch.round(torch.tensor([1200.1234567]), decimals=-3)
    tensor([1000.])
""".format(**common_args),
)

add_docstr(
    # torch.softmax doc str. Point this to torch.nn.functional.softmax
    torch.softmax,
    r"""
softmax(input, dim, *, dtype=None) -> Tensor

Alias for :func:`torch.nn.functional.softmax`.
""",
)
