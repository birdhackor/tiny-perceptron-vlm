add_docstr(
    torch.multinomial,
    r"""
multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial (a stricter definition would be multivariate,
refer to :class:`torch.distributions.multinomial.Multinomial` for more details)
probability distribution located in the corresponding row
of tensor :attr:`input`.

.. note::
    The rows of :attr:`input` do not need to sum to one (in which case we use
    the values as weights), but must be non-negative, finite and have
    a non-zero sum.

Indices are ordered from left to right according to when each was sampled
(first samples are placed in first column).

If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
:math:`(m \times \text{{num\_samples}})`.

If replacement is ``True``, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

.. note::
    When drawn without replacement, :attr:`num_samples` must be lower than
    number of non-zero elements in :attr:`input` (or the min number of non-zero
    elements in each row of :attr:`input` if it is a matrix).

Args:
    input (Tensor): the input tensor containing probabilities
    num_samples (int): number of samples to draw
    replacement (bool, optional): whether to draw with replacement or not

Keyword args:
    {generator}
    {out}

Example::

    >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    >>> torch.multinomial(weights, 2)
    tensor([1, 2])
    >>> torch.multinomial(weights, 5) # ERROR!
    RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 2,  1,  1,  1])
""".format(**common_args),
)
