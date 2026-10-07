# Original PyTorch v2.9.0 Module.eval at module.py:2894; original SHA256 d076e98c7037d342c82662c13a11dbb2584d080279a140431243475f27056dfd
def eval(self) -> Self:
        r"""Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)
