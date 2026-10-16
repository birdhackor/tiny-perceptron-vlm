# 10.8 original-source receipt
Reviewer: /root/repair_tech_10_8
Accessed: 2026-10-16 UTC
Only the line ranges below were read for the claims. Full external originals stay outside Git.
Rendered docs URLs https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html and https://docs.pytorch.org/docs/stable/generated/torch.equal.html returned HTTP 403. Raw official repositories below were successfully read instead.

## module-original.py
URL: https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py
Local external original: /workspace/work/textbook-repair-20261017/technical-10-8-original-01/module-original.py
Full-original SHA256: d139fc1dc9adb172c208b8b470fc405d8037612c0f08768d275c5bbcc284b9c9
Read/excerpt lines 2894-2932:
```text
2894:     def train(self, mode: bool = True) -> Self:
2895:         r"""Set the module in training mode.
2896: 
2897:         This has an effect only on certain modules. See the documentation of
2898:         particular modules for details of their behaviors in training/evaluation
2899:         mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
2900:         etc.
2901: 
2902:         Args:
2903:             mode (bool): whether to set training mode (``True``) or evaluation
2904:                          mode (``False``). Default: ``True``.
2905: 
2906:         Returns:
2907:             Module: self
2908:         """
2909:         if not isinstance(mode, bool):
2910:             raise ValueError("training mode is expected to be boolean")
2911:         self.training = mode
2912:         for module in self.children():
2913:             module.train(mode)
2914:         return self
2915: 
2916:     def eval(self) -> Self:
2917:         r"""Set the module in evaluation mode.
2918: 
2919:         This has an effect only on certain modules. See the documentation of
2920:         particular modules for details of their behaviors in training/evaluation
2921:         mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
2922:         etc.
2923: 
2924:         This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
2925: 
2926:         See :ref:`locally-disable-grad-doc` for a comparison between
2927:         `.eval()` and several similar mechanisms that may be confused with it.
2928: 
2929:         Returns:
2930:             Module: self
2931:         """
2932:         return self.train(False)
```

## equal-original.py
URL: https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py
Local external original: /workspace/work/textbook-repair-20261017/technical-10-8-original-01/equal-original.py
Full-original SHA256: f3962a2a4973e7d4917a3b3f265903d4a534085343bbe40d003728385d1949e5
Read/excerpt lines 4241-4263:
```text
4241: add_docstr(
4242:     torch.equal,
4243:     r"""
4244: equal(input, other) -> bool
4245: 
4246: ``True`` if two tensors have the same size and elements, ``False`` otherwise.
4247: 
4248: .. note::
4249: 
4250:     Tensors containing NaNs are never equal to each other. Additionally, this function does not
4251:     differentiate between the data types of the tensors during comparison. For more thorough tensor checks,
4252:     use :meth:`torch.testing.assert_close`.
4253: 
4254: Example::
4255: 
4256:     >>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
4257:     True
4258:     >>> torch.equal(torch.tensor([3, torch.nan]), torch.tensor([3, torch.nan]))
4259:     False
4260:     >>> torch.equal(torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([1, 2, 3], dtype=torch.float32))
4261:     True
4262: """,
4263: )
```

## optimization-original.py
URL: https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/optimization_tutorial.py
Local external original: /workspace/work/textbook-repair-20261017/technical-10-8-original-01/optimization-original.py
Full-original SHA256: ae54687a9b959fbc64375974d933089e5d705492baed6235018e2dc6899fc28a
Read/excerpt lines 150-191:
```text
150: # We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that
151: # evaluates the model's performance against our test data.
152: 
153: def train_loop(dataloader, model, loss_fn, optimizer):
154:     size = len(dataloader.dataset)
155:     # Set the model to training mode - important for batch normalization and dropout layers
156:     # Unnecessary in this situation but added for best practices
157:     model.train()
158:     for batch, (X, y) in enumerate(dataloader):
159:         # Compute prediction and loss
160:         pred = model(X)
161:         loss = loss_fn(pred, y)
162: 
163:         # Backpropagation
164:         loss.backward()
165:         optimizer.step()
166:         optimizer.zero_grad()
167: 
168:         if batch % 100 == 0:
169:             loss, current = loss.item(), batch * batch_size + len(X)
170:             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
171: 
172: 
173: def test_loop(dataloader, model, loss_fn):
174:     # Set the model to evaluation mode - important for batch normalization and dropout layers
175:     # Unnecessary in this situation but added for best practices
176:     model.eval()
177:     size = len(dataloader.dataset)
178:     num_batches = len(dataloader)
179:     test_loss, correct = 0, 0
180: 
181:     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
182:     # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
183:     with torch.no_grad():
184:         for X, y in dataloader:
185:             pred = model(X)
186:             test_loss += loss_fn(pred, y).item()
187:             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
188: 
189:     test_loss /= num_batches
190:     correct /= size
191:     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

## sgd-original.py
URL: https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py
Local external original: /workspace/work/textbook-repair-20261017/technical-10-8-original-01/sgd-original.py
Full-original SHA256: e6c647249cee41c99cd29da4cb3fe74c94d899e775c9f952308fcafdcae43725
Read/excerpt lines 367-379:
```text
367:                 grad = grad.add(buf, alpha=momentum)
368:             else:
369:                 grad = buf
370: 
371:         # Nested if is necessary to bypass jitscript rules
372:         if isinstance(lr, Tensor):
373:             if lr.requires_grad:
374:                 param.addcmul_(grad, lr, value=-1)
375:             else:
376:                 # pyrefly: ignore [bad-argument-type]
377:                 param.add_(grad, alpha=-lr)
378:         else:
379:             param.add_(grad, alpha=-lr)
```

## Repository source versions
tiny_perceptron/model.py SHA256 176e3a2c305f3ec509934abe41a1e8e48c5a76de8e9b0cea5c41eefd5a9adf69
tiny_perceptron/multimodal.py SHA256 298b165e3cdb56fdf5ce42e6e17eaf3d005b4954eb867fc5b9889eb3a3adf340
tiny_perceptron/attention.py SHA256 edb29d24392f1b7aedf04890b106add669bc0e31bfde7fde7f80ad190b007c15
tiny_perceptron/modern.py SHA256 02f6332a7676f98342ed3f168f5c178e466f7acb649bdd48554b87dc66caf24a
tiny_perceptron/data.py SHA256 6b40ba4bccc4a1ce169a90df8ce71f40e7c9e33e3d7633a36fdd8348d9536321
