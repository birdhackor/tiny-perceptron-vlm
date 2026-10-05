
Original pytorch-quantization-v2.2.0.rst lines 205-214
Post Training Static Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Post Training Static Quantization (PTQ static) quantizes the weights and activations of the model.  It
fuses activations into preceding layers where possible.  It requires
calibration with a representative dataset to determine optimal quantization
parameters for activations. Post Training Static Quantization is typically used when
both memory bandwidth and compute savings are important with CNNs being a
typical use case.


Original pytorch-quantization-v2.2.0.rst lines 278-285
  # Prepare the model for static quantization. This inserts observers in
  # the model that will observe activation tensors during calibration.
  model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

  # calibrate the prepared model to determine quantization parameters for activations
  # in a real world setting, the calibration would be done with a representative dataset
  input_fp32 = torch.randn(4, 1, 4, 4)
  model_fp32_prepared(input_fp32)
