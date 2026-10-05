import json
import torch
from tiny_perceptron.multimodal import scene, VisionEncoder

torch.set_num_threads(1)
torch.manual_seed(0)
image = scene("red", "square")
original = image.clone()
shape_before = tuple(image.shape)
shape_tuple = tuple(image.shape)
central_list = image[:, 8, 8].tolist()
central_list[0] = 99.0
assert torch.equal(image, original) and tuple(image.shape) == shape_before
assert image[:, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert image[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
assert shape_tuple == (3, 16, 16)
assert image.min().item() == 0.0 and image.max().item() == 1.0
assert image[1].count_nonzero().item() == image[2].count_nonzero().item() == 0
mask = torch.zeros(16, 16, dtype=torch.bool)
mask[4:13, 4:13] = True
assert torch.equal(image[0] > 0, mask)
unsqueezed = image.unsqueeze(0)
assert tuple(unsqueezed.shape) == (1, 3, 16, 16)
assert tuple(image.shape) == (3, 16, 16)
assert unsqueezed.numel() == image.numel() == 768
assert unsqueezed.data_ptr() == image.data_ptr()
assert torch.equal(unsqueezed[0], image)
blue = scene("blue", "square")
assert tuple(blue.shape) == (3, 16, 16)
assert blue[:, 8, 8].tolist() == [0.0, 0.0, 1.0]
assert blue[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
assert blue[0].count_nonzero().item() == blue[1].count_nonzero().item() == 0
assert torch.equal(blue[2], image[0])
batch = torch.stack([image, blue])
assert tuple(batch.shape) == (2, 3, 16, 16)
assert torch.equal(batch[0], image)
assert batch[0, :, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert batch[1, :, 8, 8].tolist() == [0.0, 0.0, 1.0]
# A non-symmetric coordinate probe separates channel, row, and column.
probe = torch.arange(3*2*4).reshape(3, 2, 4)
assert probe[:, 1, 2].tolist() == [6, 14, 22]
hwc = image.permute(1, 2, 0)
assert tuple(hwc.shape) == (16, 16, 3)
assert torch.equal(hwc.permute(2, 0, 1), image)
wrong_chw = hwc.reshape(3, 16, 16)
assert not torch.equal(wrong_chw, image)
bytes_rgb = torch.tensor([0, 128, 255], dtype=torch.uint8)
normalized = bytes_rgb.to(torch.float32) / 255
assert normalized[0].item() == 0.0 and normalized[2].item() == 1.0
assert normalized.min().item() >= 0.0 and normalized.max().item() <= 1.0
scaled_twice = image / 255
assert scaled_twice[:, 8, 8].tolist() != image[:, 8, 8].tolist()
# Only inspect the encoder's input contract; random features are no quality test.
encoder = VisionEncoder()
with torch.no_grad():
 features = encoder(image.unsqueeze(0))
try:
 encoder(hwc.unsqueeze(0))
except ValueError as error:
 shape_error = str(error)
else:
 raise AssertionError("HWC input should fail the project encoder shape contract")
print(json.dumps({
 "seed": 0, "threads": torch.get_num_threads(), "images": 2,
 "red_shape": list(image.shape), "red_dtype": str(image.dtype),
 "red_center": image[:,8,8].tolist(), "red_top_left": image[:,0,0].tolist(),
 "red_range": [image.min().item(), image.max().item()],
 "red_nonzero_by_channel": [int(image[c].count_nonzero()) for c in range(3)],
 "display_conversions_preserve_original": torch.equal(image,original),
 "unsqueezed_shape": list(unsqueezed.shape), "unsqueeze_same_storage": unsqueezed.data_ptr()==image.data_ptr(),
 "blue_shape": list(blue.shape), "blue_center": blue[:,8,8].tolist(),
 "batch_shape": list(batch.shape), "batch_first_image_shape": list(batch[0].shape),
 "batch_centers": batch[:,:,8,8].tolist(),
 "non_symmetric_axis_probe_shape": list(probe.shape), "non_symmetric_axis_probe_pixel": probe[:,1,2].tolist(),
 "hwc_shape": list(hwc.shape), "permutation_restores_chw": torch.equal(hwc.permute(2,0,1),image),
 "reshape_hwc_as_chw_matches": torch.equal(wrong_chw,image),
 "uint8_input": bytes_rgb.tolist(), "float_div255": normalized.tolist(),
 "scene_div255_center": scaled_twice[:,8,8].tolist(),
 "encoder_features_shape": list(features.shape), "encoder_hwc_native_error": shape_error,
 "assertions": "all passed"
},ensure_ascii=False,indent=2))
