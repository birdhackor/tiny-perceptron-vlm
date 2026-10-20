"""核對包裝、參數對應、輸入路徑及缺圖錯誤的獨立短程式。"""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM as LibraryMultiModalLM, expand_modalities, scene, tone

torch.set_num_threads(1)
source = ROOT / 'docs/course-review-20261019/freeze-01/sources/10.8.md'
raw = source.read_bytes()
fences = re.findall(r'```python\n(.*?)\n```', raw.decode(), re.S)
namespace = {}
text = io.StringIO()
with contextlib.redirect_stdout(text):
    for code in fences:
        exec(compile(code, str(source), 'exec'), namespace)
ours, library, lm = [namespace[name] for name in ('ours', 'library', 'lm')]
ids, sequence = namespace['ids'], namespace['sequence']
result = {
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'threads': str(torch.get_num_threads())},
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'frozen_fence_stdout': text.getvalue(),
    'forward_signature': str(inspect.signature(ours.forward)),
    'tokenizer': {'bos_id': ByteTokenizer.bos_id, 'image_id': ByteTokenizer.image_id, 'audio_id': ByteTokenizer.audio_id, 'vocab_size': ByteTokenizer.vocab_size, 'ordinary_ids_decode_as_bytes': [ids.tolist()[1] - 8, ids.tolist()[2] - 8]},
    'shared_language_identity': ours.language is library.language is lm,
    'independent_image_objects': ours.vision is not library.vision and ours.image_projector is not library.image_projector,
    'matching_parameter_keys_equal_after_load': all(torch.equal(value, library.state_dict()[key]) for key, value in ours.state_dict().items()),
    'missing_keys': list(namespace['loaded'].missing_keys),
    'unexpected_keys': list(namespace['loaded'].unexpected_keys),
}
assert result['shared_language_identity'] and result['independent_image_objects']
assert result['matching_parameter_keys_equal_after_load']
assert result['missing_keys'] == []
assert all(key.startswith(('audio.', 'audio_projector.')) for key in result['unexpected_keys'])
pre_a = namespace['MultiModalLM'](TinyLM(ModelConfig(width=8)))
pre_b = LibraryMultiModalLM(pre_a.language)
result['independent_random_image_weights_before_load'] = not torch.equal(pre_a.vision.projection.weight, pre_b.vision.projection.weight)
assert result['independent_random_image_weights_before_load']
with torch.no_grad():
    direct = lm(ids[None])['logits']
    wrapped = ours(ids)['logits']
    result['text'] = {'logits_shape': list(direct.shape), 'same_text': torch.equal(direct, wrapped), 'maximum_difference': (direct - wrapped).abs().max().item()}
    assert result['text']['same_text'] and tuple(direct.shape) == (1, 3, 264)
    embeddings = lm.embedding(ids[None])
    result['precomputed_embeddings_match_id_path'] = torch.equal(direct, lm(embeddings=embeddings)['logits'])
    image = scene()
    vision = ours.vision(image[None])
    projected = ours.image_projector(vision)
    supplied_labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, 2])
    x, y = expand_modalities(sequence, supplied_labels, lm.embedding, {5: projected[0]}, {5})
    positions = []
    embedding_calls = []
    hook_position = lm.position.register_forward_pre_hook(lambda module, args: positions.append(args[0].tolist()))
    hook_embedding = lm.embedding.register_forward_pre_hook(lambda module, args: embedding_calls.append(args[0].tolist()))
    labelled = ours(sequence, supplied_labels, image=image)
    hook_position.remove()
    hook_embedding.remove()
    unlabelled = ours(sequence, image=image)
    reference = library(sequence, supplied_labels, image=image)
    result['image'] = {'pixels_shape': list(image.shape), 'red_channel_sum': image[0].sum().item(), 'other_channels_sum': image[1:].sum().item(), 'vision_shape': list(vision.shape), 'projected_shape': list(projected.shape), 'projector_in_features': ours.image_projector.in_features, 'projector_out_features': ours.image_projector.out_features, 'language_width': lm.config.width, 'expanded_input_shape': list(x.shape), 'logits_shape': list(labelled['logits'].shape), 'shifted_labels': y.tolist(), 'positions_seen': positions, 'embedding_calls': embedding_calls, 'default_labels': unlabelled['labels'].tolist(), 'default_labels_dtype': str(unlabelled['labels'].dtype), 'default_labels_equal_full_like_shifted': torch.equal(unlabelled['labels'], torch.full_like(y, IGNORE)), 'returned_labels_match_expansion': torch.equal(labelled['labels'], y), 'library_labels_match': torch.equal(reference['labels'], labelled['labels']), 'same_image': torch.equal(labelled['logits'], reference['logits']), 'maximum_difference': (labelled['logits'] - reference['logits']).abs().max().item()}
    assert tuple(vision.shape) == (1,16,16) and tuple(projected.shape) == (1,16,8)
    assert tuple(x.shape) == (1,19,8) and tuple(labelled['logits'].shape) == (1,19,264)
    assert y.tolist() == [[IGNORE] * 17 + [73,2]]
    assert positions == [list(range(19))] and embedding_calls == [[1],[4],[73],[2]]
    assert result['image']['same_image'] and result['image']['returned_labels_match_expansion'] and result['image']['library_labels_match']
    assert result['image']['default_labels_equal_full_like_shifted']
    audio_sequence = torch.tensor([1,6,4,73,2])
    audio_output = library(audio_sequence, waveform=tone())
    result['library_audio'] = {'has_audio_module': hasattr(library,'audio'), 'has_audio_projector': hasattr(library,'audio_projector'), 'logits_shape': list(audio_output['logits'].shape)}
errors = {}
for name, call in [
    ('missing_image', lambda: ours(torch.tensor([1,5,4]))),
    ('image_passed_positionally', lambda: ours(ids, None, scene())),
    ('library_missing_image', lambda: library(torch.tensor([1,5,4]))),
    ('library_missing_audio', lambda: library(torch.tensor([1,6,4]))),
]:
    try:
        call()
    except (TypeError, ValueError) as error:
        errors[name] = {'type': type(error).__name__, 'message': str(error)}
result['errors'] = errors
assert errors['missing_image'] == {'type':'ValueError','message':'placeholder 缺少配對圖片'}
assert errors['image_passed_positionally']['type'] == 'TypeError'
assert errors['library_missing_audio']['type'] == 'ValueError'
trained = copy.deepcopy(lm)
optimizer = torch.optim.SGD(trained.parameters(), lr=0.01)
before = trained(ids[None])['logits'].detach().clone()
loss = torch.nn.functional.cross_entropy(trained(ids[None])['logits'].reshape(-1,264), torch.tensor([20,30,2]))
loss.backward()
optimizer.step()
after = trained(ids[None])['logits'].detach()
result['isolated_parameter_update'] = {'optimizer': 'SGD', 'learning_rate':0.01, 'steps':1, 'input_ids':ids.tolist(), 'targets':[20,30,2], 'valid_targets':3, 'loss':loss.item(), 'logits_changed': not torch.equal(before,after), 'maximum_difference': (before-after).abs().max().item(), 'scope':'只展示更新文字參數能改變固定輸入的分數，不評估舊技能或訓練成效。'}
assert result['isolated_parameter_update']['logits_changed']
print(json.dumps(result, ensure_ascii=False, indent=2))
