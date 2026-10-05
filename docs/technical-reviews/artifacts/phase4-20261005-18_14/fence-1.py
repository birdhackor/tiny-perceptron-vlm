import copy
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.quantization import replace_linear_layers

teacher = TinyLM(ModelConfig(width=16, layers=2))
student = TinyLM(ModelConfig(width=8, layers=1))
quantized = replace_linear_layers(copy.deepcopy(student), bits=4)


def payload_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


for name, model in (("教師結構", teacher), ("學生結構", student), ("量化學生結構", quantized)):
    print(name, "bytes", payload_bytes(model))
print("量化學生/教師比例", round(payload_bytes(quantized) / payload_bytes(teacher), 4))
