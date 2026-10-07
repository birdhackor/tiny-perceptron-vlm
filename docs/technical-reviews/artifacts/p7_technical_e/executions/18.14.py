import copy
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.quantization import replace_linear_layers
teacher=TinyLM(ModelConfig(width=16,layers=2));student=TinyLM(ModelConfig(width=8,layers=1));q=replace_linear_layers(copy.deepcopy(student),bits=4)
def payload_bytes(model):return sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
for n,m in [('教師結構',teacher),('學生結構',student),('量化學生結構',q)]:print(n,'bytes',payload_bytes(m),'parameter_bytes',sum(t.numel()*t.element_size() for t in m.parameters()),'buffer_bytes',sum(t.numel()*t.element_size() for t in m.buffers()))
print('量化學生/教師比例',round(payload_bytes(q)/payload_bytes(teacher),4))
