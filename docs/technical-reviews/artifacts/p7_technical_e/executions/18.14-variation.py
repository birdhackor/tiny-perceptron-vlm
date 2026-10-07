import copy
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.quantization import replace_linear_layers
s=TinyLM(ModelConfig(width=8,layers=1));q=replace_linear_layers(copy.deepcopy(s),bits=8)
def size(m):return sum(t.numel()*t.element_size() for t in list(m.parameters())+list(m.buffers()))
print('bits8',size(q),'fp',size(s),'teacher',size(TinyLM(ModelConfig(width=16,layers=2))))
print('parameters',sum(t.numel()*t.element_size() for t in q.parameters()),'buffers',sum(t.numel()*t.element_size() for t in q.buffers()))
