from tiny_perceptron.model import TinyLM,ModelConfig
m=TinyLM(ModelConfig(width=16,layers=1));n=sum(p.numel() for p in m.parameters());print('parameters',n,'FP32bytes',4*n);assert n==13744
