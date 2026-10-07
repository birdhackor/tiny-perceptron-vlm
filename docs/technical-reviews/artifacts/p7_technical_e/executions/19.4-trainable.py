from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
from scripts.selftrained.train import set_trainable,PERCEPTION_BRIDGE_PREFIXES
m=LimitedAssistant(SelftrainedConfig(vocab_size=550,width=32,layers=1,heads=4,kv_heads=2,ffn_hidden=64,experts=4,top_k=2))
for stage,freeze in [('pretrain',False),('sft',False),('vision',False),('ocr',False),('audio',False),('joint',True)]:
 set_trainable(m,stage,freeze);actual={n for n,p in m.named_parameters() if p.requires_grad};prefix={'vision':'vision_encoder.','ocr':'ocr_encoder.','audio':'audio_encoder.'}.get(stage)
 expected={n for n,p in m.named_parameters() if n.startswith(('lm.',*PERCEPTION_BRIDGE_PREFIXES))} if freeze else {n for n,p in m.named_parameters() if n.startswith(prefix) if prefix} if prefix else {n for n,p in m.named_parameters() if n.startswith('lm.')}
 assert actual==expected;print(stage,'trainable_named_tensors',len(actual),'lm',m.lm.embedding.weight.requires_grad,'vision_head',m.vision_encoder.head.weight.requires_grad,'vision_cnn',m.vision_encoder.cnn[0].weight.requires_grad,'vision_bridge',m.vision_encoder.symbol_projection.weight.requires_grad)
