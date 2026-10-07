from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
for a in ['moe','dense']:
 c=SelftrainedConfig(vocab_size=550,width=256,layers=4,heads=4,kv_heads=2,ffn_hidden=512,experts=4,top_k=2,max_length=512,architecture=a);m=LimitedAssistant(c);r=m.description();print(a,'全部',r['parameters'],'文字',r['language_parameters']);print('每位置的邏輯啟用文字參數',r['active_language_parameters_per_token']);print('嵌入與輸出共用同一張表',m.lm.embedding.weight is m.lm.output.weight)
