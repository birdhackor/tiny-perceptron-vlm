from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
for k in [1,2]:
 m=LimitedAssistant(SelftrainedConfig(vocab_size=550,width=256,layers=4,heads=4,kv_heads=2,ffn_hidden=512,experts=4,top_k=k,max_length=512));r=m.description();print('topk',k,'total',r['parameters'],'language',r['language_parameters'],'active',r['active_language_parameters_per_token'],'perception',sum(r['perception_parameters'].values()),'experts',r['expert_parameters'],'router',r['router_parameters'])
t=CharacterTokenizer.load('outputs/p7-technical-e-cache/moe-joint/tokenizer.json');print('vocab',t.vocab_size,'controlsplusunk',len(t.special_ids),'characters',len(t.characters));print('unknown',t.encode('🚀'),t.unk_id,t.decode(t.encode('🚀')))
