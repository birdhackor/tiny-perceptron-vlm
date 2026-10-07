families={'train':['盒子沒有數量資訊，請問有幾球？'],'test':['盒子還沒打開，你能確定裡面有幾顆嗎？']}
original_train=list(families['train']);families['test'].append('只看外觀能知道盒內球數嗎')
expected='資訊不足，請提供數量或可看清的圖片。'
for s,prompts in families.items():
    for q in prompts:print(s,q,'→',expected)
assert families['train']==original_train and len(families['test'])==2
