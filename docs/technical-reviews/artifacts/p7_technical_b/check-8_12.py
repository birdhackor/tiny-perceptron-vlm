short='答案是4。';long=short*10;candidates=[short,long];lengths=[len(text) for text in candidates];chosen=max(candidates,key=len)
print('字元數',lengths);print('長度裁判選長版',chosen==long);print('不同句子數',len(set(long.split('。')[:-1])))
assert lengths==[5,50] and chosen==long and len(set(long.split('。')[:-1]))==1
repeat3=short*3;assert len(repeat3)==15 and len(set(repeat3.split('。')[:-1]))==1
useful=short+'兩盤各有兩顆蘋果，合在一起共有四顆。';assert len(set(useful.split('。')[:-1]))==2
print('三次重複',len(repeat3),'不同句子',1);print('有用長版',useful,'不同句子',2,'人工判準','分組與合併成立')
