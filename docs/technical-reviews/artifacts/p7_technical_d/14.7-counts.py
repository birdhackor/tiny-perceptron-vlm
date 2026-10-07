s='校慶週六上午開始請在操場入口集合';print('cards',len(s),list(enumerate(s)))
for old in [8,12]:print('training_count',old,'old_positions',list(range(old)),'new_positions',list(range(old,len(s))),'old_max_distance',old-1,'new_max_distance',len(s)-1)
