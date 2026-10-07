width,expert_hidden,experts=4,2,4
per_expert=2*width*expert_hidden;router=width*experts;total=experts*per_expert+router
for k in (1,2):
 active=k*per_expert+router;same_parameters_hidden=total/(2*width);same_active_estimate_hidden=active/(2*width)
 print('k',k,'MoE總權重',total,'使用估計',active);print('同總參數Dense寬度',same_parameters_hidden,'同使用估計Dense寬度',same_active_estimate_hidden)
