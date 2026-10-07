by_question = (12 + 0) / (12 + 6)
by_task = (1.0 + 0.0) / 2
print("按題合計", by_question, "每類等權重", by_task)
assert by_question == 2/3 and by_task == .5
print("variation_equal_counts",(6+0)/(6+6),(1.0+0.0)/2)
assert (6+0)/(6+6)==by_task
