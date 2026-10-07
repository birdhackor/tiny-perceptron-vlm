teacher_answers=['2','不知道','不知道'];student_answers=teacher_answers.copy();truth=['2','4',None]
def acceptable(a,e):return ('不知道' in a or '資訊不足' in a) if e is None else a.strip()==e
checks=[acceptable(a,t) for a,t in zip(student_answers,truth)];print('agreement',sum(a==b for a,b in zip(teacher_answers,student_answers))/3,'checks',checks,'taskrate',round(sum(checks)/3,4))
