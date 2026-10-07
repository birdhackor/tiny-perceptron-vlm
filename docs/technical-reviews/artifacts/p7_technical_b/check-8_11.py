answers={'correct':'2+2=4','wrong':'2+2=5'};original_order=['correct','wrong'];swapped_order=original_order[::-1]
def choose_first(order):return order[0]
first=choose_first(original_order);second=choose_first(swapped_order)
print('原順序選',first,answers[first]);print('交換後選',second,answers[second]);print('內容身分一致',first==second)
assert (first,second)==('correct','wrong')
def choose_correct(order):return next(identity for identity in order if answers[identity]=='2+2=4')
assert choose_correct(original_order)==choose_correct(swapped_order)=='correct'
print('依內容裁判',choose_correct(original_order),choose_correct(swapped_order),'一致',True)
