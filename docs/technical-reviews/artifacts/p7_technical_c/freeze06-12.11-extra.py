def text_only_rule(transcript):
    return {'你好': 'low'}[transcript]
answers = [text_only_rule(row['transcript']) for row in examples]
print('same input deterministic answers', answers, 'equal', answers[0] == answers[1])
print('matches labels', [answer == row['pitch'] for answer, row in zip(answers, examples)])
print('rich inputs differ', (examples[0]['transcript'], examples[0]['pitch']) != (examples[1]['transcript'], examples[1]['pitch']))
assert answers[0] == answers[1]
assert examples[0]['pitch'] != examples[1]['pitch']
assert sum(answer == row['pitch'] for answer, row in zip(answers, examples)) == 1
