import sys,json,torch
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'CPU'}))

print('fence 0')
examples = [
    {"transcript": "你好", "pitch": "low"},
    {"transcript": "你好", "pitch": "high"},
]
print("轉寫相同", examples[0]["transcript"] == examples[1]["transcript"])
print("音高相同", examples[0]["pitch"] == examples[1]["pitch"])
for row in examples:
    print("只看轉寫", row["transcript"], "實際標籤", row["pitch"])

# Owner supplied proportionate control
def text_only_rule(transcript):
    return {'你好': 'low'}[transcript]
answers = [text_only_rule(row['transcript']) for row in examples]
print('same input deterministic answers', answers, 'equal', answers[0] == answers[1])
print('matches labels', [answer == row['pitch'] for answer, row in zip(answers, examples)])
print('rich inputs differ', (examples[0]['transcript'], examples[0]['pitch']) != (examples[1]['transcript'], examples[1]['pitch']))
assert answers[0] == answers[1]
assert examples[0]['pitch'] != examples[1]['pitch']
assert sum(answer == row['pitch'] for answer, row in zip(answers, examples)) == 1
