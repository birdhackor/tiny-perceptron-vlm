from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    ["hello hello hello"],
    trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]"], show_progress=False),
)
whole = tok.encode("hello")
first = tok.encode("hel")
second = tok.encode("lo")
print("整段", whole.tokens, whole.ids)
print("分段後相接", first.tokens + second.tokens, first.ids + second.ids)
assert whole.ids != first.ids + second.ids
