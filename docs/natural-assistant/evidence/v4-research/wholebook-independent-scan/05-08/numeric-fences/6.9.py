from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    ["hello hello hello"],
    trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]"], show_progress=False),
)
whole = tok.encode("hello").ids
chunks = tok.encode("hel").ids + tok.encode("lo").ids
print("整段", whole, "分段後相接", chunks)
assert whole != chunks
