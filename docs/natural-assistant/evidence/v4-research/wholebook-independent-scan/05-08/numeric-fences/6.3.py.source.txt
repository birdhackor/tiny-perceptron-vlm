from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=280,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
tok.train_from_iterator(["貓看狗。", "a cat sees a dog."], trainer)
text = "小鳥🙂new"
encoded = tok.encode(text)
print("片段", encoded.tokens)
print("ID", encoded.ids)
print("還原", tok.decode(encoded.ids))
assert tok.decode(encoded.ids) == text
