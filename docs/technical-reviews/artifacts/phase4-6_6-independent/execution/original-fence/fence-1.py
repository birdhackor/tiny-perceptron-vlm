from tiny_perceptron.data import ByteTokenizer, SPECIALS

tok = ByteTokenizer()
print(list(enumerate(SPECIALS)))
literal = tok.encode("<assistant>")
print("普通文字的ID", literal)
print("還原", tok.decode(literal))
assert tok.assistant_id not in literal
