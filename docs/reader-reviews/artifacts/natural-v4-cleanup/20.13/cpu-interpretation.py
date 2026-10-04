import unicodedata
print("generated answers", sum([42, 84, 18, 10, 3, 13, 4, 4]))
print("ended plus truncated", 171 + 7)
print("raw CER %", round(112 / 674 * 100, 2))
print("normalized CER %", round(96 / 661 * 100, 2))
print("raw denominator split", 632 + 42)
print("normalized denominator split", 619 + 42)
print("NFKC + whitespace removal", "".join(unicodedata.normalize("NFKC", "１２　貓。").split()))
