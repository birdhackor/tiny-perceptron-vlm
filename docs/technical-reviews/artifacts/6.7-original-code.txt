raw = "貓🙂".encode()
print("全部bytes", list(raw))
print("只看第一byte", raw[:1].decode("utf-8", errors="replace"))
print("先拼完整再解碼", raw.decode("utf-8"))
assert raw.decode("utf-8") == "貓🙂"
