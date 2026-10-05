class ByteTokenizer:
    """不學切詞的可還原起點；每個 UTF-8 byte 對應一個 ID。"""

    vocab_size = 264
    pad_id, bos_id, eos_id, user_id, assistant_id, image_id, audio_id, system_id = range(8)

    def encode(self, text):
        return [b + len(SPECIALS) for b in text.encode("utf-8")]

    def decode(self, ids):
        raw = bytes(i - len(SPECIALS) for i in ids if i >= len(SPECIALS))
        return raw.decode("utf-8", errors="replace")

    def state(self):
        return {"type": "byte", "specials": list(SPECIALS)}
