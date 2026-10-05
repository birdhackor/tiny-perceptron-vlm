def text_examples(records, mode="text", max_length=128, tokenizer=None):
    tok = tokenizer or ByteTokenizer()
    examples = []
    for record in records:
        if mode == "sft":
            messages = record["messages"] if isinstance(record, dict) else record
            x, y = render_chat(messages, tok)
            if len(x) > max_length:
                raise ValueError(f"SFT record has {len(x)} tokens, exceeding {max_length}; crop explicitly upstream")
            examples.append((x, y))
        elif mode == "text":
            text = record["text"] if isinstance(record, dict) else record
            ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
            # Every next-token label appears once; chunk boundaries share only the preceding input token.
            for start in range(0, len(ids) - 1, max_length):
                examples.append(shifted(ids[start : start + max_length + 1]))
        else:
            raise ValueError("mode must be text or sft")
    if not examples:
        raise ValueError("No examples to train or evaluate")
    return examples
