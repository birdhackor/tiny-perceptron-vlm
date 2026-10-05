def _prompt_ids(messages, tokenizer):
    roles = {"system": tokenizer.system_id, "user": tokenizer.user_id, "assistant": tokenizer.assistant_id}
    ids = [tokenizer.bos_id]
    for message in messages:
        ids += [roles[message["role"]]] + tokenizer.encode(message["content"]) + [tokenizer.eos_id]
    return ids + [tokenizer.assistant_id]
