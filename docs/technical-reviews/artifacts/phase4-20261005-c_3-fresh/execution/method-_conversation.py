def _conversation(question, answer):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
