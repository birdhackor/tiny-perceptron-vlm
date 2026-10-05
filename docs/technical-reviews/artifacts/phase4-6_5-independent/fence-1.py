import math

raw = "貓🙂"
bytes_count = len(raw.encode("utf-8"))
for name, total_nll in [("示例A", 5.0), ("示例B", 4.0)]:
    bpb = total_nll / (bytes_count * math.log(2))
    print(name, "原文bytes", bytes_count, "BPB", bpb)
