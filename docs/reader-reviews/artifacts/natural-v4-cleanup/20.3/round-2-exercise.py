parameters = 328_128
for label, bytes_per_number in [("32-bit", 4), ("16-bit", 2)]:
    weight_bytes = parameters * bytes_per_number
    print(label, "權重數字bytes", weight_bytes, "約GiB", round(weight_bytes / 1024**3, 2))
