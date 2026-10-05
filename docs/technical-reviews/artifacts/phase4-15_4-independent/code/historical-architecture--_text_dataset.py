def _text_dataset(ctx):
    directory = extract_asset(ctx, "tinystories")
    files = list(directory.rglob("tinystories-train-512.jsonl"))
    if len(files) != 1:
        raise ValueError(f"預期唯一 TinyStories 資料檔，找到 {len(files)} 個")
    records = load_jsonl(files[0])
    for record in records:
        record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
    data = split_records(records, ctx.seed)
    write_json(ctx.output / "dataset.json", data)
    return data
