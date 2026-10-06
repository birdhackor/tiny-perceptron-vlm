def _pku_pilot(ctx):
    raw = _asset_rows(ctx, "pku-safe-rlhf", "train-first-100.jsonl")
    records = []
    for index, row in enumerate(raw):
        safer = int(row["safer_response_id"])
        if not row[f"is_response_{safer}_safe"]:
            continue
        answer = row[f"response_{safer}"]
        records.append(
            _conversation(
                _utf8_prefix(row["prompt"], 120),
                _utf8_prefix(answer, 120),
                _digest(row["prompt"]),
                source_row=index,
                source_record_sha256=_digest(row),
                safer_response_id=safer,
                better_response_id=int(row["better_response_id"]),
            )
        )
