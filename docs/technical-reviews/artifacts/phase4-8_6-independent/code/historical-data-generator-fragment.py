conditional = {
        split: [_style_record(row, style, True) for row in rows for style in ("concise", "vivid", "json")]
        for split, rows in arithmetic.items()
    }
dates = []
for day in range(1, 25):
        date = f"2026-10-{day:02d}"
        dates.append(
            _conversation(
                f"task=date;date={date};confirm", "已確認" + date + "。", "date-" + date, style="clarification"
            )
        )
        dates.append(
            _conversation(f"task=date;id={day};date=?;confirm", "請提供日期。", "date-" + date, style="clarification")
        )
date_parts = split_records(dates, seed=ctx.seed)
conditional = {split: rows + date_parts[split] for split, rows in conditional.items()}
