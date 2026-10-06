"""Read official workspace billing through the client; never create Modal compute."""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path


def main():
    import modal

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    observed = datetime.now(UTC)
    this_month = observed.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month = (this_month - timedelta(days=1)).replace(day=1)
    workspace = modal.Workspace.from_context()
    billing = workspace.billing
    summaries = []
    for cycle in (last_month, this_month):
        value = billing.summary(cycle=cycle)
        summaries.append(
            {
                "cycle": cycle.strftime("%Y-%m"),
                "start": value.start.isoformat(),
                "end": value.end.isoformat(),
                "metered_cost_usd": str(value.metered_cost),
                "billed_cost_usd": str(value.billed_cost),
                "adjustments_usd": {key: str(amount) for key, amount in value.adjustments.items()},
                "metered_cost_breakdown_usd": {
                    key: str(amount) for key, amount in value.metered_cost_breakdown.items()
                },
            }
        )
    complete_hour = observed.replace(minute=0, second=0, microsecond=0)
    today = complete_hour.replace(hour=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "workspace-billing-summary.json").write_text(
        json.dumps(
            {"observed_at": observed.isoformat(), "workspace_name": workspace.name, "summaries": summaries},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"observed_at": observed.isoformat(), "summaries": summaries}), flush=True)
    rows = billing.report(start=last_month, end=this_month, resolution="d")
    if this_month < today:
        rows += billing.report(start=this_month, end=today, resolution="d")
    if today < complete_hour:
        rows += billing.report(start=today, end=complete_hour, resolution="h")
    by_object = defaultdict(lambda: Decimal("0"))
    raw_rows = []
    for row in rows:
        by_object[(row.object_id, row.description, row.environment_name)] += row.cost
        raw_rows.append(
            {
                "object_id": row.object_id,
                "description": row.description,
                "environment_name": row.environment_name,
                "interval_start": row.interval_start.isoformat(),
                "metered_cost_usd": str(row.cost),
                "cost_by_resource_usd": {key: str(amount) for key, amount in row.cost_by_resource.items()},
            }
        )
    result = {
        "schema_version": 1,
        "observed_at": observed.isoformat(),
        "source": "Official modal.Workspace.billing.summary/report; modal==1.6.0",
        "workspace_name": workspace.name,
        "scope": "Authenticated workspace, including other projects; not a course-only invoice",
        "summaries": summaries,
        "report_start": last_month.isoformat(),
        "report_end_exclusive": complete_hour.isoformat(),
        "report_resolution": "daily before today, hourly today; nonoverlapping complete intervals",
        "report_metered_total_usd": str(sum((row.cost for row in rows), Decimal("0"))),
        "report_objects": [
            {"object_id": key[0], "description": key[1], "environment_name": key[2], "cost_usd": str(cost)}
            for key, cost in sorted(by_object.items(), key=lambda item: item[1], reverse=True)
        ],
        "report_rows": raw_rows,
        "limitations": [
            "Partial current hour is excluded; metering may lag running work.",
            "Adjustments include applied credit/discount, not guaranteed remaining credit balance.",
            "Previous and current calendar months only; cumulative course history may start earlier.",
        ],
        "remote_compute_started": False,
        "ledger_written": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "workspace-billing.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"observed_at": result["observed_at"], "summaries": summaries, "rows": len(rows)}))


if __name__ == "__main__":
    main()
