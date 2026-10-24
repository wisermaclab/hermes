# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""JSON and tabular summaries for diagnostic benchmarks."""

import csv
import json
from pathlib import Path


def write_summary(output_dir: Path, metadata: dict, rows: list[dict]) -> None:
    """Write summary.json and, when rows exist, summary.csv."""
    summary = {**metadata, "results": rows}
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary JSON: {json_path}", flush=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary CSV: {csv_path}", flush=True)
