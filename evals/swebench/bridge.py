#!/usr/bin/env python3
"""Small bridge executed with the SWE-bench Python environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swebench.harness.utils import load_swebench_dataset


SAFE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "version")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_swebench_dataset(args.dataset, args.split, args.instance_ids)
    tasks = [{key: row.get(key, "") for key in SAFE_FIELDS} for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
