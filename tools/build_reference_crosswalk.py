"""Create a versioned private crosswalk from existing source inventories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from emg_fatigue.source_crosswalk import build_source_crosswalk


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--rar-audit", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    if output == repo or repo in output.parents:
        raise SystemExit("refusing to place the identity-bearing crosswalk in the repository")

    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    source_hashes = {
        "project_zip": next(
            row["sha256"] for row in receipt["sources"]
            if row["artifact"] == "emg integrador biomecanica.zip"
        ),
        "raw_rar": next(row["sha256"] for row in receipt["sources"] if row["artifact"] == "raw.rar"),
    }
    with args.inventory.open(newline="", encoding="utf-8-sig") as handle:
        inventory_rows = list(csv.DictReader(handle))
    rar_rows = json.loads(args.rar_audit.read_text(encoding="utf-8"))
    records = build_source_crosswalk(
        inventory_rows=inventory_rows,
        rar_audit_rows=rar_rows,
        archive_hashes=source_hashes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    output_hash = sha256(output)
    output.with_suffix(".receipt.json").write_text(
        json.dumps({
            "audit_date": receipt["audit_date"],
            "crosswalk_sha256": output_hash,
            "crosswalk_rows": len(records),
            "source_inventory_sha256": sha256(args.inventory),
            "raw_rar_audit_sha256": sha256(args.rar_audit),
            "source_receipt_sha256": sha256(args.source_receipt),
            "participant_ids_verified": False,
            "cross_archive_identity_link": "unresolved",
            "privacy": "private",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"private crosswalk rows={len(records)} sha256={output_hash}")


if __name__ == "__main__":
    main()
