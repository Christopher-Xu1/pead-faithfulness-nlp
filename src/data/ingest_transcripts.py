from __future__ import annotations

import argparse

import pandas as pd

from src.data.transcript_corpus import inventory_raw_files
from src.data.transcript_sources import download_source, resolve_sources
from src.utils.io import ensure_dir, load_yaml, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/transcript_curation.yaml")
    parser.add_argument("--raw-root", default="data/raw/transcripts")
    parser.add_argument("--manifest-output", default="data/interim/audit/ingest_manifest.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    source_ids = cfg.get("sources")
    resolved_sources = resolve_sources(source_ids)

    manifest_rows = []
    for source in resolved_sources:
        result = download_source(source=source, raw_root=args.raw_root)
        manifest_rows.append(result)
        LOGGER.info("[%s] %s", source.source_id, result["message"])

    manifest_df = pd.DataFrame(manifest_rows)
    ensure_dir("data/interim/audit")
    write_csv(manifest_df, args.manifest_output)

    inventory_df = inventory_raw_files(args.raw_root)
    write_csv(inventory_df, "data/interim/audit/raw_file_inventory.csv")
    LOGGER.info("Saved ingest manifest with %s sources.", len(manifest_df))


if __name__ == "__main__":
    main()
