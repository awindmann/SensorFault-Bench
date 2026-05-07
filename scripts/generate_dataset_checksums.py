#!/usr/bin/env python3
"""Generate file-fixity SHA256 manifests for standalone dataset bundles."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_defaults


DATASET_CHECKSUM_BUNDLES = {
    "beijing_air_tiantan": (
        "beijing_air_tiantan.parquet",
        "beijing_air_tiantan.csv",
    ),
    "penmanshiel_hourly_wt08": (
        "penmanshiel_hourly_wt08.parquet",
        "penmanshiel_hourly_wt08.csv",
    ),
}


def default_data_root() -> Path:
    defaults = load_defaults()
    if "DATA_ROOT" not in defaults:
        raise ValueError("configs/defaults.yaml is missing required key 'DATA_ROOT'.")
    return Path(defaults["DATA_ROOT"])


def sha256_file_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checksum_lines(
    data_root: Path,
    *,
    bundle: str,
) -> tuple[str, ...]:
    try:
        filenames = DATASET_CHECKSUM_BUNDLES[bundle]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_CHECKSUM_BUNDLES))
        raise ValueError(
            f"Unknown dataset checksum bundle '{bundle}'. Known bundles: {known}."
        ) from exc
    missing = [filename for filename in filenames if not (data_root / filename).is_file()]
    if missing:
        missing_display = ", ".join(str(data_root / filename) for filename in missing)
        raise FileNotFoundError(
            "Cannot build dataset file-checksum manifest. "
            f"Missing staged file(s): {missing_display}"
        )
    return tuple(
        f"{sha256_file_bytes(data_root / filename)}  {filename}"
        for filename in filenames
    )


def build_manifest_text(data_root: Path, *, bundle: str) -> str:
    return "\n".join(build_checksum_lines(data_root, bundle=bundle)) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate file-fixity checksums for a standalone dataset bundle."
        )
    )
    parser.add_argument(
        "--bundle",
        choices=sorted(DATASET_CHECKSUM_BUNDLES),
        required=True,
        help="Dataset bundle to checksum by exact file bytes.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Directory containing staged dataset upload files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    manifest_text = build_manifest_text(args.data_root, bundle=args.bundle)
    if args.output is None:
        print(manifest_text, end="")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest_text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
