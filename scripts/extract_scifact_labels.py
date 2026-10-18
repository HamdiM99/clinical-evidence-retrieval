"""Extract the official SciFact claim annotations (SUPPORT / CONTRADICT rationales) from the AllenAI release.

The BEIR version of SciFact only keeps "relevant" documents (the documents a claim cites); the original release
also says, per document, whether it SUPPORTS or CONTRADICTS the claim, or neither. The 300 BEIR test queries are
the original `claims_dev.jsonl`.

Get the archive (≈ 3 MB) from https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz, then:
  python scripts/extract_scifact_labels.py path/to/data.tar.gz   ->  data/scifact-official/claims_{dev,train}.jsonl
The archive is read in place (nothing else is extracted).
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

OUT = Path("data/scifact-official")


def main(archive: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as t:
        for split in ("dev", "train"):
            data = t.extractfile(f"data/claims_{split}.jsonl").read()
            (OUT / f"claims_{split}.jsonl").write_bytes(data)
            print(f"{OUT / f'claims_{split}.jsonl'}: {data.count(b'\n')} claims")


if __name__ == "__main__":
    main(sys.argv[1])
