#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Split a request set into an exploration set and a sealed set by a fixed hash of each request id.

    python split_requests.py REQUESTS.jsonl --out-dir DIR --salt NAME [--sealed-fraction 0.3] [--id-field id]
        [--group-field FIELD]

A request goes to the sealed set when sha256("<salt>:<key>") falls in the first SEALED_FRACTION of
the hash range, so the split depends only on the keys and the salt, not on the order of the file.
The key is the request's id, or with --group-field the value of that field: the turns of one
session share it and go to the same set, so no sealed request continues a conversation that was
explored. Writes DIR/explore.jsonl and DIR/sealed.jsonl, prints the count and sha256 of each, and
exits 1 on a missing or duplicate id.
"""
import argparse
import hashlib
import json
import os
import sys


def in_sealed(request_id: str, salt: str, fraction: float) -> bool:
    digest = hashlib.sha256(f"{salt}:{request_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < fraction * 2**64


def split(lines, salt: str, fraction: float, id_field: str = "id", group_field: str = None):
    """Return (explore, sealed) lists of raw lines; raise ValueError on a missing or duplicate id."""
    seen = set()
    explore, sealed = [], []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = row.get(id_field)
        if request_id is None:
            raise ValueError(f"line {number}: no {id_field!r}")
        request_id = str(request_id)
        if request_id in seen:
            raise ValueError(f"line {number}: duplicate {id_field} {request_id!r}")
        seen.add(request_id)
        key = request_id
        if group_field:
            if row.get(group_field) is None:
                raise ValueError(f"line {number}: no {group_field!r}")
            key = str(row[group_field])
        (sealed if in_sealed(key, salt, fraction) else explore).append(line.rstrip("\n") + "\n")
    return explore, sealed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requests")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--salt", required=True, help="a name for this study; the same salt gives the same split")
    parser.add_argument("--sealed-fraction", type=float, default=0.3)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--group-field", default=None, help="requests that share this field go to the same set")
    args = parser.parse_args()
    with open(args.requests) as f:
        try:
            explore, sealed = split(f, args.salt, args.sealed_fraction, args.id_field, args.group_field)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("explore", explore), ("sealed", sealed)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        data = "".join(rows).encode()
        with open(path, "wb") as f:
            f.write(data)
        print(f"{name}: {len(rows)} requests, sha256 {hashlib.sha256(data).hexdigest()}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
