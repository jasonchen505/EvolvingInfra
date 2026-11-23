#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Split a request set into development requests and sealed parts by a fixed hash of each request's key.

    python split_requests.py REQUESTS.jsonl --out-dir DIR --salt NAME [--sealed-fraction 0.3]
        [--sealed-parts R] [--id-field id] [--group-field FIELD] [--force]

A request is sealed when sha256("<salt>:<key>") falls in the first SEALED_FRACTION of the hash
range, so the split depends only on the keys and the salt, not on the order of the file. The key is
the request's id, or with --group-field the value of that field: requests that share it go to the
same file, so no sealed request continues a conversation that was explored. The
sealed groups are dealt into R disjoint parts by a second hash, one part per round's sealed final
(SKILL.md, step 7); every part is used once. SEALED_FRACTION is this study's choice, not the
paper's, and the share of requests follows the groups' sizes.

Writes DIR/explore.jsonl and DIR/sealed.jsonl (R = 1) or DIR/sealed_1.jsonl ... DIR/sealed_R.jsonl,
and prints the groups, requests and sha256 of each. Exits 1, writing nothing, on a missing or
duplicate id, a missing group field, an empty file, a prompt that appears in two files (the same
text or token ids and images under different ids), or an existing output file without --force.
"""
import argparse
import hashlib
import json
import os
import sys


def hashed(salt: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big")


def in_sealed(request_id: str, salt: str, fraction: float) -> bool:
    return hashed(salt, request_id) < fraction * 2**64


def content_key(row):
    prompt = row.get("input_ids") if row.get("input_ids") is not None else row.get("text")
    return hashlib.sha256(json.dumps([prompt, row.get("image_data")], sort_keys=True).encode()).hexdigest()


def split(lines, salt: str, fraction: float, id_field: str = "id", group_field: str = None, parts: int = 1):
    """Return (explore, [sealed part 1, ...]) lists of raw lines; raise ValueError on bad input."""
    seen = set()
    explore, sealed = [], [[] for _ in range(parts)]
    owner = {}
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
        if in_sealed(key, salt, fraction):
            target = 1 + hashed(f"{salt}:part", key) % parts
        else:
            target = 0
        if row.get("input_ids") is not None or row.get("text") is not None:
            content = content_key(row)
            if owner.setdefault(content, target) != target:
                raise ValueError(f"line {number}: the prompt of {request_id!r} also appears in another file")
        (explore if target == 0 else sealed[target - 1]).append(line.rstrip("\n") + "\n")
    return explore, sealed


def groups(rows, id_field, group_field):
    return len({str(json.loads(r).get(group_field or id_field)) for r in rows})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("requests")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--salt", required=True, help="a name for this study; the same salt gives the same split")
    parser.add_argument("--sealed-fraction", type=float, default=0.3)
    parser.add_argument("--sealed-parts", type=int, default=1, help="disjoint sealed parts, one per round")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--group-field", default=None, help="requests that share this field go to the same file")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()
    with open(args.requests) as f:
        try:
            explore, sealed = split(f, args.salt, args.sealed_fraction, args.id_field, args.group_field,
                                    args.sealed_parts)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
    names = ["sealed"] if args.sealed_parts == 1 else [f"sealed_{i}" for i in range(1, args.sealed_parts + 1)]
    outputs = [("explore", explore)] + list(zip(names, sealed))
    empty = [name for name, rows in outputs if not rows]
    if empty:
        print(f"empty files: {empty}; change the fraction, the parts or the salt", file=sys.stderr)
        return 1
    paths = [os.path.join(args.out_dir, f"{name}.jsonl") for name, _ in outputs]
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not args.force:
        print(f"{existing} exist; a split is written once (--force to replace it)", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    for (name, rows), path in zip(outputs, paths):
        data = "".join(rows).encode()
        with open(path, "wb") as f:
            f.write(data)
        print(f"{name}: {groups(rows, args.id_field, args.group_field)} groups, {len(rows)} requests, "
              f"sha256 {hashlib.sha256(data).hexdigest()}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
