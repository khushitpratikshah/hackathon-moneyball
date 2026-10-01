#!/usr/bin/env python3
"""Rebuild every file from PROJECT_EXPORT.md and verify its sha256.
Usage: python tools/restore_from_export.py PROJECT_EXPORT.md <target_dir>
"""
import hashlib
import re
import stat
import sys
from pathlib import Path

MARK = re.compile(r"^<!-- FILE: (?P<path>\S+) sha256=(?P<sha>[0-9a-f]{64}) -->$")


def main(export: str, target: str):
    lines = Path(export).read_text(encoding="utf-8").split("\n")
    target = Path(target)
    i, restored = 0, 0
    while i < len(lines):
        m = MARK.match(lines[i])
        if not m:
            i += 1
            continue
        opener = re.match(r"^(`{3,})\w*$", lines[i + 1])
        if not opener:
            sys.exit(f"missing code fence after marker for {m['path']}")
        fence, j, body = opener.group(1), i + 2, []
        while lines[j] != fence:
            body.append(lines[j])
            j += 1
        data = ("\n".join(body) + "\n").encode("utf-8")
        if hashlib.sha256(data).hexdigest() != m["sha"]:
            sys.exit(f"CHECKSUM MISMATCH for {m['path']}")
        dest = target / m["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if dest.suffix == ".sh" or dest.name.endswith("_export.py"):
            dest.chmod(dest.stat().st_mode | stat.S_IXUSR)
        restored += 1
        i = j + 1
    print(f"restored {restored} files into {target}, all checksums match")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
