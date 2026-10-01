#!/usr/bin/env python3
"""Build PROJECT_EXPORT.md: every project file in its own labeled code fence, with sha256 checksums.
Run from the repo root:  python tools/build_export.py
Restore later with:      python tools/restore_from_export.py PROJECT_EXPORT.md <target_dir>
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = [  # (heading, [paths])
    ("File 1: ARCHITECTURE_AND_SPECS.md", ["ARCHITECTURE_AND_SPECS.md"]),
    ("File 2: pipeline.py", ["pipeline.py"]),
    ("File 3: github-actions-ingest.yml (lives at .github/workflows/ingest.yml)", [".github/workflows/ingest.yml"]),
    ("File 4: Raspberry Pi setup script and systemd units",
     ["pi/setup_ramdisk.sh", "pi/moneyball.service", "pi/moneyball.timer"]),
    ("File 5: colab/analysis.py", ["colab/analysis.py"]),
    ("File 6: tests/test_pipeline.py", ["tests/test_pipeline.py"]),
    ("Supporting files", ["tests/make_synthetic.py", "requirements.txt", ".gitignore", "PLAN.md", "RUNBOOK.md",
                          "tools/build_export.py", "tools/restore_from_export.py"]),
]
LANG = {".py": "python", ".yml": "yaml", ".sh": "bash", ".md": "markdown", ".service": "ini", ".timer": "ini", ".txt": "text"}

RESUME = """## Resume prompt for a new chat

Paste this, then attach or paste the files from this export:

> I am continuing a project called "Moneyballing Hackathons". Read ARCHITECTURE_AND_SPECS.md first, then RUNBOOK.md.
> The code is already written and tested offline (18 tests). Nothing has run against live Devpost or Hugging Face yet.
> Do not redesign the architecture. Help me execute the runbook, starting with the preflight checks (canary, then
> probing a real winning hackathon to confirm the winner-badge selector), and help me interpret ledger status
> output, circuit-breaker trips, and the Colab results.
"""


def fence_for(text: str) -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def main():
    out = ["# Moneyballing Hackathons: complete project export", "",
           "Every file below is the exact content of the real file. Restore all of them with",
           "`python tools/restore_from_export.py PROJECT_EXPORT.md <target_dir>` (it checks every sha256).", "",
           RESUME, "## Manifest", "", "| path | lines | sha256 |", "|---|---|---|"]
    bodies = []
    for heading, paths in SECTIONS:
        bodies += ["", f"## {heading}", ""]
        for rel in paths:
            data = (ROOT / rel).read_bytes()
            text = data.decode("utf-8")
            if not text.endswith("\n"):
                raise SystemExit(f"{rel} must end with a newline")
            sha = hashlib.sha256(data).hexdigest()
            out.append(f"| `{rel}` | {text.count(chr(10))} | `{sha[:16]}` |")
            fence = fence_for(text)
            lang = LANG.get(Path(rel).suffix, "")
            bodies += [f"### `{rel}`", "", f"<!-- FILE: {rel} sha256={sha} -->", f"{fence}{lang}", text + fence, ""]
    (ROOT / "PROJECT_EXPORT.md").write_text("\n".join(out + bodies), encoding="utf-8")
    print("wrote PROJECT_EXPORT.md")


if __name__ == "__main__":
    sys.exit(main())
