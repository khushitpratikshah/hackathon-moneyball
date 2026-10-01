#!/usr/bin/env python3
"""moneyball_ops.py: setup and operations companion for the Moneyballing Hackathons pipeline.

One file, four jobs:

  init-pi        wizard for a Raspberry Pi 5: RAM disk, code, venv, secrets file, systemd units
  pair-secrets   checks your Hugging Face token and repo, writes /etc/moneyball.env, sets GitHub secrets
  probe-studio   try CSS selectors on a real gallery page, then save the winner selector once it matches
  dashboard      live view of the ledger: progress, per-worker rates, poisoned keys, circuit-breaker alerts

Support commands:

  canary         is this IP (or GitHub Actions) being blocked right now
  timer          enable, disable or show the nightly Pi timer (enable is gated on canary and probe)
  selftest       offline checks of this tool itself, no network and no root needed

INSTALL (macOS or Linux, Python 3.10 or newer; Windows is not supported)

  python3 -m venv ~/.venvs/moneyball_ops && source ~/.venvs/moneyball_ops/bin/activate
  pip install rich huggingface_hub pyarrow selectolax curl_cffi

  What each mode needs:
    init-pi        rich (plus apt, sudo, systemd on the Pi)
    pair-secrets   rich, huggingface_hub; optional GitHub CLI `gh` logged in (gh auth login)
    probe-studio   rich, selectolax, curl_cffi; uses pipeline.py for a parity check when it can find it
    dashboard      rich, huggingface_hub, pyarrow; optional `gh` for Actions run annotations
    selftest       all of the above except gh

  On the Pi, after init-pi has built the venv you can run everything with:
    /opt/moneyball/.venv/bin/python moneyball_ops.py <command>

EXAMPLES

  python moneyball_ops.py selftest
  python moneyball_ops.py init-pi --repo-url https://github.com/YOU/devpost-moneyball
  python moneyball_ops.py init-pi --dry-run                 # show every change, make none
  python moneyball_ops.py pair-secrets --gh-repo YOU/devpost-moneyball
  python moneyball_ops.py pair-secrets --check-read-token   # also test the Colab read token
  python moneyball_ops.py canary                            # this machine
  python moneyball_ops.py canary --gha 3                    # three GitHub Actions canaries
  python moneyball_ops.py probe-studio --suggest            # list big finished hackathons to probe
  python moneyball_ops.py probe-studio https://SOME-HACKATHON.devpost.com/project-gallery
  python moneyball_ops.py probe-studio --html saved_gallery.html --expect 3 --selector '.winner'
  python moneyball_ops.py timer enable
  python moneyball_ops.py dashboard                         # live view, keys: r d p q
  python moneyball_ops.py dashboard --once                  # one snapshot, exit code 0, 2 or 3
  python moneyball_ops.py dashboard --local /path/to/out    # read a local pipeline --local-only folder

Secrets: the Hugging Face token is read without echo, never put on a command line, never printed, and
only written to the root-owned 0600 file /etc/moneyball.env and to GitHub secrets (through stdin).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import getpass
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

sys.dont_write_bytecode = True  # keep .pyc files off the SD card

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markup import escape as esc_markup
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover
    sys.exit("moneyball_ops needs the 'rich' package: pip install rich")

VERSION = "1.0"
MOUNT = "/mnt/moneyball-ram"
APP_DIR = "/opt/moneyball"
ENV_FILE = "/etc/moneyball.env"
FSTAB = "/etc/fstab"
UNIT_DIR = "/etc/systemd/system"
DEVPOST_API = "https://devpost.com/api/hackathons"
PLACEHOLDER_TOKEN = "PASTE_THE_WRITE_TOKEN_HERE"
PLACEHOLDER_REPO = "YOUR_HF_USER/hackathon-moneyball"
STAGES = ("discover", "gallery", "detail")
FINAL = {"ok", "empty", "gone"}
UTC = dt.timezone.utc

console = Console(highlight=False)


def esc(x: Any) -> str:
    return esc_markup(str(x))


class OpsError(Exception):
    """A step failed in a way the user needs to read about. Message is shown as is."""


# ----------------------------------------------------------------------------- small utilities
def conf_dir() -> Path:
    return Path(os.environ.get("MONEYBALL_OPS_HOME") or (Path.home() / ".config" / "moneyball_ops"))


def load_state() -> dict:
    try:
        return json.loads((conf_dir() / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def save_state(update: dict) -> dict:
    """Merge update into state.json. Holds no secrets, only repo names, timestamps and probe results."""
    st = load_state()
    st.update(update)
    d = conf_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(st, f, indent=2, sort_keys=True)
    os.replace(tmp, d / "state.json")
    return st


def utcnow() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso(t: Optional[dt.datetime] = None) -> str:
    return (t or utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def shard_of(key: Any, n: int) -> int:
    """Same function as pipeline.py, so shard numbers here match what the workers do."""
    return int(sha1(str(key))[:8], 16) % n


def fmt_age(sec: Optional[float]) -> str:
    if sec is None:
        return "never"
    sec = int(max(sec, 0))
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    if sec < 172800:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def parse_size(s: str, mem_total: int) -> int:
    """tmpfs size such as 1g, 512m, 300000k, 2097152 or 25%. Returns bytes."""
    m = re.fullmatch(r"\s*(\d+)\s*([kKmMgG%]?)\s*", s or "")
    if not m:
        raise OpsError(f"cannot read size '{s}'. Use something like 1g or 512m")
    n, u = int(m.group(1)), m.group(2).lower()
    if u == "%":
        return mem_total * n // 100
    return n * {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[u]


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return str(n)


STATUS_STYLE = {"ok": ("OK", "green"), "changed": ("CHANGED", "cyan"), "skip": ("SKIP", "dim"),
                "warn": ("WARN", "yellow"), "fail": ("FAIL", "bold red")}


class Report:
    """Prints each step result as it happens and keeps them for the closing summary."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, step: str, status: str, detail: str = "") -> None:
        self.rows.append((step, status, detail))
        label, style = STATUS_STYLE[status]
        console.print(f"  [{style}]{label:<8}[/] {esc(step)}" + (f"  [dim]{esc(detail)}[/]" if detail else ""))

    def worst(self) -> str:
        order = ["ok", "skip", "changed", "warn", "fail"]
        return max((r[1] for r in self.rows), key=order.index, default="ok")

    def table(self, title: str) -> Table:
        t = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
        t.add_column("step")
        t.add_column("result")
        t.add_column("detail", overflow="fold")
        for step, status, detail in self.rows:
            label, style = STATUS_STYLE[status]
            t.add_row(esc(step), f"[{style}]{label}[/]", esc(detail))
        return t


# ----------------------------------------------------------------------------- machine access wrapper
class Sys:
    """Every change to the machine goes through this class, so --dry-run prints instead of acting and
    --sandbox-root redirects absolute paths into a scratch folder (system commands are skipped)."""

    def __init__(self, dry: bool = False, root: Optional[str] = None, yes: bool = False) -> None:
        self.dry, self.yes = dry, yes
        self.root = Path(root).resolve() if root else None

    @property
    def sandbox(self) -> bool:
        return self.root is not None

    def p(self, path: Any) -> Path:
        s = str(path)
        if self.root is not None and s.startswith("/"):
            return self.root / s.lstrip("/")
        return Path(s)

    def need_sudo(self) -> bool:
        return not self.sandbox and os.geteuid() != 0

    def direct(self) -> bool:
        """True when this process can write system paths itself (root or sandbox)."""
        return self.sandbox or os.geteuid() == 0

    def run(self, argv: list[str], *, priv: bool = False, input: Optional[str] = None, check: bool = True,
            system: bool = False, readonly: bool = False, timeout: int = 600,
            env: Optional[dict] = None, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        shown = " ".join(shlex.quote(x) for x in argv)
        if priv and self.need_sudo():
            argv = ["sudo"] + list(argv)
        if self.dry and not readonly:
            console.print(f"    [dim]dry-run: {esc(shown)}[/]")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if self.sandbox and system:
            if not readonly:
                console.print(f"    [dim]sandbox: skipped {esc(shown)}[/]")
            return subprocess.CompletedProcess(argv, 0 if not readonly else 1, "", "")
        try:
            cp = subprocess.run(argv, input=input, capture_output=True, text=True, timeout=timeout,
                                env=env, cwd=cwd)
        except FileNotFoundError:
            raise OpsError(f"command not found: {argv[0]}")
        except subprocess.TimeoutExpired:
            raise OpsError(f"timed out after {timeout}s: {shown}")
        if check and cp.returncode != 0:
            raise OpsError(f"{shown}\n{(cp.stderr or cp.stdout).strip()[-600:]}")
        return cp

    def exists(self, path: Any) -> bool:
        return self.p(path).exists()

    def read_text(self, path: Any) -> Optional[str]:
        t = self.p(path)
        if not t.exists():
            return None
        try:
            return t.read_text()
        except PermissionError:
            return self.run(["cat", str(path)], priv=True, readonly=True).stdout

    def mkdir(self, path: Any, owner: Optional[str] = None, mode: int = 0o755) -> None:
        if self.dry:
            console.print(f"    [dim]dry-run: mkdir -p {esc(path)}[/]")
            return
        if self.direct():
            t = self.p(path)
            t.mkdir(parents=True, exist_ok=True)
            os.chmod(t, mode)
            if owner and not self.sandbox:
                shutil.chown(t, user=owner)
        else:
            self.run(["mkdir", "-p", str(path)], priv=True)
            if owner:
                self.run(["chown", f"{owner}:", str(path)], priv=True)

    def write_text(self, path: Any, content: str, mode: int = 0o644, secret: bool = False) -> None:
        """Atomic write (temp file then rename). Created with the final mode, so a secret file is never
        briefly world readable."""
        if self.dry:
            what = "contents hidden" if secret else f"{len(content)} bytes"
            console.print(f"    [dim]dry-run: write {esc(path)} mode {mode:04o} ({what})[/]")
            return
        if self.direct():
            t = self.p(path)
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_name(t.name + ".moneyball-new")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.chmod(tmp, mode)
            os.replace(tmp, t)
            return
        tmp = f"{path}.moneyball-new"
        self.run(["sh", "-c", 'umask 077; cat > "$1"', "sh", tmp], priv=True, input=content)
        self.run(["chmod", f"{mode:04o}", tmp], priv=True)
        self.run(["chown", "root:root", tmp], priv=True)
        self.run(["mv", "-f", tmp, str(path)], priv=True)

    def chmod_root(self, path: Any, mode: int) -> None:
        if self.dry:
            console.print(f"    [dim]dry-run: chown root:root + chmod {mode:04o} {esc(path)}[/]")
            return
        if self.sandbox:
            os.chmod(self.p(path), mode)
            return
        self.run(["chown", "root:root", str(path)], priv=True)
        self.run(["chmod", f"{mode:04o}", str(path)], priv=True)

    def copy_file(self, src: Any, dst: Any) -> None:
        if self.dry:
            console.print(f"    [dim]dry-run: cp -p {esc(src)} {esc(dst)}[/]")
        elif self.direct():
            shutil.copy2(self.p(src), self.p(dst))
        else:
            self.run(["cp", "-p", str(src), str(dst)], priv=True)

    def ensure_sudo(self) -> None:
        if self.dry or not self.need_sudo():
            return
        if not shutil.which("sudo"):
            raise OpsError("sudo is required (or run this as root with --user NAME)")
        console.print("[dim]Asking for sudo once so later steps do not stop to prompt.[/]")
        if subprocess.run(["sudo", "-v"]).returncode != 0:
            raise OpsError("sudo was not granted")


# ----------------------------------------------------------------------------- env file, fstab, units
def env_parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s[0] in "#;" or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] == "'":
            v = v[1:-1]
        elif len(v) >= 2 and v[0] == v[-1] == '"':
            v = re.sub(r'\\(["\\])', r"\1", v[1:-1])
        out[k] = v
    return out


def env_quote(v: str) -> str:
    """Quote a value for a systemd EnvironmentFile line. Plain values stay plain."""
    if "\n" in v or "\r" in v or "\x00" in v:
        raise OpsError("value contains a newline or NUL, which an environment file cannot hold")
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=,\-]*", v):
        return v
    if "'" not in v:
        return f"'{v}'"
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def env_update(text: str, updates: dict[str, str]) -> str:
    """Replace or append KEY=value lines, keeping every other line and comment as it was."""
    pending = dict(updates)
    out: list[str] = []
    for ln in text.splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", ln)
        if m and m.group(1) in updates:
            k = m.group(1)
            if k in pending:
                out.append(f"{k}={env_quote(pending.pop(k))}")
            continue  # a repeated key is dropped
        out.append(ln)
    for k, v in pending.items():
        out.append(f"{k}={env_quote(v)}")
    return "\n".join(out).rstrip("\n") + "\n"


ENV_TEMPLATE = f"HF_REPO={PLACEHOLDER_REPO}\nHF_TOKEN={PLACEHOLDER_TOKEN}\nDELAY=2.5\nWINNER_CSS=\n"


def env_set(sysm: Sys, updates: dict[str, str], path: str = ENV_FILE, secret: bool = False) -> None:
    cur = sysm.read_text(path)
    new = env_update(cur if cur is not None else ENV_TEMPLATE, updates)
    sysm.write_text(path, new, mode=0o600, secret=secret)


def fstab_line(mount: str, size: str, uid: int, gid: int) -> str:
    return f"tmpfs {mount} tmpfs rw,nosuid,nodev,noexec,noatime,size={size},mode=0700,uid={uid},gid={gid} 0 0"


def fstab_with_ramdisk(text: str, mount: str, line: str) -> tuple[str, str]:
    """Return (new_text, action). action is unchanged, replaced or appended. Refuses to touch an entry
    that mounts something other than tmpfs at the same place."""
    want = line.split()
    out: list[str] = []
    seen = False
    action = "unchanged"
    for ln in text.splitlines():
        s = ln.strip()
        f = s.split()
        if s and not s.startswith("#") and len(f) >= 3 and f[1] == mount:
            if f[2] != "tmpfs":
                raise OpsError(f"{FSTAB} already mounts {f[2]} at {mount}. Not touching it.")
            if seen:
                out.append("#moneyball_ops duplicate: " + ln)
                action = "replaced"
                continue
            seen = True
            if f == want:
                out.append(ln)
            else:
                out.append(line)
                action = "replaced"
            continue
        out.append(ln)
    if not seen:
        out += ["# moneyball_ops: RAM disk for the scraper (everything it writes lives here)", line]
        action = "appended"
    return "\n".join(out).rstrip("\n") + "\n", action


SERVICE_TMPL = """# /etc/systemd/system/moneyball.service
# The Pi owns shard @SHARD@ of @NSHARDS@. Everything the job writes lands on the tmpfs at @MOUNT@.
# Code and the venv are only read from disk.
[Unit]
Description=Moneyball Devpost ingestion burst (shard @SHARD@)
Wants=network-online.target
After=network-online.target
RequiresMountsFor=@MOUNT@

[Service]
Type=oneshot
User=@USER@
WorkingDirectory=@APP@
EnvironmentFile=/etc/moneyball.env
Environment=WORKER_ID=pi5
Environment=WORKDIR=@MOUNT@/work
Environment=HF_HOME=@MOUNT@/hf_home
Environment=XDG_CACHE_HOME=@MOUNT@/cache
Environment=TMPDIR=@MOUNT@/tmp
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStartPre=/usr/bin/mkdir -p @MOUNT@/work @MOUNT@/hf_home @MOUNT@/cache @MOUNT@/tmp
ExecStart=@APP@/.venv/bin/python pipeline.py run --shard @SHARD@ --nshards @NSHARDS@ --max-minutes @MAXMIN@@EXTRA@
ExecStopPost=/bin/sh -c 'rm -rf @MOUNT@/work @MOUNT@/hf_home @MOUNT@/cache @MOUNT@/tmp'
TimeoutStartSec=@TIMEOUT@
Nice=10
MemoryMax=2G
"""

TIMER_TMPL = """# /etc/systemd/system/moneyball.timer
# Enable with: sudo systemctl enable --now moneyball.timer
# /etc/moneyball.env (chmod 600, owner root) holds:
#   HF_REPO=youruser/hackathon-moneyball
#   HF_TOKEN=hf_xxx
[Unit]
Description=Run the moneyball burst nightly

[Timer]
OnCalendar=*-*-* @ONCAL@
RandomizedDelaySec=20min
Persistent=true

[Install]
WantedBy=timers.target
"""


def render_service(user: str, *, app: str = APP_DIR, mount: str = MOUNT, shard: int = 4, nshards: int = 5,
                   max_minutes: int = 180, min_registrations: int = 0) -> str:
    extra = f" --min-registrations {min_registrations}" if min_registrations else ""
    timeout = f"{math.ceil(max_minutes / 60) + 1}h"
    rep = {"@SHARD@": str(shard), "@NSHARDS@": str(nshards), "@MOUNT@": mount, "@USER@": user, "@APP@": app,
           "@MAXMIN@": str(max_minutes), "@EXTRA@": extra, "@TIMEOUT@": timeout}
    s = SERVICE_TMPL
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def render_timer(oncalendar: str = "01:40:00") -> str:
    if not re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", oncalendar):
        raise OpsError("timer time must look like 01:40 or 01:40:00")
    if len(oncalendar) == 5:
        oncalendar += ":00"
    return TIMER_TMPL.replace("@ONCAL@", oncalendar)


# ----------------------------------------------------------------------------- inspecting the machine
def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").replace("\x00", "").strip()
    except OSError:
        return ""


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for ln in _read("/proc/meminfo").splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", ln)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def os_pretty() -> str:
    for ln in _read("/etc/os-release").splitlines():
        if ln.startswith("PRETTY_NAME="):
            return ln.split("=", 1)[1].strip('"')
    return platform.platform()


def inspect_system() -> dict:
    mi = meminfo()
    model = _read("/proc/device-tree/model")
    return {
        "machine": platform.machine(), "model": model, "os": os_pretty(),
        "python": platform.python_version(), "mem_total": mi.get("MemTotal", 0),
        "mem_avail": mi.get("MemAvailable", 0), "swap_total": mi.get("SwapTotal", 0),
        "systemd": Path("/run/systemd/system").is_dir(), "apt": bool(shutil.which("apt-get")),
        "sudo": bool(shutil.which("sudo")), "git": bool(shutil.which("git")),
        "is_pi": "raspberry pi" in model.lower(), "is_pi5": "raspberry pi 5" in model.lower(),
        "kernel": platform.release(),
    }


def default_user() -> str:
    if os.geteuid() == 0:
        u = os.environ.get("SUDO_USER")
        if not u or u == "root":
            raise OpsError("running as root: pass --user NAME for the account that will own the code and run the service")
        return u
    return getpass.getuser()


def user_ids(user: str, sandbox: bool) -> tuple[int, int, Path]:
    if sandbox:
        return os.getuid(), os.getgid(), Path("/home") / user
    import pwd
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        raise OpsError(f"no such user: {user}")
    return pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)


def is_pi_like() -> bool:
    return platform.system() == "Linux" and (Path(ENV_FILE).exists() or "raspberry pi" in _read("/proc/device-tree/model").lower())


# ----------------------------------------------------------------------------- init-pi
SHELL_HELPER = """export PYTHONDONTWRITEBYTECODE=1
export WORKDIR=@MOUNT@/work
export HF_HOME=@MOUNT@/hf_home
export XDG_CACHE_HOME=@MOUNT@/cache
export TMPDIR=@MOUNT@/tmp
mkdir -p "$WORKDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
cd @APP@
"""


def find_source(a: argparse.Namespace) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for c in ([Path(a.src)] if a.src else []) + [here, here.parent]:
        if (c / "pipeline.py").is_file():
            return c
    return None


def step_system(rep: Report, info: dict, size: str, lenient: bool = False) -> int:
    rep.add("CPU architecture", "ok" if info["machine"] in ("aarch64", "arm64") else "warn",
            f"{info['machine']}" + ("" if info["machine"] in ("aarch64", "arm64") else " (the runbook assumes a 64 bit Pi)"))
    rep.add("Board", "ok" if info["is_pi5"] else "warn", info["model"] or "not a Raspberry Pi (fine for a dry run or a test machine)")
    if info["systemd"]:
        rep.add("OS", "ok", info["os"])
    elif lenient:
        rep.add("OS", "warn", info["os"] + " (systemd is not running here, fine for a dry run or sandbox)")
    else:
        rep.add("OS", "fail", info["os"] + " (systemd is not running, the units cannot work)")
    py_ok = tuple(int(x) for x in info["python"].split(".")[:2]) >= (3, 10)
    rep.add("Python", "ok" if py_ok else "fail", info["python"] + ("" if py_ok else " (3.10 or newer needed)"))
    mt, ma = info["mem_total"], info["mem_avail"]
    rep.add("RAM", "ok" if mt else "warn", f"{human_bytes(mt)} total, {human_bytes(ma)} available" if mt else "could not read /proc/meminfo")
    want = parse_size(size, mt or 4 * 1024 ** 3)
    if mt:
        if want > ma:
            rep.add("RAM disk size", "fail", f"{size} is more than the {human_bytes(ma)} available right now")
            raise OpsError("RAM disk size exceeds available memory")
        if want > mt * 0.5:
            rep.add("RAM disk size", "warn", f"{size} is over half of RAM. The service also has MemoryMax=2G.")
        else:
            rep.add("RAM disk size", "ok", f"{size} ceiling (RAM is only used as files accumulate)")
        if mt < 3.5 * 1024 ** 3:
            rep.add("Memory headroom", "warn", "under 4 GB: the service sets MemoryMax=2G and the RAM disk shares that memory")
    return want


def step_packages(sysm: Sys, rep: Report, info: dict, skip: bool) -> None:
    if skip:
        rep.add("Packages", "skip", "--skip-packages")
        return
    venv_ok = sysm.sandbox or (info["apt"] and sysm.run(["dpkg", "-s", "python3-venv"], check=False, readonly=True).returncode == 0) \
        or (not info["apt"] and sysm.run([sys.executable, "-c", "import ensurepip"], check=False, readonly=True).returncode == 0)
    if venv_ok and info["git"]:
        rep.add("Packages (python3-venv, git)", "ok", "already installed")
        return
    if not info["apt"]:
        rep.add("Packages (python3-venv, git)", "warn", "no apt-get here: install python3-venv and git yourself")
        return
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    with console.status("apt-get update and install ..."):
        sysm.run(["apt-get", "update", "-qq"], priv=True, system=True, env=env)
        sysm.run(["apt-get", "install", "-y", "-qq", "python3-venv", "git"], priv=True, system=True, env=env)
    rep.add("Packages (python3-venv, git)", "changed", "installed")


def fstab_verify(sysm: Sys) -> tuple[int, str]:
    cp = sysm.run(["findmnt", "--verify", "--tab-file", FSTAB], check=False, system=True, readonly=True)
    return cp.returncode, (cp.stdout + cp.stderr).strip()


def step_ramdisk(sysm: Sys, rep: Report, user: str, size: str, args: argparse.Namespace) -> None:
    uid, gid, _ = user_ids(user, sysm.sandbox)
    line = fstab_line(MOUNT, size, uid, gid)
    cur = sysm.read_text(FSTAB)
    if cur is None:
        if not sysm.sandbox:
            raise OpsError(f"{FSTAB} not found")
        cur = "# sandbox fstab\n"
    new, action = fstab_with_ramdisk(cur, MOUNT, line)
    if action == "unchanged":
        rep.add(f"{FSTAB} entry", "ok", "already correct")
    else:
        before_rc, _ = fstab_verify(sysm)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        backup = f"{FSTAB}.moneyball-bak-{stamp}"
        if sysm.exists(FSTAB):
            sysm.copy_file(FSTAB, backup)
        sysm.write_text(FSTAB, new, mode=0o644)
        after_rc, out = fstab_verify(sysm)
        if before_rc == 0 and after_rc not in (0, 64) and not sysm.dry:
            sysm.copy_file(backup, FSTAB)
            raise OpsError(f"findmnt --verify rejected the new fstab, original restored from {backup}\n{out}")
        rep.add(f"{FSTAB} entry", "changed", f"{action}; original saved as {backup}")
    if not sysm.exists(MOUNT):
        sysm.mkdir(MOUNT, mode=0o755)
    sysm.run(["systemctl", "daemon-reload"], priv=True, system=True)
    mounted = sysm.run(["findmnt", "-n", "-o", "FSTYPE", MOUNT], check=False, system=True, readonly=True)
    if mounted.returncode == 0 and mounted.stdout.strip():
        sysm.run(["mount", "-o", f"remount,size={size}", MOUNT], priv=True, system=True)
        how = "remounted"
    else:
        sysm.run(["mount", MOUNT], priv=True, system=True)
        how = "mounted"
    if sysm.sandbox:
        how = "mount skipped (sandbox)"
    if not sysm.sandbox and not sysm.dry:
        fs = sysm.run(["findmnt", "-n", "-o", "FSTYPE", MOUNT], check=False, readonly=True).stdout.strip()
        if fs != "tmpfs":
            raise OpsError(f"{MOUNT} is not a tmpfs after mounting (found '{fs or 'nothing'}')")
        probe = Path(MOUNT) / ".moneyball_ops_probe"
        try:
            probe.write_text("x")
            probe.unlink()
        except OSError as e:
            raise OpsError(f"{MOUNT} is not writable by {user}: {e}")
    rep.add(f"tmpfs {MOUNT}", "changed" if action != "unchanged" or how == "mounted" else "ok", f"{how}, size ceiling {size}")
    if args.volatile_journal:
        sysm.write_text("/etc/systemd/journald.conf.d/volatile.conf", "[Journal]\nStorage=volatile\nRuntimeMaxUse=50M\n")
        sysm.run(["systemctl", "restart", "systemd-journald"], priv=True, system=True)
        rep.add("Journal in RAM", "changed", "logs vanish on reboot")
    if args.disable_swap:
        sysm.run(["systemctl", "disable", "--now", "dphys-swapfile"], priv=True, system=True, check=False)
        rep.add("Swap off", "changed", "dphys-swapfile disabled")


def step_code(sysm: Sys, rep: Report, user: str, args: argparse.Namespace) -> Path:
    app = Path(args.app_dir)
    real_app = sysm.p(app)
    if (real_app / "pipeline.py").is_file():
        if args.update and (real_app / ".git").is_dir() and not sysm.sandbox:
            cp = sysm.run(["git", "-C", str(app), "pull", "--ff-only"], check=False)
            rep.add(f"Code at {app}", "changed" if "Already up to date" not in cp.stdout else "ok",
                    (cp.stdout or cp.stderr).strip().splitlines()[-1] if (cp.stdout or cp.stderr).strip() else "")
        else:
            rep.add(f"Code at {app}", "ok", "already present")
        return real_app
    sysm.mkdir(app, owner=user)
    src = find_source(args)
    if src:
        if sysm.dry:
            console.print(f"    [dim]dry-run: copy {esc(src)} to {esc(app)}[/]")
        else:
            shutil.copytree(src, real_app, symlinks=True, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".pytest_cache", "out", "work", "hf", "probe"))
        rep.add(f"Code at {app}", "changed", f"copied from {src}")
    elif args.repo_url:
        sysm.run(["git", "clone", "--depth", "1", args.repo_url, str(app)], timeout=300)
        rep.add(f"Code at {app}", "changed", f"cloned {args.repo_url}")
    else:
        raise OpsError("no code to install. Run this from inside a checkout, or pass --src PATH or --repo-url URL")
    return real_app


def step_venv(sysm: Sys, rep: Report, real_app: Path, args: argparse.Namespace) -> None:
    if args.skip_venv:
        rep.add("Virtualenv and dependencies", "skip", "--skip-venv")
        return
    if not (real_app / "requirements.txt").is_file() and not sysm.dry:
        raise OpsError(f"{real_app}/requirements.txt is missing")
    py = real_app / ".venv" / "bin" / "python"
    with console.status("creating venv and installing dependencies (a few minutes on a Pi) ..."):
        if not py.exists():
            sysm.run([sys.executable, "-m", "venv", str(real_app / ".venv")], cwd=str(real_app))
        sysm.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout=900)
        sysm.run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt", "pytest", "rich"],
                 cwd=str(real_app), timeout=1800)
    cp = sysm.run([str(py), "-c", "import curl_cffi, selectolax, pyarrow, huggingface_hub; print('imports ok')"], check=False)
    if cp.returncode != 0 and not sysm.dry:
        raise OpsError("dependency import check failed (curl_cffi is the usual culprit on a Pi):\n" + (cp.stderr or "").strip()[-500:])
    rep.add("Virtualenv and dependencies", "changed", "imports ok")


def step_shell_helper(sysm: Sys, rep: Report, home: Path, args: argparse.Namespace) -> None:
    content = SHELL_HELPER.replace("@MOUNT@", MOUNT).replace("@APP@", args.app_dir)
    t = sysm.p(home / ".moneyball_shell")
    cur = t.read_text() if t.exists() else None
    if cur == content:
        rep.add("~/.moneyball_shell", "ok", "already current")
        return
    if sysm.dry:
        console.print(f"    [dim]dry-run: write {esc(t)}[/]")
    else:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content)
    rep.add("~/.moneyball_shell", "changed", "keeps hand-run commands writing to the RAM disk")


def step_env(sysm: Sys, rep: Report) -> bool:
    """Create /etc/moneyball.env with placeholders if missing; always enforce root 0600.
    Returns True when the file still holds placeholder values."""
    cur = sysm.read_text(ENV_FILE)
    if cur is None:
        sysm.write_text(ENV_FILE, ENV_TEMPLATE, mode=0o600)
        rep.add(ENV_FILE, "changed", "created with placeholders (mode 0600, root)")
        cur = ENV_TEMPLATE
    else:
        sysm.chmod_root(ENV_FILE, 0o600)
        rep.add(ENV_FILE, "ok", "exists; owner and mode enforced to root 0600")
    vals = env_parse(cur)
    return vals.get("HF_TOKEN", PLACEHOLDER_TOKEN) in ("", PLACEHOLDER_TOKEN) or vals.get("HF_REPO", PLACEHOLDER_REPO) in ("", PLACEHOLDER_REPO)


def service_state(sysm: Sys) -> dict[str, str]:
    out: dict[str, str] = {}
    for unit, props in (("moneyball.service", "ActiveState,SubState,Result,UnitFileState"),
                        ("moneyball.timer", "ActiveState,UnitFileState,NextElapseUSecRealtime")):
        cp = sysm.run(["systemctl", "show", "-p", props, unit], check=False, system=True, readonly=True)
        for ln in cp.stdout.splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[f"{unit.split('.')[1]}.{k}"] = v
    return out


def step_units(sysm: Sys, rep: Report, user: str, args: argparse.Namespace) -> None:
    svc = render_service(user, app=args.app_dir, shard=args.shard, nshards=args.nshards,
                         max_minutes=args.max_minutes, min_registrations=args.min_registrations)
    tmr = render_timer(args.timer_time)
    changed = []
    for name, content in (("moneyball.service", svc), ("moneyball.timer", tmr)):
        path = f"{UNIT_DIR}/{name}"
        if sysm.read_text(path) == content:
            continue
        sysm.write_text(path, content, mode=0o644)
        changed.append(name)
    sysm.run(["systemctl", "daemon-reload"], priv=True, system=True)
    rep.add("systemd units", "changed" if changed else "ok",
            ("wrote " + ", ".join(changed)) if changed else "already current")
    v = sysm.run(["systemd-analyze", "verify", f"{UNIT_DIR}/moneyball.service"], check=False, system=True, readonly=True)
    if sysm.sandbox:
        rep.add("systemd-analyze verify", "skip", "sandbox")
    elif v.returncode == 0 and not v.stderr.strip():
        rep.add("systemd-analyze verify", "ok", "unit ok")
    else:
        rep.add("systemd-analyze verify", "warn", (v.stderr or v.stdout).strip()[-300:] or f"exit {v.returncode}")
    st = service_state(sysm)
    if sysm.sandbox or sysm.dry:
        rep.add("Service status", "skip", "sandbox or dry run")
        return
    svc_line = f"{st.get('service.ActiveState', '?')}/{st.get('service.SubState', '?')}, file {st.get('service.UnitFileState', '?')}"
    rep.add("moneyball.service", "ok", svc_line + " (inactive and static is normal: the timer starts it)")
    en = st.get("timer.UnitFileState", "?")
    rep.add("moneyball.timer", "ok" if en != "enabled" else "warn", f"{en} (stays disabled until canary and probe pass; use `timer enable`)"
            if en != "enabled" else "already enabled")


def step_tests(sysm: Sys, rep: Report, real_app: Path, args: argparse.Namespace) -> None:
    if args.skip_tests or args.skip_venv:
        rep.add("Offline tests", "skip", "skipped by flag")
        return
    if not (real_app / "tests").is_dir() and not sysm.dry:
        rep.add("Offline tests", "skip", "no tests folder")
        return
    py = real_app / ".venv" / "bin" / "python"
    base = f"{MOUNT}/pytest"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cp = sysm.run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={base}", "tests"],
                  check=False, cwd=str(real_app), env=env, timeout=600)
    if not sysm.dry:
        shutil.rmtree(base, ignore_errors=True)
    m = re.search(r"(\d+) passed", cp.stdout)
    if cp.returncode == 0 or sysm.dry:
        rep.add("Offline tests", "ok", f"{m.group(1)} passed" if m else "passed")
    else:
        rep.add("Offline tests", "fail", (cp.stdout or cp.stderr).strip()[-400:])


def cmd_init_pi(a: argparse.Namespace) -> int:
    sysm = Sys(dry=a.dry_run, root=a.sandbox_root, yes=a.yes)
    rep = Report()
    user = a.user or default_user()
    _, _, home = user_ids(user, sysm.sandbox)
    info = inspect_system()
    console.print(Panel.fit(f"[bold]init-pi[/]  user [cyan]{esc(user)}[/]  app [cyan]{esc(a.app_dir)}[/]  RAM disk [cyan]{MOUNT}[/] ({esc(a.size)})\n"
                            f"shard {a.shard} of {a.nshards}, burst {a.max_minutes} min, timer {esc(a.timer_time)} local"
                            + ("\n[yellow]dry run: nothing will be changed[/]" if a.dry_run else "")
                            + (f"\n[yellow]sandbox root {esc(sysm.root)}: system commands skipped[/]" if sysm.sandbox else ""),
                            title="Moneyball Pi setup", border_style="cyan"))
    console.print("\n[bold]1. Checking this machine[/]")
    try:
        step_system(rep, info, a.size, lenient=sysm.dry or sysm.sandbox)
        if rep.worst() == "fail":
            raise OpsError("the machine checks above failed")
        plan = ["RAM disk: add or fix the tmpfs line in /etc/fstab (original backed up), mount it",
                f"Code: make sure {a.app_dir} has the pipeline", "Virtualenv with the pipeline dependencies",
                "~/.moneyball_shell helper and /etc/moneyball.env (root, 0600)",
                "systemd service and timer (timer stays disabled)", "Run the offline test suite inside the RAM disk"]
        console.print("\n[bold]Plan[/]\n" + "\n".join(f"  - {p}" for p in plan))
        if not (a.yes or a.dry_run) and not Confirm.ask("Go ahead?", default=True):
            console.print("Stopped before making any change.")
            return 1
        sysm.ensure_sudo()
        console.print("\n[bold]2. Packages[/]")
        step_packages(sysm, rep, info, a.skip_packages)
        console.print("\n[bold]3. RAM disk[/]")
        step_ramdisk(sysm, rep, user, a.size, a)
        console.print("\n[bold]4. Code and virtualenv[/]")
        real_app = step_code(sysm, rep, user, a)
        step_venv(sysm, rep, real_app, a)
        step_shell_helper(sysm, rep, home, a)
        console.print("\n[bold]5. Secrets file[/]")
        placeholders = step_env(sysm, rep)
        if placeholders and not a.no_pair and not a.dry_run and not a.yes:
            if Confirm.ask("Enter and check your Hugging Face token now (pair-secrets)?", default=True):
                pa = argparse.Namespace(**{**vars(a), "gh_repo": None, "no_gh": False, "check_read_token": False,
                                           "delay": None, "pi_delay": None, "token_from_env": False, "env_file": ENV_FILE,
                                           "no_env": False, "repo": None, "strict": False})
                try:
                    cmd_pair_secrets(pa, sysm=sysm)
                    placeholders = False
                except OpsError as e:
                    rep.add("pair-secrets", "warn", str(e).splitlines()[0])
        if placeholders:
            rep.add("Secrets", "warn", "still placeholders: run `moneyball_ops.py pair-secrets`")
        console.print("\n[bold]6. systemd[/]")
        step_units(sysm, rep, user, a)
        console.print("\n[bold]7. Offline tests[/]")
        step_tests(sysm, rep, real_app, a)
    except OpsError as e:
        rep.add("Stopped", "fail", str(e).splitlines()[0])
        console.print(f"\n[bold red]{esc(e)}[/]")
    console.print()
    console.print(rep.table("init-pi summary"))
    if rep.worst() != "fail":
        console.print(Panel("1. python moneyball_ops.py pair-secrets      (if the secrets file still has placeholders)\n"
                            "2. python moneyball_ops.py canary            (must pass from this Pi)\n"
                            "3. python moneyball_ops.py probe-studio URL  (confirm the winner selector, runbook 3.4)\n"
                            "4. python moneyball_ops.py timer enable      (only after 2 and 3 pass)",
                            title="Next", border_style="green"))
    return 1 if rep.worst() == "fail" else 0


# ----------------------------------------------------------------------------- Hugging Face checks
@dataclasses.dataclass
class Check:
    name: str
    status: str  # ok, warn or fail
    detail: str = ""


@dataclasses.dataclass
class HFResult:
    checks: list[Check] = dataclasses.field(default_factory=list)
    repo_missing: bool = False
    token_kind: str = "unknown"

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == "warn" for c in self.checks)


def http_status(e: BaseException) -> Optional[int]:
    r = getattr(e, "response", None)
    code = getattr(r, "status_code", None)
    return code if isinstance(code, int) else None


def token_scope(info: dict, repo: str) -> tuple[str, list[Check]]:
    """Read what Hugging Face says about the token. The shape of this block is not guaranteed, so every
    result here is advice only; the real write test decides pass or fail."""
    out: list[Check] = []
    at = ((info.get("auth") or {}).get("accessToken")) or {}
    role = at.get("role") or "unknown"
    fg = at.get("fineGrained") or {}
    if role == "fineGrained":
        scoped = fg.get("scoped") or []
        glob = fg.get("global") or []
        names = []
        write_here = False
        for s in scoped:
            ent = (s.get("entity") or {}).get("name", "")
            names.append(ent)
            if ent.lower() == repo.lower() and any("write" in str(p) for p in (s.get("permissions") or [])):
                write_here = True
        if write_here:
            out.append(Check("Token scope", "ok", f"fine-grained, write permission on {repo}"))
        else:
            out.append(Check("Token scope", "warn", "fine-grained, but no write permission on this repo was listed "
                             f"(scoped to: {', '.join(names) or 'nothing'}). The write test below decides."))
        if glob:
            out.append(Check("Token breadth", "warn", f"it also holds account-wide permissions ({', '.join(map(str, glob))[:80]}). "
                             "The runbook asks for this one repo only."))
    elif role in ("write", "admin"):
        out.append(Check("Token scope", "warn", f"a classic '{role}' token can write to everything you own. "
                         "Recreate it as a fine-grained token limited to this one dataset repo."))
    elif role == "read":
        out.append(Check("Token scope", "ok", "read-only token"))
    else:
        out.append(Check("Token scope", "warn", "could not read the token's scope from the API"))
    return role, out


def validate_hf(token: str, repo: str, *, mode: str = "write", api: Any = None, write_test: bool = True) -> HFResult:
    """Check token validity, repo visibility and, in write mode, real write access.
    mode 'write' is for the scraper token, mode 'read' for the Colab token (writes must fail)."""
    res = HFResult()
    token = (token or "").strip()
    if not token:
        res.add("Token format", "fail", "empty")
        return res
    if re.search(r"\s", token):
        res.add("Token format", "fail", "contains whitespace (a paste error?)")
        return res
    res.add("Token format", "ok" if token.startswith("hf_") else "warn",
            "looks like an hf_ token" if token.startswith("hf_") else "does not start with hf_")
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo or ""):
        res.add("Repo name", "fail", f"'{repo}' is not in the form USER/NAME")
        return res
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise OpsError("huggingface_hub is not installed: pip install huggingface_hub")
        api = HfApi(token=token)
    try:
        info = api.whoami()
        res.add("Token accepted", "ok", f"signed in as {info.get('name', '?')}")
    except Exception as e:
        s = http_status(e)
        res.add("Token accepted", "fail", "Hugging Face rejected the token (401). Wrong, expired or revoked." if s == 401
                else f"could not reach Hugging Face: {type(e).__name__}: {str(e)[:120]}")
        return res
    res.token_kind, scope_checks = token_scope(info, repo)
    res.checks += scope_checks
    try:
        ri = api.repo_info(repo, repo_type="dataset")
        private = getattr(ri, "private", None)
        if private is True:
            res.add("Repo is private", "ok", f"dataset {repo} is private")
        elif private is False:
            res.add("Repo is private", "fail", f"dataset {repo} is PUBLIC. It will hold scraped handles and raw HTML. "
                    "Make it private in the repo settings first.")
        else:
            res.add("Repo is private", "warn", "could not read the visibility flag")
    except Exception as e:
        s = http_status(e)
        if s == 404 or type(e).__name__ == "RepositoryNotFoundError":
            res.repo_missing = True
            res.add("Repo exists", "fail", f"dataset {repo} not found, or this token cannot see it")
        elif s in (401, 403):
            res.add("Repo exists", "fail", f"this token has no access to {repo} ({s})")
        else:
            res.add("Repo exists", "fail", f"{type(e).__name__}: {str(e)[:140]}")
        return res
    try:
        n = len(list(api.list_repo_files(repo, repo_type="dataset")))
        res.add("Read access", "ok", f"{n} files visible")
    except Exception as e:
        res.add("Read access", "fail", f"{type(e).__name__}: {str(e)[:140]}")
        return res
    if not write_test:
        res.add("Write access", "warn", "not tested (dry run)")
        return res
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    name = f"healthcheck-{os.urandom(4).hex()}.txt"
    wrote, status = False, None
    try:
        api.create_commit(repo, repo_type="dataset", commit_message="moneyball_ops healthcheck",
                          operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=b"ok")])
        wrote = True
    except Exception as e:
        status = http_status(e)
        detail = f"{status or type(e).__name__}"
    if wrote:
        try:
            api.create_commit(repo, repo_type="dataset", commit_message="moneyball_ops healthcheck cleanup",
                              operations=[CommitOperationDelete(path_in_repo=name)])
            cleaned = "test file added and removed"
        except Exception:
            cleaned = f"test file {name} added but could not be removed, delete it by hand"
        if mode == "write":
            res.add("Write access", "ok", cleaned)
        else:
            res.add("Read-only as intended", "warn", "this token CAN write. The Colab token should be read-only. " + cleaned)
    else:
        if mode == "write":
            if status in (401, 403):
                res.add("Write access", "fail", f"write refused ({status}). Recreate the token with 'Write access to contents/settings "
                        "of selected repos' ticked for this repo.")
            else:
                res.add("Write access", "fail", f"write failed: {detail}")
        else:
            if status in (401, 403):
                res.add("Read-only as intended", "ok", "write was refused, which is what a read token should do")
            else:
                res.add("Read-only as intended", "warn", f"write test inconclusive: {detail}")
    return res


def show_hf_result(res: HFResult, title: str) -> None:
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    t.add_column("check")
    t.add_column("result")
    t.add_column("detail", overflow="fold")
    for c in res.checks:
        label, style = {"ok": ("OK", "green"), "warn": ("WARN", "yellow"), "fail": ("FAIL", "bold red")}[c.status]
        t.add_row(esc(c.name), f"[{style}]{label}[/]", esc(c.detail))
    console.print(t)


# ----------------------------------------------------------------------------- GitHub CLI wrapper
class GH:
    """Thin wrapper over the gh CLI. Secret values only ever travel through stdin."""

    def __init__(self, repo: Optional[str] = None) -> None:
        self.bin = shutil.which("gh")
        self.repo = repo

    def _run(self, args: list[str], *, input: Optional[str] = None, check: bool = True, timeout: int = 90,
             use_repo: bool = True) -> subprocess.CompletedProcess:
        if not self.bin:
            raise OpsError("the GitHub CLI (gh) is not installed: https://cli.github.com")
        argv = [self.bin] + args
        if use_repo and self.repo:
            argv += ["--repo", self.repo]
        try:
            cp = subprocess.run(argv, input=input, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise OpsError(f"gh timed out: gh {' '.join(args[:3])}")
        if check and cp.returncode != 0:
            raise OpsError(f"gh {' '.join(args[:3])} failed: {(cp.stderr or cp.stdout).strip()[-400:]}")
        return cp

    def ready(self) -> tuple[bool, str]:
        if not self.bin:
            return False, "gh is not installed"
        cp = self._run(["auth", "status"], check=False, use_repo=False, timeout=30)
        return (cp.returncode == 0), ("logged in" if cp.returncode == 0 else "not logged in: run `gh auth login`")

    def detect_repo(self) -> Optional[str]:
        cp = self._run(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], check=False, use_repo=False, timeout=30)
        s = cp.stdout.strip()
        return s if cp.returncode == 0 and re.fullmatch(r"[\w.\-]+/[\w.\-]+", s) else None

    def set_secret(self, name: str, value: str) -> None:
        self._run(["secret", "set", name], input=value)

    def set_var(self, name: str, value: str) -> None:
        self._run(["variable", "set", name, f"--body={value}"])

    def get_var(self, name: str) -> Optional[str]:
        cp = self._run(["variable", "get", name], check=False, timeout=30)
        return cp.stdout.strip() if cp.returncode == 0 else None

    def list_vars(self) -> dict[str, str]:
        cp = self._run(["variable", "list", "--json", "name,value"], check=False, timeout=30)
        if cp.returncode == 0:
            try:
                return {r["name"]: r.get("value", "") for r in json.loads(cp.stdout)}
            except (ValueError, KeyError):
                pass
        cp = self._run(["variable", "list"], check=False, timeout=30)
        out: dict[str, str] = {}
        for ln in cp.stdout.splitlines():
            parts = ln.split("\t") if "\t" in ln else ln.split(None, 2)
            if len(parts) >= 2:
                out[parts[0].strip()] = parts[1].strip()
        return out

    def list_secret_names(self) -> list[str]:
        cp = self._run(["secret", "list", "--json", "name"], check=False, timeout=30)
        if cp.returncode == 0:
            try:
                return [r["name"] for r in json.loads(cp.stdout)]
            except (ValueError, KeyError):
                pass
        cp = self._run(["secret", "list"], check=False, timeout=30)
        return [ln.split()[0] for ln in cp.stdout.splitlines() if ln.strip()]


# ----------------------------------------------------------------------------- pair-secrets
def ask_token(label: str, from_env: bool) -> str:
    if from_env:
        t = os.environ.get("HF_TOKEN", "")
        if not t:
            raise OpsError("--token-from-env was given but HF_TOKEN is not set")
        return t.strip()
    return Prompt.ask(f"Hugging Face {label} [dim](input hidden)[/]", password=True).strip()


def cmd_pair_secrets(a: argparse.Namespace, sysm: Optional[Sys] = None) -> int:
    sysm = sysm or Sys(dry=a.dry_run, root=a.sandbox_root, yes=a.yes)
    st = load_state()
    rep = Report()
    console.print(Panel.fit("Checks your Hugging Face token and dataset, then stores them where the workers read them.\n"
                            "The token is never printed or placed on a command line.",
                            title="pair-secrets", border_style="cyan"))
    env_path = a.env_file
    existing = {}
    if not a.no_env and sysm.exists(env_path):
        with contextlib.suppress(OpsError):
            existing = env_parse(sysm.read_text(env_path) or "")
    repo = a.repo or os.environ.get("HF_REPO") or existing.get("HF_REPO") or st.get("hf_repo")
    if repo in (None, "", PLACEHOLDER_REPO):
        repo = Prompt.ask("Hugging Face dataset repo [dim](USER/NAME, private)[/]", default="") if not a.yes else ""
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo or ""):
        raise OpsError("need a dataset repo like YOUR_HF_USER/hackathon-moneyball (pass --repo)")
    token = ""
    res: Optional[HFResult] = None
    for attempt in range(3):
        token = ask_token("WRITE token", a.token_from_env)
        with console.status("checking with Hugging Face ..."):
            res = validate_hf(token, repo, mode="write", write_test=not sysm.dry)
        if res.repo_missing and not sysm.dry:
            show_hf_result(res, "Hugging Face checks")
            if Confirm.ask(f"Dataset {esc(repo)} does not exist. Create it as a private dataset?", default=True):
                try:
                    from huggingface_hub import HfApi
                    HfApi(token=token).create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                    console.print("[green]created[/]")
                except Exception as e:
                    raise OpsError(f"could not create it ({type(e).__name__}). Create it in the browser at "
                                   "https://huggingface.co/new-dataset with visibility Private, then rerun.")
                with console.status("checking again ..."):
                    res = validate_hf(token, repo, mode="write", write_test=not sysm.dry)
        show_hf_result(res, "Hugging Face checks")
        if not res.failed:
            break
        if a.token_from_env or a.yes or attempt == 2 or not Confirm.ask("Try a different token?", default=True):
            raise OpsError("the Hugging Face checks failed, nothing was stored")
    assert res is not None
    rep.add("Hugging Face token and repo", "warn" if res.warned else "ok",
            f"{res.token_kind} token, dataset {repo}")
    if res.warned:
        if a.strict:
            raise OpsError("warnings present and --strict was given")
        if not a.yes and not Confirm.ask("There are warnings above. Continue anyway?", default=False):
            raise OpsError("stopped on warnings, nothing was stored")
    if a.check_read_token:
        rtok = ask_token("READ token for Colab", False)
        with console.status("checking the read token ..."):
            rres = validate_hf(rtok, repo, mode="read", write_test=not sysm.dry)
        show_hf_result(rres, "Read token checks")
        rep.add("Colab read token", "fail" if rres.failed else ("warn" if rres.warned else "ok"), "")
        del rtok
    # local secrets file
    want_env = not a.no_env and (sysm.sandbox or is_pi_like() or sysm.exists(env_path) or
                                 (not a.yes and Confirm.ask(f"Write {env_path} on this machine (say no on a laptop)?", default=False)))
    if want_env:
        sysm.ensure_sudo()
        updates = {"HF_REPO": repo, "HF_TOKEN": token}
        if a.pi_delay is not None:
            updates["DELAY"] = str(a.pi_delay)
        elif "DELAY" not in existing:
            updates["DELAY"] = "2.5"
        env_set(sysm, updates, env_path, secret=True)
        if not sysm.dry:
            back = env_parse(sysm.read_text(env_path) or "")
            same = hashlib.sha256(back.get("HF_TOKEN", "").encode()).digest() == hashlib.sha256(token.encode()).digest()
            mode = sysm.p(env_path).stat().st_mode & 0o777
            if not same or back.get("HF_REPO") != repo:
                raise OpsError(f"{env_path} did not read back correctly")
            rep.add(env_path, "ok" if mode == 0o600 else "fail", f"written, mode {mode:04o}" + ("" if mode == 0o600 else " (expected 0600)"))
        else:
            rep.add(env_path, "skip", "dry run")
    else:
        rep.add("Local secrets file", "skip", "not requested")
    # GitHub
    if a.no_gh:
        rep.add("GitHub secrets", "skip", "--no-gh")
    else:
        gh = GH(a.gh_repo or st.get("gh_repo"))
        ok, why = gh.ready()
        if not ok:
            rep.add("GitHub secrets", "skip", why + ". Set them later with: gh secret set HF_TOKEN; gh variable set HF_REPO --body REPO")
        else:
            if not gh.repo:
                gh.repo = gh.detect_repo() or (Prompt.ask("GitHub repo [dim](OWNER/NAME)[/]") if not a.yes else None)
            if not gh.repo or not re.fullmatch(r"[\w.\-]+/[\w.\-]+", gh.repo):
                rep.add("GitHub secrets", "skip", "no GitHub repo given (use --gh-repo OWNER/NAME)")
            elif not (a.yes or Confirm.ask(f"Set secret HF_TOKEN and variables HF_REPO, DELAY on {esc(gh.repo)}?", default=True)):
                rep.add("GitHub secrets", "skip", "declined")
            elif sysm.dry:
                console.print(f"    [dim]dry-run: gh secret set HF_TOKEN (stdin), gh variable set HF_REPO / DELAY on {esc(gh.repo)}[/]")
                rep.add("GitHub secrets", "skip", "dry run")
            else:
                gh.set_secret("HF_TOKEN", token)
                gh.set_var("HF_REPO", repo)
                cur_delay = gh.get_var("DELAY")
                if a.delay is not None or cur_delay is None:
                    gh.set_var("DELAY", str(a.delay if a.delay is not None else 2.0))
                names, vars_ = gh.list_secret_names(), gh.list_vars()
                good = "HF_TOKEN" in names and vars_.get("HF_REPO") == repo
                rep.add(f"GitHub {gh.repo}", "ok" if good else "fail",
                        f"secret HF_TOKEN {'present' if 'HF_TOKEN' in names else 'MISSING'}, HF_REPO {'matches' if vars_.get('HF_REPO') == repo else 'MISMATCH'}, "
                        f"DELAY={vars_.get('DELAY', '?')}")
                cp = gh._run(["api", f"repos/{gh.repo}/actions/permissions/workflow"], check=False, use_repo=False, timeout=30)
                with contextlib.suppress(ValueError):
                    if json.loads(cp.stdout).get("default_workflow_permissions") == "write":
                        rep.add("Actions workflow permissions", "warn", "default is read AND write. Set 'Read repository contents' "
                                "in repo Settings, Actions, General (the workflow also pins read).")
                console.print("[dim]Reminder (runbook 1.2): in the repo's Actions settings require approval for all outside "
                              "collaborators' fork pull requests.[/]")
                save_state({"gh_repo": gh.repo})
    save_state({"hf_repo": repo, "paired_at": iso()})
    token = ""
    console.print()
    console.print(rep.table("pair-secrets summary"))
    return 1 if rep.worst() == "fail" else 0


# ----------------------------------------------------------------------------- fetching, canary, timer
class Fetcher:
    """One polite request at a time with Chrome's TLS fingerprint. Same block rules as pipeline.py,
    but no retry storms: a block is reported, not fought."""

    def __init__(self, delay: float = 2.0, timeout: int = 30) -> None:
        try:
            from curl_cffi import requests as cr
        except ImportError:
            raise OpsError("curl_cffi is not installed: pip install curl_cffi")
        self.s = cr.Session(impersonate="chrome", timeout=timeout, headers={"Accept-Language": "en-US,en;q=0.9"})
        self.delay, self.last = delay, 0.0

    def _wait(self) -> None:
        gap = self.delay * (1 + random.uniform(-0.4, 0.4))
        dt_ = time.monotonic() - self.last
        if dt_ < gap:
            time.sleep(gap - dt_)
        self.last = time.monotonic()

    def get(self, url: str, params: Optional[dict] = None) -> tuple[str, Any, Optional[str]]:
        self._wait()
        try:
            r = self.s.get(url, params=params, allow_redirects=True)
        except Exception as e:
            return "error", f"exc:{type(e).__name__}", None
        code = r.status_code
        body = r.text if code == 200 else ""
        blocked = (code in (403, 429, 503) or r.headers.get("cf-mitigated") == "challenge"
                   or (code == 200 and "Just a moment..." in body[:3000]))
        if code == 200 and not blocked:
            return "ok", code, body
        if code in (404, 410):
            return "gone", code, None
        return ("blocked" if blocked else "error"), code, None


CANARY_TESTS = [("api", DEVPOST_API, {"status[]": "ended", "order_by": "recently-added", "page": 1}),
                ("gallery", "https://cmu-hackathon.devpost.com/project-gallery", None),
                ("project", "https://devpost.com/software/coterie-57fv6e", None)]


def local_canary(delay: float = 2.0) -> list[tuple[str, str, Any]]:
    f = Fetcher(delay)
    out = []
    for name, url, params in CANARY_TESTS:
        out.append((name, *f.get(url, params)[:2]))
    return out


def gha_canaries(gh: GH, n: int) -> list[tuple[str, bool]]:
    def latest() -> Optional[str]:
        cp = gh._run(["run", "list", "--workflow=ingest.yml", "--limit", "1", "--json", "databaseId", "-q", ".[0].databaseId"], check=False, timeout=30)
        return cp.stdout.strip() or None

    results: list[tuple[str, bool]] = []
    for i in range(n):
        before = latest()
        gh._run(["workflow", "run", "ingest.yml", "-f", "stage=canary"])
        rid = None
        with console.status(f"canary {i + 1}/{n}: waiting for the run to appear ..."):
            for _ in range(30):
                time.sleep(3)
                cur = latest()
                if cur and cur != before:
                    rid = cur
                    break
        if not rid:
            results.append(("no run appeared", False))
            continue
        with console.status(f"canary {i + 1}/{n}: run {rid} in progress ..."):
            cp = gh._run(["run", "watch", rid, "--exit-status"], check=False, timeout=900)
        results.append((rid, cp.returncode == 0))
        time.sleep(5)
    return results


def cmd_canary(a: argparse.Namespace) -> int:
    if a.gha:
        gh = GH(a.gh_repo or load_state().get("gh_repo"))
        ok, why = gh.ready()
        if not ok:
            raise OpsError(f"GitHub CLI: {why}")
        gh.repo = gh.repo or gh.detect_repo()
        if not gh.repo:
            raise OpsError("pass --gh-repo OWNER/NAME")
        res = gha_canaries(gh, a.gha)
        t = Table(title=f"GitHub Actions canaries on {gh.repo}", box=box.SIMPLE_HEAVY)
        t.add_column("run")
        t.add_column("result")
        for rid, good in res:
            t.add_row(esc(rid), "[green]PASS[/]" if good else "[bold red]FAIL[/]")
        console.print(t)
        passes = sum(1 for _, g in res if g)
        save_state({"gha_canary": {"at": iso(), "passes": passes, "total": len(res)}})
        console.print("3 of 3 is a pass. One failure in three means expect occasional circuit-breaker trips; "
                      "consistent failure means GitHub's IP ranges are blocked (runbook 4.6 case C).")
        return 0 if passes == len(res) else 3
    res = local_canary(a.delay)
    t = Table(title=f"Canary from this machine ({platform.node()})", box=box.SIMPLE_HEAVY)
    t.add_column("endpoint")
    t.add_column("result")
    t.add_column("code")
    for name, kind, code in res:
        t.add_row(name, {"ok": "[green]ok[/]", "blocked": "[bold red]blocked[/]"}.get(kind, f"[yellow]{kind}[/]"), esc(code))
    console.print(t)
    good = all(k == "ok" for _, k, _ in res)
    save_state({"canary_local": {"at": iso(), "ok": good, "host": platform.node(),
                                 "results": {n: k for n, k, _ in res}}})
    if good:
        console.print("[green]Not blocked right now.[/] That means 'not blocked at this moment', not 'never blocked'.")
        return 0
    if any(k == "blocked" for _, k, _ in res):
        console.print("[bold red]This IP is being challenged.[/] Do not start the run from here. Wait a few hours and retry; "
                      "if it persists, use Actions only (runbook 4.6 case C).")
    else:
        console.print("[yellow]Errors that are not blocks[/] (network problem on this machine, most likely).")
    return 3


def timer_gate(sysm: Sys) -> tuple[list[str], list[str]]:
    """Return (blocking issues, warnings) for enabling the nightly timer."""
    st = load_state()
    issues: list[str] = []
    warns: list[str] = []
    c = st.get("canary_local")
    ca = parse_ts(c["at"]) if c else None
    if not c or not c.get("ok"):
        issues.append("no passing canary from this machine on record: run `canary`")
    elif ca and (utcnow() - ca).total_seconds() > 86400:
        issues.append("the last passing canary is more than 24 hours old: run `canary` again")
    p = st.get("probe")
    if not p or not p.get("verified"):
        issues.append("winner selector not verified: run `probe-studio`, test against a real gallery, then :save")
    if not sysm.exists(f"{UNIT_DIR}/moneyball.timer"):
        issues.append("the systemd units are not installed: run `init-pi`")
    cur = sysm.read_text(ENV_FILE)
    vals = env_parse(cur) if cur else {}
    if not cur or vals.get("HF_TOKEN", PLACEHOLDER_TOKEN) in ("", PLACEHOLDER_TOKEN) or vals.get("HF_REPO", PLACEHOLDER_REPO) in ("", PLACEHOLDER_REPO):
        issues.append(f"{ENV_FILE} still has placeholder values: run `pair-secrets`")
    g = st.get("gha_canary")
    if not g:
        warns.append("no GitHub Actions canary on record (`canary --gha 3`). Fine if Actions is not used.")
    elif g.get("passes", 0) < g.get("total", 1):
        warns.append(f"GitHub Actions canary passed {g.get('passes')} of {g.get('total')}")
    return issues, warns


def cmd_timer(a: argparse.Namespace) -> int:
    sysm = Sys(dry=a.dry_run, root=a.sandbox_root, yes=a.yes)
    if a.action == "status":
        stt = service_state(sysm)
        t = Table(title="Pi timer", box=box.SIMPLE_HEAVY)
        t.add_column("item")
        t.add_column("value")
        for k in sorted(stt):
            t.add_row(k, esc(stt[k]))
        console.print(t)
        issues, warns = timer_gate(sysm)
        for i in issues:
            console.print(f"[yellow]gate:[/] {esc(i)}")
        for w in warns:
            console.print(f"[dim]note: {esc(w)}[/]")
        if not issues:
            console.print("[green]All gates pass, the timer can be enabled.[/]")
        return 0
    if a.action == "disable":
        sysm.ensure_sudo()
        sysm.run(["systemctl", "disable", "--now", "moneyball.timer"], priv=True, system=True)
        console.print("[green]timer disabled[/] (a running burst is not stopped; use `sudo systemctl stop moneyball.service`)")
        return 0
    issues, warns = timer_gate(sysm)
    for w in warns:
        console.print(f"[dim]note: {esc(w)}[/]")
    if issues and not a.force:
        console.print("[bold red]Not enabling the timer yet:[/]")
        for i in issues:
            console.print(f"  - {esc(i)}")
        console.print("(runbook section 3: do not start the long run until canary and probe pass; --force overrides)")
        return 1
    sysm.ensure_sudo()
    sysm.run(["systemctl", "enable", "--now", "moneyball.timer"], priv=True, system=True)
    cp = sysm.run(["systemctl", "list-timers", "moneyball.timer", "--no-pager"], check=False, system=True, readonly=True)
    console.print(esc(cp.stdout.strip()) or "[dim]timer enabled[/]")
    console.print("[green]Nightly timer enabled.[/] To run one burst right now: sudo systemctl start moneyball.service")
    return 0


# ----------------------------------------------------------------------------- probe-studio
SOFT = re.compile(r"^(?:https?://devpost\.com)?/software/([^/?#]+)/?$")
WINNERISH = re.compile(r"winner|prize|badge|award|trophy|ribbon|medal", re.I)
HELP_TEXT = """Type a CSS selector to test it against every project card, for example   .winner   or   span[class*="prize"]
  :expect N      how many Winner badges YOU count on this page in your browser (needed to pass)
  :suggest       selectors that would bring the flagged count to your expected number
  :classes       class names in the cards that look like badges or prizes, with card counts
  :card N        show the raw HTML of card N (find the real badge markup here)
  :cards [N]     list the first N cards with their flags
  :clear         drop the selector and use only the built-in heuristics
  :load X        load another gallery or project URL, or a saved .html file
  :save          write the verified selector to /etc/moneyball.env and the GitHub variable
  :help          this text          :quit (or q)   leave"""


def _a(n: Any, k: str) -> str:
    return (n.attributes.get(k) or "") if n is not None else ""


def _txt(n: Any) -> str:
    return n.text(deep=True, separator=" ", strip=True) if n is not None else ""


@dataclasses.dataclass
class Card:
    slug: str
    node: Any
    text: str
    likes: Optional[int]
    comments: Optional[int]


@dataclasses.dataclass
class Page:
    kind: str  # gallery or project
    source: str
    html: str
    tree: Any
    cards: list[Card]
    expected: Optional[int]


def load_selectolax() -> Any:
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        raise OpsError("selectolax is not installed: pip install selectolax")
    return HTMLParser


def find_cards(html: str) -> tuple[Any, list[Card], Optional[int]]:
    """Same card rule as pipeline.parse_gallery: the smallest ancestor of a project link that holds
    exactly one distinct project link, so it survives CSS class renames."""
    t = load_selectolax()(html)
    first: dict[str, Any] = {}
    order: list[str] = []
    for a in t.css("a[href]"):
        m = SOFT.match(_a(a, "href").strip())
        if m and m.group(1) != "built-with" and m.group(1) not in first:
            first[m.group(1)] = a
            order.append(m.group(1))

    def slugs_in(node: Any) -> set[str]:
        out = set()
        for x in node.css("a[href]"):
            m = SOFT.match(_a(x, "href").strip())
            if m:
                out.add(m.group(1))
        return out

    cards: list[Card] = []
    for slug in order:
        node = first[slug]
        for _ in range(8):
            p = node.parent
            if p is None or p.tag in ("body", "html", "ul", "ol", "main", "section", "[document]"):
                break
            if len(slugs_in(p)) > 1:
                break
            node = p
        text = _txt(node)
        m = re.search(r"(\d+)\s+(\d+)\s*$", text)
        cards.append(Card(slug, node, text, int(m.group(1)) if m else None, int(m.group(2)) if m else None))
    body = t.body.text(deep=True, separator=" ", strip=True) if t.body else ""
    m = re.search(r"(\d+)\s*[\u2013\-]\s*(\d+)\s*of\s*([\d,]+)", body)
    return t, cards, (int(m.group(3).replace(",", "")) if m else None)


def default_signals(node: Any) -> list[tuple[str, str]]:
    """What pipeline.py flags as a winner badge without any WINNER_CSS."""
    out: list[tuple[str, str]] = []
    for n in node.css("[class]"):
        if "winner" in _a(n, "class").lower():
            w = _txt(n)
            if w:
                out.append(("class", w[:60]))
    for n in node.css("img[alt], [title]"):
        v = _a(n, "alt") or _a(n, "title")
        if "winner" in v.lower():
            out.append(("alt/title", v[:60]))
    for n in node.css("span, small, b, strong, div, p"):
        s = _txt(n)
        if s.lower() == "winner":
            out.append(("text", s))
    return out


def run_selector(page: Page, css: str) -> dict:
    """Apply a selector inside each card exactly as pipeline.py does (descendants of the card)."""
    res: dict = {"error": None, "cards": set(), "elements": 0, "samples": {}}
    if not css:
        return res
    for i, c in enumerate(page.cards):
        try:
            hits = c.node.css(css)
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
            res["cards"], res["elements"] = set(), 0
            return res
        if hits:
            res["cards"].add(i)
            res["elements"] += len(hits)
            res["samples"][i] = (_txt(hits[0])[:40] or "(no text)")
    return res


def compute_metrics(page: Page, css: str, expect: Optional[int]) -> dict:
    M = len(page.cards)
    default = {i for i, c in enumerate(page.cards) if default_signals(c.node)}
    sel = run_selector(page, css)
    union = default | sel["cards"]
    notes: list[str] = []
    if M == 0:
        verdict = "NO CARDS"
        notes.append("No project links found. Wrong URL, a login wall, or the markup changed.")
    elif expect is None:
        verdict = "SET EXPECT"
        notes.append("Count the Winner badges on this page in your browser and enter :expect N.")
    elif len(union) == expect and 0 < len(union) < M:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        if len(union) == 0:
            notes.append(f"Nothing is flagged but you expect {expect}.")
        elif len(union) >= M and M > 1:
            notes.append("Every card is flagged: something overmatches.")
        elif len(union) > expect:
            notes.append(f"{len(union) - expect} too many flagged.")
        else:
            notes.append(f"{expect - len(union)} winner(s) still missed.")
    if M > 1 and len(default) == M:
        notes.append("The built-in heuristics alone already flag EVERY card. WINNER_CSS can only add matches, "
                     "so a selector cannot fix this. Look at :card 1 for a class named like 'winner' that is on every card.")
    if css and sel["error"]:
        notes.append(f"Selector rejected by the parser: {sel['error']}")
        verdict = "BAD SELECTOR"
    elif css and not sel["cards"] and M:
        notes.append("Selector matches no card.")
    if css and M > 1 and len(sel["cards"]) == M:
        notes.append("Selector matches every card, so it cannot tell winners apart.")
    if page.expected and M and M < page.expected:
        notes.append(f"Page shows {page.expected} projects in total; this is the first page with {M} cards (page 1 is enough for this check).")
    return {"M": M, "default": default, "sel": sel, "union": union, "verdict": verdict, "notes": notes}


def suggest_selectors(page: Page, expect: Optional[int], limit: int = 8) -> list[dict]:
    M = len(page.cards)
    default = {i for i, c in enumerate(page.cards) if default_signals(c.node)}
    tok: dict[str, set[int]] = defaultdict(set)
    for i, c in enumerate(page.cards):
        for n in c.node.css("[class]"):
            for t in _a(n, "class").split():
                if re.fullmatch(r"[A-Za-z_][\w-]*", t):
                    tok[t].add(i)
    rows = []
    for t, idx in tok.items():
        if not (0 < len(idx) < M):
            continue
        union = default | idx
        if union == default:
            continue
        rows.append({"selector": f".{t}", "alone": len(idx), "union": len(union),
                     "delta": abs(len(union) - expect) if expect is not None else 0,
                     "hint": 0 if WINNERISH.search(t) else 1})
    rows.sort(key=lambda r: (r["delta"], r["hint"], -r["alone"], r["selector"]))
    return rows[:limit]


def class_census(page: Page) -> list[tuple[str, int]]:
    tok: dict[str, set[int]] = defaultdict(set)
    for i, c in enumerate(page.cards):
        for n in c.node.css("[class]"):
            for t in _a(n, "class").split():
                if WINNERISH.search(t):
                    tok[t].add(i)
    return sorted(((t, len(s)) for t, s in tok.items()), key=lambda x: (-x[1], x[0]))


# ---- pipeline parity -------------------------------------------------------
def find_pipeline(hint: Optional[str] = None) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    cands = ([Path(hint)] if hint else []) + [Path(APP_DIR) / "pipeline.py", here / "pipeline.py",
                                             here.parent / "pipeline.py", Path.cwd() / "pipeline.py"]
    for c in cands:
        if c.is_file():
            return c
    return None


def load_pipeline(hint: Optional[str] = None) -> Any:
    p = find_pipeline(hint)
    if not p:
        return None
    try:
        spec = importlib.util.spec_from_file_location("pipeline_for_probe", p)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception as e:
        console.print(f"[dim]pipeline.py found at {esc(p)} but could not be loaded ({esc(type(e).__name__)}); parity check off[/]")
        return None


def pipeline_gallery(mod: Any, html: str, css: str) -> tuple[set[str], int]:
    old = getattr(mod, "WINNER_CSS", "")
    mod.WINNER_CSS = css
    try:
        cards, _ = mod.parse_gallery(html)
    finally:
        mod.WINNER_CSS = old
    return {c["slug"] for c in cards if c["is_winner"]}, len(cards)


# ---- loading ---------------------------------------------------------------
def load_page(src: str, fetcher: Optional[Fetcher], delay: float = 1.5) -> Page:
    if Path(src).is_file():
        html, kind_hint = Path(src).read_text(errors="ignore"), None
    elif src.startswith("http"):
        f = fetcher or Fetcher(delay)
        kind, code, text = f.get(src)
        if kind != "ok":
            hint = ("This IP is being challenged. Run `canary`, wait, and try again later." if kind == "blocked"
                    else "No such page." if kind == "gone" else "Network or server problem.")
            raise OpsError(f"fetch failed: {kind} {code}. {hint}")
        html = text or ""
        kind_hint = "project" if re.search(r"/software/[^/?#]+/?$", src) and "built-with" not in src else None
    else:
        raise OpsError(f"'{src}' is neither a URL nor an existing file")
    tree, cards, expected = find_cards(html)
    kind = kind_hint or ("gallery" if len(cards) >= 2 else "project")
    if kind == "project":
        cards, expected = [], None
    return Page(kind, src, html, tree, cards, expected)


def suggest_hackathons(fetcher: Fetcher, pages: int = 20, min_reg: int = 300) -> list[dict]:
    out: list[dict] = []
    with console.status("reading the Devpost hackathon list (about one request every 1.5 s) ..."):
        for pg in range(1, pages + 1):
            kind, code, text = fetcher.get(DEVPOST_API, {"status[]": "ended", "order_by": "recently-added", "page": pg})
            if kind != "ok":
                raise OpsError(f"the Devpost list returned {kind} {code} on page {pg}. If blocked, run `canary` and wait.")
            try:
                hs = json.loads(text or "{}").get("hackathons") or []
            except ValueError:
                raise OpsError("the Devpost list did not return JSON")
            out += [h for h in hs if h.get("winners_announced") and (h.get("registrations_count") or 0) >= min_reg and not h.get("invite_only")]
    out.sort(key=lambda h: -(h.get("registrations_count") or 0))
    return out[:8]


# ---- rendering ---------------------------------------------------------------
def render_metrics(page: Page, css: str, expect: Optional[int], mod: Any) -> Panel:
    m = compute_metrics(page, css, expect)
    color = {"PASS": "green", "FAIL": "bold red", "BAD SELECTOR": "bold red", "NO CARDS": "bold red"}.get(m["verdict"], "yellow")
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("page", Text(page.source[:90]))
    t.add_row("project cards detected", f"{m['M']}" + (f"   (page says {page.expected} in total)" if page.expected else ""))
    t.add_row("you expect winners", str(expect) if expect is not None else "[yellow]not set (:expect N)[/]")
    t.add_row("built-in heuristics flag", str(len(m["default"])))
    t.add_row("your selector", Text(css or "(none)"))
    t.add_row("  cards it matches", f"{len(m['sel']['cards'])}  ({m['sel']['elements']} elements)" if css else "-")
    t.add_row("pipeline will flag", f"[bold]{len(m['union'])}[/] of {m['M']}  (built-in OR selector)")
    if mod is not None and page.kind == "gallery":
        try:
            slugs, _ = pipeline_gallery(mod, page.html, css)
            mine = {page.cards[i].slug for i in m["union"]}
            if slugs == mine:
                t.add_row("pipeline.py parity", "[green]identical result from the real parser[/]")
            else:
                t.add_row("pipeline.py parity", f"[bold red]DIFFERENT[/]: real parser flags {len(slugs)}, this view {len(mine)}")
        except Exception as e:
            t.add_row("pipeline.py parity", f"[yellow]check failed: {esc(type(e).__name__)}[/]")
    elif mod is None:
        t.add_row("pipeline.py parity", "[dim]pipeline.py not found, check skipped (--pipeline PATH)[/]")
    t.add_row("verdict", f"[{color}]{m['verdict']}[/]")
    parts: list[Any] = [t]
    for n in m["notes"]:
        parts.append(Text("  - " + n, style="yellow"))
    return Panel(Group(*parts), title="probe", border_style=color.replace("bold ", ""))


def render_cards(page: Page, css: str, n: int = 8) -> Table:
    m = compute_metrics(page, css, None)
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    for h in ("#", "slug", "likes/cmts", "built-in", "selector", "card text"):
        t.add_column(h, overflow="fold")
    for i, c in enumerate(page.cards[:n]):
        sig = default_signals(c.node)
        t.add_row(str(i + 1), c.slug[:28], f"{c.likes}/{c.comments}" if c.likes is not None else "-",
                  ("[green]yes[/] " + esc(",".join(sorted({k for k, _ in sig})))) if sig else "-",
                  ("[green]yes[/] " + esc(m["sel"]["samples"].get(i, ""))) if i in m["sel"]["cards"] else "-",
                  esc(c.text[:46]))
    return t


def render_project(page: Page, css: str, mod: Any) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("page", Text(page.source[:90]))
    t.add_row("#app-details-left present", "yes" if page.tree.css_first("#app-details-left") else "[yellow]no[/] (story word counts would include page chrome)")
    cls = [(_txt(n)[:60]) for n in page.tree.css("[class]") if "winner" in _a(n, "class").lower() and _txt(n)]
    alt = [_a(n, "alt")[:60] for n in page.tree.css("img[alt]") if "winner" in _a(n, "alt").lower()]
    t.add_row("built-in winner signals", f"{len(cls)} class, {len(alt)} image" + (": " + esc("; ".join((cls + alt)[:3])) if cls or alt else ""))
    if css:
        try:
            hits = page.tree.css(css)
            t.add_row("your selector", Text(css))
            t.add_row("  matches", f"{len(hits)} element(s): " + esc(" | ".join((_txt(h)[:40] or "(no text)") for h in hits[:4])))
        except Exception as e:
            t.add_row("your selector", f"[bold red]rejected: {esc(type(e).__name__)}[/]")
    if mod is not None:
        try:
            r = mod.parse_project(page.html)
            for k in ("title", "tagline", "built_with", "story_root_found", "team_size", "video_platform", "video_id",
                      "likes", "comments", "n_updates", "first_update_text", "try_links", "is_winner_page", "winner_texts"):
                v = r.get(k)
                t.add_row(f"parse_project.{k}", esc(str(v)[:100]))
        except Exception as e:
            t.add_row("parse_project", f"[yellow]failed: {esc(type(e).__name__)}[/]")
    return Panel(t, title="project page", border_style="cyan")


# ---- saving ----------------------------------------------------------------
def apply_winner_css(sysm: Sys, css: str, gh_repo: Optional[str], rep: Report, *, env_path: str = ENV_FILE) -> None:
    if sysm.exists(env_path) or sysm.sandbox:
        sysm.ensure_sudo()
        env_set(sysm, {"WINNER_CSS": css}, env_path)
        rep.add(env_path, "ok" if not sysm.dry else "skip", "WINNER_CSS written" if css else "WINNER_CSS cleared")
    else:
        rep.add(env_path, "skip", f"not on the Pi. There, run: moneyball_ops.py probe-studio --apply-css {shlex.quote(css)}")
    gh = GH(gh_repo or load_state().get("gh_repo"))
    ok, why = gh.ready()
    if not ok:
        rep.add("GitHub variable WINNER_CSS", "skip", f"{why}. By hand: gh variable set WINNER_CSS --body {shlex.quote(css)}")
        return
    gh.repo = gh.repo or gh.detect_repo()
    if not gh.repo:
        rep.add("GitHub variable WINNER_CSS", "skip", "no repo known (use --gh-repo OWNER/NAME)")
        return
    if sysm.dry:
        rep.add("GitHub variable WINNER_CSS", "skip", "dry run")
        return
    if css:
        gh.set_var("WINNER_CSS", css)
        back = gh.list_vars().get("WINNER_CSS")
        rep.add(f"GitHub {gh.repo} WINNER_CSS", "ok" if back == css else "fail", "set" if back == css else f"reads back as {back!r}")
    else:
        gh._run(["variable", "delete", "WINNER_CSS"], check=False)
        rep.add(f"GitHub {gh.repo} WINNER_CSS", "ok", "removed (unset behaves as empty)")


def save_flow(sysm: Sys, page: Page, css: str, expect: Optional[int], mod: Any, gh_repo: Optional[str]) -> Optional[int]:
    """Verify, then persist. Returns the expect value used, or None if nothing was saved."""
    if page.kind != "gallery":
        console.print("[yellow]Saving needs a gallery page, because the check is a winner count. :load a gallery URL first.[/]")
        return None
    if expect is None:
        expect = IntPrompt.ask("How many Winner badges do you count on this page in your browser")
    m = compute_metrics(page, css, expect)
    console.print(render_metrics(page, css, expect, mod))
    if m["verdict"] != "PASS":
        if not Confirm.ask("The check is not a PASS. Save anyway and mark it verified?", default=False):
            return None
    if mod is not None:
        slugs, _ = pipeline_gallery(mod, page.html, css)
        if slugs != {page.cards[i].slug for i in m["union"]}:
            if not Confirm.ask("pipeline.py disagrees with this view. Save anyway?", default=False):
                return None
    rep = Report()
    console.print("\n[bold]Saving[/]")
    apply_winner_css(sysm, css, gh_repo, rep)
    save_state({"probe": {"verified": True, "css": css, "expect": expect, "cards": m["M"], "flagged": len(m["union"]),
                          "url": page.source, "at": iso(), "parity": mod is not None}})
    rep.add("Probe recorded", "ok", "timer enable is now allowed on this machine (if canary also passed)")
    console.print(rep.table("probe-studio save"))
    return expect


# ---- the REPL ---------------------------------------------------------------
def repl(sysm: Sys, page: Page, css: str, expect: Optional[int], mod: Any, a: argparse.Namespace, fetcher: Optional[Fetcher]) -> int:
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401  (gives arrow keys and history to input())
    console.print(Panel(HELP_TEXT, title="probe-studio", border_style="cyan"))

    def show() -> None:
        if page.kind == "gallery":
            console.print(render_metrics(page, css, expect, mod))
            console.print(render_cards(page, css, 6))
        else:
            console.print(render_project(page, css, mod))

    show()
    while True:
        try:
            line = Prompt.ask("[bold cyan]selector[/]", default="", show_default=False).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not line:
            continue
        if line in ("q", ":q", ":quit", ":exit"):
            return 0
        if not line.startswith(":"):
            css = line
            show()
            continue
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd == "help":
            console.print(HELP_TEXT)
        elif cmd == "expect":
            if arg.isdigit():
                expect = int(arg)
                show()
            else:
                console.print("usage: :expect 3")
        elif cmd == "clear":
            css = ""
            show()
        elif cmd == "cards":
            n = int(arg) if arg.isdigit() else 20
            console.print(render_cards(page, css, n))
        elif cmd == "card":
            if arg.isdigit() and 1 <= int(arg) <= len(page.cards):
                html = page.cards[int(arg) - 1].node.html or ""
                console.print(Syntax(html[:2500] + (" ..." if len(html) > 2500 else ""), "html", word_wrap=True))
            else:
                console.print(f"usage: :card N  (1 to {len(page.cards)})")
        elif cmd == "classes":
            rows = class_census(page)
            if not rows:
                console.print("no class names containing winner, prize, badge, award, trophy, ribbon or medal in these cards. "
                              "Try :card 1 to read the markup, or :suggest.")
            for tkn, n in rows[:15]:
                console.print(f"  .{esc(tkn)}   in {n} of {len(page.cards)} cards")
        elif cmd == "suggest":
            if page.kind != "gallery":
                console.print("suggestions need a gallery page")
                continue
            rows = suggest_selectors(page, expect)
            if expect is None:
                console.print("[yellow]Set :expect N first so the ranking knows the target.[/]")
            if not rows:
                console.print("No single class name separates the cards in a useful way. Read :card 1 and write a selector by hand.")
            for r in rows:
                console.print(f"  {esc(r['selector']):<34} alone {r['alone']:>3}   built-in OR it = {r['union']:>3}" +
                              (f"   off by {r['delta']}" if expect is not None else ""))
        elif cmd == "load":
            try:
                page = load_page(arg, fetcher, a.delay)
                css = ""
                show()
            except OpsError as e:
                console.print(f"[red]{esc(e)}[/]")
        elif cmd == "save":
            used = save_flow(sysm, page, css, expect, mod, a.gh_repo)
            if used is not None:
                expect = used
                return 0
        else:
            console.print(f"unknown command :{esc(cmd)}  (:help)")


def cmd_probe_studio(a: argparse.Namespace) -> int:
    sysm = Sys(dry=a.dry_run, root=a.sandbox_root, yes=a.yes)
    if a.apply_css is not None:
        console.print(Panel.fit("This writes WINNER_CSS without testing it here.", title="apply-css", border_style="yellow"))
        if not (a.yes or Confirm.ask("Did you verify this selector with probe-studio against a real gallery on another machine?", default=False)):
            return 1
        rep = Report()
        apply_winner_css(sysm, a.apply_css, a.gh_repo, rep)
        st = load_state().get("probe") or {}
        save_state({"probe": {**st, "verified": True, "css": a.apply_css, "at": iso(), "applied_without_test": True}})
        console.print(rep.table("apply-css"))
        return 0
    fetcher: Optional[Fetcher] = None
    src = a.url or a.html
    if a.suggest:
        fetcher = Fetcher(a.delay)
        hs = suggest_hackathons(fetcher, a.pages)
        t = Table(title="Big finished hackathons with winners announced (open one in your browser first)", box=box.SIMPLE_HEAVY)
        t.add_column("registrations", justify="right")
        t.add_column("title")
        t.add_column("gallery")
        for h in hs:
            t.add_row(str(h.get("registrations_count")), esc((h.get("title") or "")[:40]), esc(h.get("submission_gallery_url")))
        console.print(t)
        if not src and sys.stdin.isatty():
            src = Prompt.ask("Gallery URL to probe (Enter to stop)", default="") or None
        if not src:
            return 0
    if not src:
        if sys.stdin.isatty():
            src = Prompt.ask("Completed hackathon gallery URL (or a project URL, or a saved .html file)")
        else:
            raise OpsError("give a URL or --html FILE")
    if not Path(src).is_file() and fetcher is None:
        fetcher = Fetcher(a.delay)
    page = load_page(src, fetcher, a.delay)
    mod = load_pipeline(a.pipeline)
    css, expect = a.selector or "", a.expect
    if a.report or not sys.stdin.isatty():
        if page.kind == "gallery":
            console.print(render_metrics(page, css, expect, mod))
            console.print(render_cards(page, css, 10))
            return 0 if compute_metrics(page, css, expect)["verdict"] == "PASS" else 1
        console.print(render_project(page, css, mod))
        return 0
    return repl(sysm, page, css, expect, mod, a, fetcher)


# ----------------------------------------------------------------------------- dashboard: data sources
class Source:
    """Reads ledger and data tables. Files in the dataset are immutable, so parsed tables are cached by path."""

    def __init__(self) -> None:
        self._cache: dict[tuple, list[dict]] = {}

    def sync(self, totals: bool) -> None:  # pragma: no cover - overridden
        pass

    def files(self, prefix: str) -> list[Path]:  # pragma: no cover - overridden
        return []

    def read_rows(self, prefix: str, columns: Optional[list[str]] = None) -> list[dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise OpsError("pyarrow is not installed: pip install pyarrow")
        rows: list[dict] = []
        for p in self.files(prefix):
            key = (str(p), tuple(columns or ()))
            got = self._cache.get(key)
            if got is None:
                names = pq.read_schema(p).names
                cols = [c for c in columns if c in names] if columns else None
                got = pq.read_table(p, columns=cols).to_pylist()
                self._cache[key] = got
            rows += got
        return rows

    def size_bytes(self) -> int:
        return 0


class LocalSource(Source):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise OpsError(f"{root} is not a folder")

    def files(self, prefix: str) -> list[Path]:
        return sorted((self.root / prefix).glob("*.parquet"))


class HFSource(Source):
    def __init__(self, repo: str, token: str, cache: Path) -> None:
        super().__init__()
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise OpsError("huggingface_hub is not installed: pip install huggingface_hub")
        self.repo, self.token, self.cache = repo, token, cache
        self.api = HfApi(token=token)
        self.remote: set[str] = set()
        cache.mkdir(parents=True, exist_ok=True)

    def sync(self, totals: bool) -> None:
        from huggingface_hub import snapshot_download
        self.remote = set(self.api.list_repo_files(self.repo, repo_type="dataset"))
        pats = ["ledger/*"] + (["data/hackathons/*", "data/cards/*"] if totals else [])
        snapshot_download(self.repo, repo_type="dataset", allow_patterns=pats, local_dir=str(self.cache), token=self.token)

    def files(self, prefix: str) -> list[Path]:
        return [p for p in sorted((self.cache / prefix).glob("*.parquet")) if f"{prefix}/{p.name}" in self.remote]

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.cache.rglob("*.parquet"))


# ----------------------------------------------------------------------------- dashboard: analysis
def recommended_delay(current: Optional[float]) -> float:
    """Double the current delay, never below 4 s (the runbook's first step) or above 10 s."""
    base = current if current else 2.0
    return float(min(max(base * 2, 4.0), 10.0))


def analyze(ledger: list[dict], hacks: Optional[list[dict]], cards: Optional[list[dict]], *, now: dt.datetime,
            nshards: int = 5, max_attempts: int = 5, min_reg: int = 0, active_min: int = 30, quality_min: int = 20) -> dict:
    by_stage: dict[str, list[dict]] = {s: [] for s in STAGES}
    for r in ledger:
        if r.get("stage") in by_stage:
            by_stage[r["stage"]].append(r)
    for s in STAGES:
        by_stage[s].sort(key=lambda r: r.get("ts") or "")
    done: dict[str, set] = {s: set() for s in STAGES}
    status: dict[str, dict] = {s: {} for s in STAGES}
    attempts: dict[str, Counter] = {s: Counter() for s in STAGES}
    last_row: dict[str, dict] = {s: {} for s in STAGES}
    for s in STAGES:
        for r in by_stage[s]:
            k, st = r["key"], r["status"]
            last_row[s][k] = r
            if st in FINAL:
                done[s].add(k)
                status[s].setdefault(k, st)
            else:
                attempts[s][k] += 1
        for k, r in last_row[s].items():
            if k not in done[s]:
                status[s][k] = r["status"]
    # what the work list is
    elig_ids: Optional[set] = None
    latest: dict = {}
    if hacks is not None:
        for h in hacks:
            hid = h["hackathon_id"]
            if hid not in latest or (h.get("fetched_at") or "") >= (latest[hid].get("fetched_at") or ""):
                latest[hid] = h
        elig_ids = {str(h["hackathon_id"]) for h in latest.values()
                    if h.get("winners_announced") and h.get("gallery_url") and (h.get("registrations") or 0) >= min_reg}
    sel_ids: Optional[set] = {c["slug"] for c in cards if c.get("detail_selected")} if cards is not None else None
    scope = {"gallery": elig_ids, "detail": sel_ids}
    stages: dict[str, dict] = {}
    for s in STAGES:
        cnt = Counter(status[s].values())
        sc = scope.get(s)
        n_done = len(done[s] & sc) if sc is not None else len(done[s])
        total = len(sc) if sc is not None else None
        rows = by_stage[s]
        recent = 0
        for r in rows:
            ts = parse_ts(r.get("ts"))
            if ts and r["status"] in FINAL and (now - ts).total_seconds() <= 86400:
                recent += 1
        first = parse_ts(rows[0]["ts"]) if rows else None
        win_h = min(24.0, max(1.0, ((now - first).total_seconds() / 3600) if first else 1.0))
        rate = recent / win_h
        remaining = (total - n_done) if total is not None else None
        eta = (remaining / rate) if (remaining and rate > 0) else None
        stages[s] = {"done": n_done, "total": total, "pct": (100.0 * n_done / total) if total else None,
                     "counts": dict(cnt), "rate_h": rate, "eta_h": eta, "rows": len(rows),
                     "complete": (any(r["status"] == "empty" for r in rows) if s == "discover" else None)}
    # workers
    wrows: dict[str, list] = defaultdict(list)
    for s in STAGES:
        for r in by_stage[s]:
            ts = parse_ts(r.get("ts"))
            if ts:
                wrows[r.get("worker") or "?"].append((ts, r["status"], s, r["key"], str(r.get("note") or "")))
    workers = []
    for w, rows in sorted(wrows.items()):
        rows.sort(key=lambda x: x[0])
        last = rows[-1]
        silent = (now - last[0]).total_seconds() / 60
        c1 = Counter(x[1] for x in rows if (now - x[0]).total_seconds() <= 3600)
        c24 = Counter(x[1] for x in rows if (now - x[0]).total_seconds() <= 86400)
        n1 = sum(c1.values())
        codes = Counter(x[4] for x in rows if x[1] in ("blocked", "error") and (now - x[0]).total_seconds() <= 86400)
        if last[1] == "blocked" and silent >= active_min:
            state = "TRIPPED"
        elif n1 >= 3 and c1["blocked"] / n1 >= 0.5 and silent < active_min:
            state = "BLOCKED"
        elif n1 >= 10 and (c1["blocked"] + c1["error"]) / n1 >= 0.2:
            state = "DEGRADED"
        elif silent < active_min:
            state = "ACTIVE"
        else:
            state = "IDLE"
        workers.append({"name": w, "state": state, "last_ts": iso(last[0]), "silent_min": silent,
                        "last_status": last[1], "keys_h": sum(c1[k] for k in FINAL), "ok24": c24["ok"],
                        "gone24": c24["gone"], "blocked24": c24["blocked"], "error24": c24["error"],
                        "codes": codes.most_common(3)})
    # shards
    shards = None
    if elig_ids is not None or sel_ids is not None:
        shards = []
        poisoned_keys = {(s, k) for s in ("gallery", "detail") for k, n in attempts[s].items()
                         if k not in done[s] and n >= max_attempts}
        for sh in range(nshards):
            g_tot = [k for k in (elig_ids or ()) if shard_of(k, nshards) == sh]
            d_tot = [k for k in (sel_ids or ()) if shard_of(k, nshards) == sh]
            shards.append({"shard": sh, "usual": "pi5" if sh == nshards - 1 else f"gha-{sh}",
                           "gallery_done": sum(1 for k in g_tot if k in done["gallery"]), "gallery_total": len(g_tot),
                           "detail_done": sum(1 for k in d_tot if k in done["detail"]), "detail_total": len(d_tot),
                           "poisoned": sum(1 for (s, k) in poisoned_keys if shard_of(k, nshards) == sh)})
    # poisoned keys
    poison = []
    for s in ("gallery", "detail"):
        for k, n in attempts[s].items():
            if k in done[s] or n < max_attempts:
                continue
            r = last_row[s][k]
            poison.append({"stage": s, "key": k, "attempts": n, "status": r["status"], "note": str(r.get("note") or "")[:60],
                           "ts": r.get("ts"), "shard": shard_of(k, nshards)})
    poison.sort(key=lambda p: (p["stage"], -p["attempts"], p["key"]))
    # gallery quality: notes of finished galleries
    quality = None
    ok_g = [r for k, r in last_row["gallery"].items() if status["gallery"].get(k) == "ok"]
    if ok_g:
        zero = short = 0
        worst: list = []
        for r in ok_g:
            try:
                n = json.loads(r.get("note") or "{}")
            except ValueError:
                continue
            if n.get("winners") == 0:
                zero += 1
            exp, got = n.get("expected"), n.get("cards") or 0
            if exp and got < 0.9 * exp:
                short += 1
                worst.append((exp - got, r["key"], got, exp))
        worst.sort(reverse=True)
        quality = {"galleries": len(ok_g), "zero_winners": zero, "short": short, "worst_short": worst[:5]}
    # alerts
    alerts: list[dict] = []
    for w in workers:
        codes = ", ".join(f"{c}x{n}" for c, n in w["codes"]) or "no codes"
        if w["state"] == "TRIPPED":
            alerts.append({"level": "crit", "code": "breaker", "worker": w["name"], "action": "delay",
                           "title": f"{w['name']}: circuit breaker most likely tripped",
                           "detail": f"Its last write ({fmt_age(w['silent_min'] * 60)} ago) was a blocked key, then it went quiet. Codes: {codes}. "
                                     "Runbook 4.6: do not rerun at once. Wait a few hours, run canary, then raise DELAY."})
        elif w["state"] == "BLOCKED":
            alerts.append({"level": "crit", "code": "blocked-now", "worker": w["name"], "action": "delay",
                           "title": f"{w['name']}: being blocked right now",
                           "detail": f"Most of its last hour of keys came back blocked. Codes: {codes}. The breaker will stop it after 8 blocks in a row."})
        elif w["state"] == "DEGRADED":
            alerts.append({"level": "warn", "code": "degraded", "worker": w["name"], "action": "delay",
                           "title": f"{w['name']}: many failures in the last hour",
                           "detail": f"Blocked or error share is 20% or more. Codes: {codes}. Consider a higher DELAY."})
    if quality and quality["galleries"] >= quality_min:
        zs = quality["zero_winners"] / quality["galleries"]
        if zs >= 0.6:
            alerts.append({"level": "crit", "code": "no-winners", "action": None,
                           "title": f"{zs:.0%} of finished galleries have zero winners flagged",
                           "detail": "The winner badge is probably not being detected. Re-run probe-studio on a real gallery before more crawling."})
        elif zs >= 0.25:
            alerts.append({"level": "warn", "code": "few-winners", "action": None,
                           "title": f"{zs:.0%} of finished galleries have zero winners flagged",
                           "detail": "Some events legitimately have none, but this is high. Spot check with probe-studio."})
        ss = quality["short"] / quality["galleries"]
        if ss >= 0.1:
            alerts.append({"level": "warn", "code": "short-galleries", "action": None,
                           "title": f"{ss:.0%} of galleries hold fewer cards than the page's own total",
                           "detail": "The ?page=N pagination guess may be wrong (it was not verified). Worst: " +
                                     ", ".join(f"hackathon {k} got {g} of {e}" for _, k, g, e in quality["worst_short"][:3])})
    if poison:
        alerts.append({"level": "warn", "code": "poison", "action": None,
                       "title": f"{len(poison)} key(s) failed {max_attempts} times and are being skipped",
                       "detail": "Press p to list them. After the cause is fixed, retry with --max-attempts 10 on the run commands."})
    work_left = any((stages[s]["total"] or 0) > stages[s]["done"] for s in ("gallery", "detail"))
    if workers and work_left and all(w["silent_min"] > 36 * 60 for w in workers):
        alerts.append({"level": "warn", "code": "silent", "action": None,
                       "title": f"no worker has written for {fmt_age(min(w['silent_min'] for w in workers) * 60)}",
                       "detail": "Is the Actions workflow disabled or the Pi timer off? (`timer status`, `gh workflow list`)"})
    if not ledger:
        alerts.append({"level": "info", "code": "empty", "action": None, "title": "the ledger is empty",
                       "detail": "Nothing has been written yet. Run discover first (runbook 4.2)."})
    level = "crit" if any(a["level"] == "crit" for a in alerts) else ("warn" if any(a["level"] == "warn" for a in alerts) else "ok")
    return {"generated": iso(now), "level": level, "stages": stages, "workers": workers, "shards": shards, "poison": poison,
            "quality": quality, "alerts": alerts, "hackathons": len(latest) if hacks is not None else None,
            "eligible": len(elig_ids) if elig_ids is not None else None,
            "selected": len(sel_ids) if sel_ids is not None else None, "nshards": nshards}


# ----------------------------------------------------------------------------- dashboard: evidence outside the ledger
def local_pi_info() -> Optional[dict]:
    """Service, timer and journal facts when this machine is the Pi."""
    if not shutil.which("systemctl") or not Path(UNIT_DIR, "moneyball.service").exists():
        return None

    def show(unit: str, props: str) -> dict:
        cp = subprocess.run(["systemctl", "show", "-p", props, unit], capture_output=True, text=True, timeout=10)
        return dict(ln.split("=", 1) for ln in cp.stdout.splitlines() if "=" in ln)

    info: dict = {"service": show("moneyball.service", "ActiveState,SubState,Result"),
                  "timer": show("moneyball.timer", "ActiveState,UnitFileState,NextElapseUSecRealtime"), "journal_trips": None}
    try:
        j = subprocess.run(["journalctl", "-u", "moneyball", "--since", "36 hours ago", "--no-pager", "-o", "cat"],
                           capture_output=True, text=True, timeout=15)
        if j.returncode == 0:
            info["journal_trips"] = sum(1 for ln in j.stdout.splitlines() if "cooling down" in ln)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def gha_evidence(gh: GH) -> dict:
    """Recent workflow runs and any 'blocked, cooling down' annotations. Best effort: this reads GitHub's
    annotation API, which was not tested against a live repo when this tool was written."""
    out: dict = {"runs": [], "trips": [], "error": None}
    try:
        cp = gh._run(["run", "list", "--workflow=ingest.yml", "--limit", "4", "--json", "databaseId,status,conclusion,createdAt,event"], timeout=30)
        runs = json.loads(cp.stdout or "[]")
        out["runs"] = runs
        for r in runs:
            if r.get("status") != "completed":
                continue
            jobs = gh._run(["api", f"repos/{gh.repo}/actions/runs/{r['databaseId']}/jobs", "--jq", ".jobs[] | [.id,.name] | @tsv"],
                           use_repo=False, timeout=30).stdout
            for ln in jobs.splitlines():
                jid, _, name = ln.partition("\t")
                ann = gh._run(["api", f"repos/{gh.repo}/check-runs/{jid}/annotations"], check=False, use_repo=False, timeout=30)
                for an in json.loads(ann.stdout or "[]"):
                    if "cooling down" in (an.get("message") or ""):
                        out["trips"].append({"run": r["databaseId"], "job": name, "at": r.get("createdAt"), "message": an["message"][:120]})
    except (OpsError, ValueError, KeyError) as e:
        out["error"] = str(e)[:160]
    return out


# ----------------------------------------------------------------------------- dashboard: rendering
STATE_STYLE = {"ACTIVE": "green", "IDLE": "dim", "DEGRADED": "yellow", "BLOCKED": "bold red", "TRIPPED": "bold red"}


def pct_bar(pct: Optional[float], width: int = 24, color: str = "green") -> Any:
    if pct is None:
        return Text("n/a", style="dim")
    return ProgressBar(total=100, completed=min(pct, 100), width=width, complete_style=color, finished_style=color)


def render_dashboard(snap: Optional[dict], meta: dict, view: str = "main") -> Any:
    age = (time.time() - meta["updated"]) if meta.get("updated") else None
    head = Text.assemble(("moneyball ops", "bold cyan"), f"   {meta.get('repo', '')}   ",
                         ("updated " + fmt_age(age) + " ago" if age is not None else "waiting for the first load ...", "dim"),
                         ("   [loading]" if meta.get("busy") else "", "yellow"))
    parts: list[Any] = [Panel(head, border_style="cyan", padding=(0, 1))]
    if meta.get("error"):
        parts.append(Panel(Text(meta["error"], style="red"), title="last refresh failed (showing older data)", border_style="red"))
    if snap is None:
        parts.append(Text("  loading the ledger from Hugging Face, the first load downloads the ledger files ...", style="dim"))
        return Group(*parts)
    if view == "poison":
        t = Table(title=f"Poisoned keys ({len(snap['poison'])})", box=box.SIMPLE_HEAVY)
        for h in ("stage", "key", "tries", "last status", "note", "last try (UTC)", "shard"):
            t.add_column(h, overflow="fold")
        for p in snap["poison"][:60]:
            t.add_row(p["stage"], esc(p["key"]), str(p["attempts"]), p["status"], esc(p["note"]), esc(p["ts"]), str(p["shard"]))
        parts += [t, Text("  more than 60 are not shown. Retry after fixing the cause with --max-attempts 10.   [p] back  [q] quit", style="dim")]
        return Group(*parts)
    if snap["alerts"]:
        lines = []
        for a in snap["alerts"]:
            style = {"crit": "bold red", "warn": "yellow", "info": "cyan"}[a["level"]]
            lines.append(Text.assemble((f"{a['level'].upper():<5} ", style), (a["title"], "bold"), "\n      " + a["detail"]
                                       + ("   [press d to change DELAY]" if a.get("action") == "delay" else "")))
        parts.append(Panel(Group(*lines), title="alerts", border_style={"crit": "red", "warn": "yellow", "ok": "green"}[snap["level"]]))
    else:
        parts.append(Panel(Text("no alerts", style="green"), border_style="green", padding=(0, 1)))
    st = Table(box=box.SIMPLE_HEAVY, title="Progress")
    for h in ("stage", "progress", "done / total", "%", "status mix", "keys/h", "ETA"):
        st.add_column(h, overflow="fold")
    for s in STAGES:
        d = snap["stages"][s]
        if s == "discover":
            prog = Text("complete" if d["complete"] else f"{d['done']} pages", style="green" if d["complete"] else "yellow")
            dt_ = f"{d['done']:,} pages"
            pct = "-"
        else:
            prog = pct_bar(d["pct"])
            dt_ = f"{d['done']:,} / {d['total']:,}" if d["total"] is not None else f"{d['done']:,} / ?"
            pct = f"{d['pct']:.1f}%" if d["pct"] is not None else "?"
        mix = " ".join(f"{k}={v}" for k, v in sorted(d["counts"].items()))
        eta = f"~{d['eta_h'] / 24:.1f} d" if d["eta_h"] and d["eta_h"] > 36 else (f"~{d['eta_h']:.0f} h" if d["eta_h"] else "-")
        st.add_row(s, prog, dt_, pct, esc(mix), f"{d['rate_h']:.0f}", eta)
    parts.append(st)
    wt = Table(box=box.SIMPLE_HEAVY, pad_edge=False, title="Workers (counts are the last 24 h; one row is one key tried)")
    for h in ("worker", "state", "last write", "keys/h", "ok", "gone", "blocked", "error", "codes 24h"):
        if h == "state":
            wt.add_column(h, no_wrap=True, min_width=8)
        elif h == "worker":
            wt.add_column(h, overflow="ellipsis", no_wrap=True, max_width=12)
        elif h == "codes 24h":
            wt.add_column("codes", overflow="fold", min_width=8)
        else:
            wt.add_column(h, no_wrap=True)
    for w in snap["workers"]:
        wt.add_row(esc(w["name"]), Text(w["state"], style=STATE_STYLE.get(w["state"], "")), fmt_age(w["silent_min"] * 60) + " ago",
                   str(w["keys_h"]), f"{w['ok24']:,}", str(w["gone24"]), str(w["blocked24"]), str(w["error24"]),
                   esc(", ".join(f"{c}x{n}" for c, n in w["codes"]) or "-"))
    parts.append(wt)
    if snap["shards"]:
        sh = Table(box=box.SIMPLE_HEAVY, title=f"Shards (by key hash, {snap['nshards']} total)")
        for h in ("shard", "usual worker", "galleries", "detail pages", "poisoned"):
            sh.add_column(h)
        for r in snap["shards"]:
            sh.add_row(str(r["shard"]), r["usual"], f"{r['gallery_done']:,} / {r['gallery_total']:,}",
                       f"{r['detail_done']:,} / {r['detail_total']:,}", Text(str(r["poisoned"]), style="yellow" if r["poisoned"] else ""))
        parts.append(sh)
    q = snap.get("quality")
    if q:
        parts.append(Text(f"  finished galleries: {q['galleries']:,}   with zero winners flagged: {q['zero_winners']:,}   "
                          f"shorter than the page's own total: {q['short']:,}", style="dim"))
    if meta.get("pi"):
        pi = meta["pi"]
        parts.append(Text(f"  this Pi: service {pi['service'].get('ActiveState', '?')}/{pi['service'].get('SubState', '?')}, "
                          f"timer {pi['timer'].get('UnitFileState', '?')}, next run {pi['timer'].get('NextElapseUSecRealtime') or 'none'}"
                          + (f", journal breaker messages (36h): {pi['journal_trips']}" if pi.get("journal_trips") is not None else ""), style="dim"))
    if meta.get("gha"):
        g = meta["gha"]
        if g.get("error"):
            parts.append(Text(f"  Actions: could not read ({g['error'][:80]})", style="dim"))
        else:
            runs = ", ".join(f"{r.get('conclusion') or r.get('status')}" for r in g["runs"][:4]) or "no runs"
            parts.append(Text(f"  Actions, latest runs: {runs}   cooling-down annotations: {len(g['trips'])}", style="dim"))
    parts.append(Text("  [r] refresh now   [d] change DELAY   [p] poisoned keys   [q] quit"
                      + (f"   cache {human_bytes(meta['cache'])}" if meta.get("cache") else ""), style="bold dim"))
    return Group(*parts)


def merge_evidence(snap: dict, meta: dict) -> None:
    """Fold direct evidence (Pi journal, Actions annotations) into the alerts."""
    pi, gha = meta.get("pi") or {}, meta.get("gha") or {}
    if pi.get("journal_trips"):
        snap["alerts"].append({"level": "crit", "code": "journal", "action": "delay", "title": f"Pi journal: breaker message seen {pi['journal_trips']} time(s) in 36 h",
                               "detail": "'blocked, cooling down until next run' was logged by the Pi service. Confirmed trip."})
    if gha.get("trips"):
        names = ", ".join(sorted({t["job"] for t in gha["trips"]}))
        snap["alerts"].append({"level": "crit", "code": "annotation", "action": "delay", "title": f"Actions: {len(gha['trips'])} cooling-down annotation(s) in recent runs",
                               "detail": f"Jobs: {names}. Confirmed trip."})
    if snap["alerts"] and any(a["level"] == "crit" for a in snap["alerts"]):
        snap["level"] = "crit"


# ----------------------------------------------------------------------------- dashboard: DELAY change
def delay_flow(a: argparse.Namespace, sysm: Sys) -> None:
    console.print(Rule("Change DELAY"))
    console.print("Runbook 4.6: do not rerun straight away. Wait a few hours, run canary, then slow down.\n"
                  "Jobs that are running keep their old delay. The new value applies from the next start.\n")
    gh = GH(a.gh_repo or load_state().get("gh_repo"))
    gh_ok = False
    cur_gh: Optional[float] = None
    if not a.no_gha:
        ok, why = gh.ready()
        if ok:
            gh.repo = gh.repo or gh.detect_repo()
            gh_ok = bool(gh.repo)
        if gh_ok:
            v = gh.get_var("DELAY")
            with contextlib.suppress(TypeError, ValueError):
                cur_gh = float(v) if v else None
        console.print(f"GitHub Actions variable DELAY on {esc(gh.repo) if gh_ok else '(unavailable: ' + esc(why) + ')'}: {cur_gh if cur_gh is not None else 'not set (default 2.0)'}")
    env_txt = sysm.read_text(ENV_FILE) if (sysm.exists(ENV_FILE)) else None
    cur_pi: Optional[float] = None
    if env_txt is not None:
        with contextlib.suppress(TypeError, ValueError):
            cur_pi = float(env_parse(env_txt).get("DELAY", ""))
        console.print(f"This machine's {ENV_FILE} DELAY: {cur_pi if cur_pi is not None else 'not set'}")
    if not gh_ok and env_txt is None:
        console.print("[yellow]Nothing to change from here: no usable gh and no /etc/moneyball.env on this machine.[/]")
        Prompt.ask("press Enter", default="")
        return
    sug = recommended_delay(cur_gh if cur_gh is not None else (cur_pi - 0.5 if cur_pi else None))
    new_gh = FloatPrompt.ask("New DELAY in seconds for Actions", default=sug) if gh_ok else None
    new_pi = FloatPrompt.ask("New DELAY in seconds for this Pi", default=(new_gh + 0.5 if new_gh else sug + 0.5)) if env_txt is not None else None
    for label, v in (("Actions", new_gh), ("Pi", new_pi)):
        if v is not None and not (0.5 <= v <= 120):
            console.print(f"[red]{label}: {v} is outside 0.5 to 120 seconds, nothing changed[/]")
            Prompt.ask("press Enter", default="")
            return
    if not Confirm.ask("Apply?", default=True):
        return
    rep = Report()
    if new_gh is not None:
        try:
            gh.set_var("DELAY", str(new_gh))
            rep.add(f"GitHub {gh.repo} DELAY", "ok" if gh.list_vars().get("DELAY") == str(new_gh) else "warn", f"{new_gh}")
        except OpsError as e:
            rep.add("GitHub DELAY", "fail", str(e).splitlines()[0])
    if new_pi is not None:
        try:
            sysm.ensure_sudo()
            env_set(sysm, {"DELAY": str(new_pi)})
            rep.add(f"{ENV_FILE} DELAY", "ok", f"{new_pi} (read fresh on every start, no daemon reload needed)")
        except OpsError as e:
            rep.add(f"{ENV_FILE} DELAY", "fail", str(e).splitlines()[0])
    console.print(rep.table("DELAY"))
    Prompt.ask("press Enter to return to the dashboard", default="")


# ----------------------------------------------------------------------------- dashboard: runner
class KeyReader:
    """Single key reads without Enter (Unix terminals). Arrow keys and other escape sequences are swallowed."""

    def __init__(self) -> None:
        self.ok, self.fd, self.old = False, -1, None

    def __enter__(self) -> "KeyReader":
        try:
            import termios
            import tty
            if sys.stdin.isatty():
                self.fd = sys.stdin.fileno()
                self.old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.ok = True
        except Exception:
            self.ok = False
        return self

    def __exit__(self, *exc: Any) -> None:
        self.suspend()

    def suspend(self) -> None:
        if self.ok and self.old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def resume(self) -> None:
        if self.ok:
            import tty
            tty.setcbreak(self.fd)

    def get(self, timeout: float) -> Optional[str]:
        if not self.ok:
            time.sleep(timeout)
            return None
        import select
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self.fd, 32)
        if data.startswith(b"\x1b"):
            return "ESC"
        return data.decode(errors="ignore")[:1] or None


class Poller(threading.Thread):
    def __init__(self, fetch: Callable[[], tuple[dict, dict]], interval: float) -> None:
        super().__init__(daemon=True)
        self.fetch, self.interval = fetch, interval
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.snap: Optional[dict] = None
        self.meta: dict = {"busy": True}
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            with self.lock:
                self.meta["busy"] = True
            try:
                snap, meta = self.fetch()
                with self.lock:
                    self.snap, self.meta = snap, {**meta, "busy": False, "updated": time.time(), "error": None}
            except Exception as e:  # keep the last good data on screen
                with self.lock:
                    self.meta = {**self.meta, "busy": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            self.wake.wait(self.interval)
            self.wake.clear()

    def view(self) -> tuple[Optional[dict], dict]:
        with self.lock:
            return self.snap, dict(self.meta)


def resolve_hf_access(a: argparse.Namespace, sysm: Sys) -> tuple[str, str]:
    st = load_state()
    repo = a.repo or os.environ.get("HF_REPO") or st.get("hf_repo")
    token = os.environ.get("HF_TOKEN")
    if (not repo or not token) and sysm.exists(ENV_FILE) and sys.stdin.isatty():
        if Confirm.ask(f"Read the repo and token stored in {ENV_FILE} (needs sudo)?", default=True):
            sysm.ensure_sudo()
            vals = env_parse(sysm.read_text(ENV_FILE) or "")
            repo = repo or vals.get("HF_REPO")
            token = token or vals.get("HF_TOKEN")
    if not token:
        with contextlib.suppress(Exception):
            from huggingface_hub import get_token
            token = get_token()
    if not repo or repo == PLACEHOLDER_REPO:
        if not sys.stdin.isatty():
            raise OpsError("no dataset repo: pass --repo USER/NAME or set HF_REPO")
        repo = Prompt.ask("Hugging Face dataset repo [dim](USER/NAME)[/]")
    if not token or token == PLACEHOLDER_TOKEN:
        if not sys.stdin.isatty():
            raise OpsError("no token: set HF_TOKEN (a read-only token is enough)")
        token = Prompt.ask("Hugging Face token [dim](read-only is enough, input hidden)[/]", password=True).strip()
    return repo, token


def default_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "moneyball_ops"


def cmd_dashboard(a: argparse.Namespace) -> int:
    sysm = Sys(dry=a.dry_run, root=a.sandbox_root, yes=a.yes)
    totals = not a.no_totals
    if a.local:
        src: Source = LocalSource(a.local)
        repo_label = f"local folder {a.local}"
    else:
        repo, token = resolve_hf_access(a, sysm)
        src = HFSource(repo, token, Path(a.cache) if a.cache else default_cache())
        repo_label = repo
        token = ""
    gh: Optional[GH] = None
    if not a.no_gha and not a.local:
        g = GH(a.gh_repo or load_state().get("gh_repo"))
        if g.ready()[0]:
            g.repo = g.repo or g.detect_repo()
            gh = g if g.repo else None
    state = {"gha_at": 0.0, "gha": None}

    def fetch() -> tuple[dict, dict]:
        src.sync(totals)
        ledger: list[dict] = []
        for s in STAGES:
            ledger += src.read_rows(f"ledger/{s}")
        hacks = src.read_rows("data/hackathons", ["hackathon_id", "winners_announced", "gallery_url", "registrations", "fetched_at"]) if totals else None
        cards = src.read_rows("data/cards", ["hackathon_id", "slug", "detail_selected"]) if totals else None
        snap = analyze(ledger, hacks, cards, now=utcnow(), nshards=a.nshards, max_attempts=a.max_attempts, min_reg=a.min_registrations)
        meta: dict = {"repo": repo_label, "cache": src.size_bytes()}
        with contextlib.suppress(Exception):
            meta["pi"] = local_pi_info()
        if gh is not None and time.time() - state["gha_at"] > 300:
            state["gha"], state["gha_at"] = gha_evidence(gh), time.time()
        meta["gha"] = state["gha"]
        merge_evidence(snap, meta)
        return snap, meta

    interactive = sys.stdout.isatty() and sys.stdin.isatty() and not a.once and not a.json
    if not interactive:
        snap, meta = fetch()
        if a.json:
            print(json.dumps(snap, indent=2, default=str))
        else:
            console.print(render_dashboard(snap, {**meta, "updated": time.time()}))
        return {"ok": 0, "warn": 2, "crit": 3}[snap["level"]]
    poller = Poller(fetch, max(a.refresh, 60))
    poller.start()
    view = "main"
    try:
        with KeyReader() as keys, Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                snap, meta = poller.view()
                live.update(render_dashboard(snap, {**meta, "repo": meta.get("repo") or repo_label}, view), refresh=True)
                k = keys.get(0.3)
                if k in ("q", "Q"):
                    break
                if k in ("r", "R"):
                    poller.wake.set()
                elif k in ("p", "P"):
                    view = "poison" if view == "main" else "main"
                elif k in ("d", "D"):
                    live.stop()
                    keys.suspend()
                    try:
                        delay_flow(a, sysm)
                    except (OpsError, KeyboardInterrupt) as e:
                        console.print(f"[red]{esc(e)}[/]")
                        time.sleep(2)
                    keys.resume()
                    live.start(refresh=True)
    finally:
        poller.stop_flag = True
        poller.wake.set()
    return 0


# ----------------------------------------------------------------------------- selftest
FIXTURE_GALLERY = """<html><body><nav><a href="https://devpost.com/software">Projects</a></nav>
<div class="gallery"><ul>
<li class="gallery-item"><a href="https://devpost.com/software/alpha"><h5>Alpha</h5></a>
 <span class="winner">Winner</span><a href="https://devpost.com/ann">Ann</a><a href="https://devpost.com/bob">Bob</a> 3 1</li>
<li class="gallery-item"><a href="https://devpost.com/software/beta"><h5>Beta</h5></a>
 <a href="https://devpost.com/cy">Cy</a> 2 0</li>
<li class="gallery-item"><a href="https://devpost.com/software/gamma"><h5>Gamma</h5></a>
 <a href="https://devpost.com/di">Di</a> 1 0</li>
</ul></div><p><b>1</b> \u2013 <b>3</b> of <b>3</b></p></body></html>"""


def _fake_gh_dir(tmp: Path) -> Path:
    d = tmp / "fakebin"
    d.mkdir()
    script = d / "gh"
    script.write_text("""#!/bin/sh
echo "$@" >> "$GH_LOG"
if [ "$1" = "secret" ] && [ "$2" = "set" ]; then cat >> "$GH_STDIN"; exit 0; fi
case "$1 $2" in
  "auth status") exit 0;;
  "variable list") echo '[{"name":"HF_REPO","value":"u/r"},{"name":"DELAY","value":"2.0"}]'; exit 0;;
  "variable get") echo "2.0"; exit 0;;
  "secret list") echo '[{"name":"HF_TOKEN"}]'; exit 0;;
esac
exit 0
""")
    script.chmod(0o755)
    return d


def cmd_selftest(a: argparse.Namespace) -> int:
    import types
    tmp_ctx = tempfile.TemporaryDirectory(prefix="moneyball_ops_selftest_")
    tmp = Path(tmp_ctx.name)
    os.environ["MONEYBALL_OPS_HOME"] = str(tmp / "ops_home")
    results: list[tuple[str, str, str]] = []

    def run(name: str, fn: Callable[[], Optional[str]]) -> None:
        try:
            note = fn()
            results.append((name, "ok" if note != "skip" else "skip", "" if note != "skip" else "dependency not installed"))
        except AssertionError as e:
            where = ""
            tb = e.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            if tb is not None:
                where = f" (line {tb.tb_lineno})"
            results.append((name, "fail", f"assertion failed{where}: {e}"))
        except Exception as e:
            results.append((name, "fail", f"{type(e).__name__}: {str(e)[:200]}"))

    def t_env() -> None:
        vals = {"WINNER_CSS": 'span[class*="winner"]', "A": ".x #y", "B": "it's here", "C": "plain-1.2", "E": ""}
        text = "# keep this comment\nHF_REPO=a/b\nWINNER_CSS=old\nWINNER_CSS=dup\n"
        new = env_update(text, vals)
        parsed = env_parse(new)
        for k, v in vals.items():
            assert parsed[k] == v, f"{k}: {parsed.get(k)!r} != {v!r}"
        assert "# keep this comment" in new and parsed["HF_REPO"] == "a/b"
        assert new.count("WINNER_CSS=") == 1, "duplicate key kept"
        try:
            env_quote("a\nb")
            raise AssertionError("newline accepted")
        except OpsError:
            pass

    def t_fstab() -> None:
        line = fstab_line(MOUNT, "1g", 1000, 1000)
        orig = "proc /proc proc defaults 0 0\nUUID=x / ext4 defaults,noatime 0 1\n"
        new, act = fstab_with_ramdisk(orig, MOUNT, line)
        assert act == "appended" and new.startswith(orig.rstrip("\n")) and new.rstrip().endswith(line)
        again, act2 = fstab_with_ramdisk(new, MOUNT, line)
        assert act2 == "unchanged" and again == new
        small, act3 = fstab_with_ramdisk(new, MOUNT, fstab_line(MOUNT, "512m", 1000, 1000))
        assert act3 == "replaced" and "size=512m" in small and "size=1g" not in small
        dup, act4 = fstab_with_ramdisk(new + line + "\n", MOUNT, line)
        assert act4 == "replaced" and sum(1 for ln in dup.splitlines() if ln.startswith("tmpfs")) == 1
        try:
            fstab_with_ramdisk(orig + f"/dev/sda1 {MOUNT} ext4 defaults 0 2\n", MOUNT, line)
            raise AssertionError("accepted a non-tmpfs entry")
        except OpsError:
            pass

    def t_units() -> None:
        s = render_service("pi")
        assert "User=pi" in s and "--shard 4 --nshards 5 --max-minutes 180\n" in s and "TimeoutStartSec=4h" in s and "@" not in s
        s2 = render_service("bob", shard=0, nshards=1, max_minutes=600, min_registrations=20, app="/srv/mb")
        assert "--shard 0 --nshards 1 --max-minutes 600 --min-registrations 20" in s2 and "TimeoutStartSec=11h" in s2
        assert "WorkingDirectory=/srv/mb" in s2
        assert "OnCalendar=*-*-* 02:15:00" in render_timer("02:15")
        try:
            render_timer("tomorrow")
            raise AssertionError("bad time accepted")
        except OpsError:
            pass

    def t_init_sandbox() -> None:
        root, src = tmp / "root", tmp / "src"
        (root / "etc").mkdir(parents=True)
        (root / "etc" / "fstab").write_text("proc /proc proc defaults 0 0\n")
        src.mkdir()
        (src / "pipeline.py").write_text("# fake\n")
        (src / "requirements.txt").write_text("rich\n")
        ns = argparse.Namespace(dry_run=False, yes=True, sandbox_root=str(root), user="pi", app_dir="/opt/moneyball", size="1g",
                                shard=4, nshards=5, max_minutes=180, min_registrations=0, timer_time="01:40", src=str(src),
                                repo_url=None, update=False, skip_packages=True, skip_venv=True, skip_tests=True,
                                volatile_journal=False, disable_swap=False, no_pair=True)
        with console.capture():
            rc = cmd_init_pi(ns)
            rc2 = cmd_init_pi(ns)  # second run must change nothing
        assert rc == 0 and rc2 == 0, f"exit codes {rc}, {rc2}"
        fstab = (root / "etc" / "fstab").read_text()
        assert fstab.count("/mnt/moneyball-ram") == 1 and "size=1g" in fstab and "noexec" in fstab
        assert list((root / "etc").glob("fstab.moneyball-bak-*")), "no fstab backup written"
        env = root / "etc" / "moneyball.env"
        assert env.exists() and (env.stat().st_mode & 0o777) == 0o600, "env file mode"
        assert (root / "etc/systemd/system/moneyball.service").read_text() == render_service("pi")
        assert (root / "etc/systemd/system/moneyball.timer").read_text() == render_timer("01:40")
        assert (root / "opt/moneyball/pipeline.py").exists()
        helper = (root / "home/pi/.moneyball_shell").read_text()
        assert "WORKDIR=/mnt/moneyball-ram/work" in helper and helper.rstrip().endswith("cd /opt/moneyball")

    def t_probe() -> Optional[str]:
        try:
            load_selectolax()
        except OpsError:
            return "skip"
        _, cards, exp = find_cards(FIXTURE_GALLERY)
        page = Page("gallery", "fixture", FIXTURE_GALLERY, None, cards, exp)
        assert len(cards) == 3 and exp == 3 and cards[0].likes == 3 and cards[0].comments == 1
        m = compute_metrics(page, "", 1)
        assert m["default"] == {0} and m["verdict"] == "PASS", m["verdict"]
        assert compute_metrics(page, "", 2)["verdict"] == "FAIL"
        assert compute_metrics(page, "", None)["verdict"] == "SET EXPECT"
        html2 = FIXTURE_GALLERY.replace('<span class="winner">Winner</span>', '<i class="trophy-x">x</i>')
        _, c2, e2 = find_cards(html2)
        p2 = Page("gallery", "fixture2", html2, None, c2, e2)
        assert compute_metrics(p2, "", 1)["verdict"] == "FAIL"
        assert compute_metrics(p2, ".trophy-x", 1)["verdict"] == "PASS"
        assert compute_metrics(p2, ".li", 1)["verdict"] == "FAIL"
        top = suggest_selectors(p2, 1)
        assert top and top[0]["selector"] == ".trophy-x", top
        every = compute_metrics(p2, "a", 1)
        assert any("every card" in n for n in every["notes"]), every["notes"]
        bad = compute_metrics(p2, "[[[", 1)
        assert bad["verdict"] in ("BAD SELECTOR", "FAIL")
        pp = find_pipeline()
        if pp is not None:
            mod = load_pipeline(str(pp))
            if mod is not None:
                slugs, n = pipeline_gallery(mod, html2, ".trophy-x")
                assert slugs == {"alpha"} and n == 3, (slugs, n)
        return None

    def t_dashboard() -> None:
        now = dt.datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)

        def ts(minutes_ago: float) -> str:
            return iso(now - dt.timedelta(minutes=minutes_ago))

        def row(stage: str, key: str, status: str, note: str, ago: float, worker: str) -> dict:
            return {"stage": stage, "key": key, "status": status, "note": note, "ts": ts(ago), "worker": worker}

        led = [row("discover", f"page:{i}", "ok", "9", 3000 + i, "gha-discover") for i in range(1, 4)]
        led.append(row("discover", "page:4", "empty", "", 2990, "gha-discover"))
        for hid, w, c, e in ((1, 0, 30, 30), (2, 0, 20, 20), (3, 3, 30, 40)):
            led.append(row("gallery", str(hid), "ok", json.dumps({"cards": c, "winners": w, "expected": e}), 400, "gha-0"))
        led.append(row("gallery", "4", "blocked", "429", 380, "gha-1"))
        for i in range(5):
            led.append(row("detail", f"s{i}", "ok", "", 5 + i, "gha-0"))
        led.append(row("detail", "s6", "gone", "404", 50, "gha-0"))
        for i in range(5):
            led.append(row("detail", "s5", "error", "500", 200 + i, "gha-1"))
        for i in range(6):
            led.append(row("detail", f"p{i}", "ok", "", 400 - i, "pi5"))
        led.append(row("detail", "s7", "blocked", "429", 180, "pi5"))
        hacks = [{"hackathon_id": i, "winners_announced": i != 5, "gallery_url": "x", "registrations": 100, "fetched_at": "2026-10-01T00:00:00Z"} for i in range(1, 6)]
        cards = [{"hackathon_id": 1, "slug": f"s{i}", "detail_selected": True} for i in range(10)]
        snap = analyze(led, hacks, cards, now=now, quality_min=2)
        g, d = snap["stages"]["gallery"], snap["stages"]["detail"]
        assert g["total"] == 4 and g["done"] == 3 and abs(g["pct"] - 75.0) < 1e-6, g
        assert d["total"] == 10 and d["done"] == 6, d
        assert snap["stages"]["discover"]["complete"] is True
        assert [p["key"] for p in snap["poison"]] == ["s5"] and snap["poison"][0]["attempts"] == 5
        st = {w["name"]: w["state"] for w in snap["workers"]}
        assert st["pi5"] == "TRIPPED" and st["gha-0"] == "ACTIVE", st
        codes = {al["code"] for al in snap["alerts"]}
        assert {"breaker", "no-winners", "poison", "short-galleries"} <= codes, codes
        assert snap["level"] == "crit"
        assert sum(s["gallery_total"] for s in snap["shards"]) == 4
        assert sum(s["poisoned"] for s in snap["shards"]) == 1
        json.dumps(snap, default=str)
        assert recommended_delay(2.0) == 4.0 and recommended_delay(None) == 4.0 and recommended_delay(5.0) == 10.0 and recommended_delay(9) == 10.0
        with console.capture() as cap:
            console.print(render_dashboard(snap, {"updated": time.time(), "repo": "x"}))
            console.print(render_dashboard(snap, {"updated": time.time()}, "poison"))
        assert "TRIPPED" in cap.get() and "s5" in cap.get()

    def t_local_source() -> Optional[str]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return "skip"
        schema = pa.schema([("stage", pa.string()), ("key", pa.string()), ("status", pa.string()), ("note", pa.string()),
                            ("ts", pa.string()), ("worker", pa.string())])
        root = tmp / "local_ds"
        (root / "ledger" / "detail").mkdir(parents=True)
        now = utcnow()
        rows = [{"stage": "detail", "key": f"k{i}", "status": "ok", "note": "", "ts": iso(now), "worker": "pi5"} for i in range(5)]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "ledger/detail/a.parquet")
        src = LocalSource(str(root))
        got = src.read_rows("ledger/detail")
        assert len(got) == 5 and src.read_rows("ledger/gallery") == []
        snap = analyze(got, None, None, now=now)
        assert snap["stages"]["detail"]["done"] == 5 and snap["stages"]["detail"]["total"] is None
        ns = argparse.Namespace(local=str(root), no_totals=True, once=True, json=True, nshards=5, max_attempts=5, min_registrations=0,
                                dry_run=False, sandbox_root=None, yes=True, no_gha=True, gh_repo=None, repo=None, cache=None, refresh=120)
        with console.capture(), contextlib.redirect_stdout(io.StringIO()) as jout:
            rc = cmd_dashboard(ns)
        assert rc in (0, 2, 3)
        assert json.loads(jout.getvalue())["stages"]["detail"]["done"] == 5, "dashboard --json output"
        return None

    def t_hf() -> Optional[str]:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return "skip"

        class FakeErr(Exception):
            def __init__(self, code: int) -> None:
                super().__init__(f"http {code}")
                self.response = types.SimpleNamespace(status_code=code)

        class FakeApi:
            def __init__(self, repo: str, private: bool = True, role: str = "fineGrained", write_ok: bool = True, found: bool = True) -> None:
                self.repo, self.private, self.role, self.write_ok, self.found, self.commits = repo, private, role, write_ok, found, 0

            def whoami(self) -> dict:
                return {"name": "me", "auth": {"accessToken": {"role": self.role, "fineGrained": {
                    "scoped": [{"entity": {"name": self.repo, "type": "dataset"}, "permissions": ["repo.write"]}], "global": []}}}}

            def repo_info(self, repo: str, repo_type: str = "dataset") -> Any:
                if not self.found:
                    raise FakeErr(404)
                return types.SimpleNamespace(private=self.private)

            def list_repo_files(self, repo: str, repo_type: str = "dataset") -> list[str]:
                return ["a.parquet"]

            def create_commit(self, repo: str, **kw: Any) -> None:
                if not self.write_ok:
                    raise FakeErr(403)
                self.commits += 1

        repo, tok = "me/ds", "hf_" + "a" * 30
        good = validate_hf(tok, repo, api=FakeApi(repo))
        assert not good.failed and not good.warned, [(c.name, c.status) for c in good.checks]
        api = FakeApi(repo)
        validate_hf(tok, repo, api=api)
        assert api.commits == 2, "write test should add then delete"
        assert validate_hf(tok, repo, api=FakeApi(repo, private=False)).failed
        r403 = validate_hf(tok, repo, api=FakeApi(repo, write_ok=False))
        assert r403.failed and any("refused" in c.detail for c in r403.checks)
        classic = validate_hf(tok, repo, api=FakeApi(repo, role="write"))
        assert classic.warned and not classic.failed
        assert validate_hf(tok, repo, api=FakeApi(repo, found=False)).repo_missing
        ro = validate_hf(tok, repo, mode="read", api=FakeApi(repo, write_ok=False))
        assert not ro.failed and not ro.warned and any(c.name == "Read-only as intended" and c.status == "ok" for c in ro.checks)
        assert validate_hf("has space", repo, api=FakeApi(repo)).failed and validate_hf(tok, "nope", api=FakeApi(repo)).failed
        return None

    def t_gh_and_pair() -> Optional[str]:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return "skip"
        fake = _fake_gh_dir(tmp)
        log, sin = tmp / "gh.log", tmp / "gh.stdin"
        old_path = os.environ["PATH"]
        os.environ.update(PATH=f"{fake}{os.pathsep}{old_path}", GH_LOG=str(log), GH_STDIN=str(sin))
        secret = "hf_SELFTEST" + "x" * 24
        try:
            gh = GH("o/n")
            gh.set_secret("HF_TOKEN", secret)
            gh.set_var("HF_REPO", "u/r")
            assert sin.read_text() == secret, "secret did not arrive on stdin"
            assert secret not in log.read_text(), "secret leaked into command arguments"
            assert "--body=u/r" in log.read_text()
            assert gh.list_vars()["HF_REPO"] == "u/r" and gh.list_secret_names() == ["HF_TOKEN"] and gh.get_var("DELAY") == "2.0"
            # whole pair-secrets flow in a sandbox, with the Hugging Face check stubbed
            global validate_hf
            real = validate_hf
            res = HFResult()
            res.add("stub", "ok")
            validate_hf = lambda *x, **k: res  # type: ignore[assignment]
            root = tmp / "pair_root"
            (root / "etc").mkdir(parents=True)
            os.environ["HF_TOKEN"] = secret
            ns = argparse.Namespace(dry_run=False, yes=True, sandbox_root=str(root), repo="u/r", token_from_env=True, strict=False,
                                    check_read_token=False, no_env=False, env_file=ENV_FILE, gh_repo="o/n", no_gh=False,
                                    delay=None, pi_delay=None)
            try:
                with console.capture() as cap:
                    rc = cmd_pair_secrets(ns)
            finally:
                validate_hf = real  # type: ignore[assignment]
                os.environ.pop("HF_TOKEN", None)
            assert rc == 0, f"pair-secrets exit {rc}"
            assert secret not in cap.get(), "token was printed"
            env = root / "etc" / "moneyball.env"
            assert env_parse(env.read_text())["HF_TOKEN"] == secret and (env.stat().st_mode & 0o777) == 0o600
            assert env_parse(env.read_text())["DELAY"] == "2.5"
        finally:
            os.environ["PATH"] = old_path
        return None

    def t_timer_gate() -> None:
        sysm = Sys(root=str(tmp / "gate_root"))
        issues, _ = timer_gate(sysm)
        assert len(issues) >= 4, issues
        (tmp / "gate_root/etc/systemd/system").mkdir(parents=True)
        (tmp / "gate_root/etc/systemd/system/moneyball.timer").write_text("x")
        (tmp / "gate_root/etc").mkdir(exist_ok=True)
        (tmp / "gate_root/etc/moneyball.env").write_text("HF_REPO=u/r\nHF_TOKEN=hf_real\n")
        save_state({"canary_local": {"at": iso(), "ok": True}, "probe": {"verified": True}})
        issues, _ = timer_gate(sysm)
        assert issues == [], issues
        save_state({"canary_local": {"at": iso(utcnow() - dt.timedelta(hours=30)), "ok": True}})
        assert any("24 hours" in i for i in timer_gate(sysm)[0])

    tests = [("env file parse, quote and update", t_env), ("fstab edit is safe and idempotent", t_fstab),
             ("systemd units render", t_units), ("init-pi end to end in a sandbox (twice)", t_init_sandbox),
             ("probe: cards, selectors, suggestions, parity", t_probe), ("dashboard analysis and rendering", t_dashboard),
             ("dashboard reads a local dataset", t_local_source), ("Hugging Face validation (stubbed API)", t_hf),
             ("secrets go through stdin only; pair-secrets never prints the token", t_gh_and_pair),
             ("timer gate", t_timer_gate)]
    for name, fn in tests:
        run(name, fn)
    tmp_ctx.cleanup()
    t = Table(title="moneyball_ops selftest", box=box.SIMPLE_HEAVY)
    t.add_column("check")
    t.add_column("result")
    t.add_column("detail", overflow="fold")
    for name, status, detail in results:
        label, style = STATUS_STYLE[status]
        t.add_row(esc(name), f"[{style}]{label}[/]", esc(detail))
    console.print(t)
    failed = [r for r in results if r[1] == "fail"]
    console.print("[green]all checks passed[/]" if not failed else f"[bold red]{len(failed)} check(s) failed[/]")
    return 1 if failed else 0


# ----------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="show every change, make none")
    common.add_argument("-y", "--yes", action="store_true", help="accept defaults, do not ask to confirm")
    common.add_argument("--sandbox-root", help=argparse.SUPPRESS)
    ap = argparse.ArgumentParser(prog="moneyball_ops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"moneyball_ops {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-pi", parents=[common], help="prepare a Raspberry Pi 5: RAM disk, code, venv, secrets file, units")
    p.add_argument("--user", help="account that owns the code and runs the service (default: you)")
    p.add_argument("--app-dir", default=APP_DIR)
    p.add_argument("--size", default="1g", help="tmpfs size ceiling, e.g. 1g or 512m (default 1g)")
    p.add_argument("--shard", type=int, default=4)
    p.add_argument("--nshards", type=int, default=5)
    p.add_argument("--max-minutes", type=int, default=180)
    p.add_argument("--min-registrations", type=int, default=0, help="added to the service command; must match the Actions workflow")
    p.add_argument("--timer-time", default="01:40", help="local start time of the nightly burst (default 01:40)")
    p.add_argument("--src", help="folder holding pipeline.py to copy (default: next to this script)")
    p.add_argument("--repo-url", help="git URL to clone when there is no local copy")
    p.add_argument("--update", action="store_true", help="git pull --ff-only if the code is already there")
    p.add_argument("--skip-packages", action="store_true")
    p.add_argument("--skip-venv", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--volatile-journal", action="store_true", help="keep the systemd journal in RAM")
    p.add_argument("--disable-swap", action="store_true", help="turn off dphys-swapfile")
    p.add_argument("--no-pair", action="store_true", help="do not offer to run pair-secrets")
    p.set_defaults(fn=cmd_init_pi)

    p = sub.add_parser("pair-secrets", parents=[common], help="check the Hugging Face token and repo, store them, set GitHub secrets")
    p.add_argument("--repo", help="dataset repo USER/NAME")
    p.add_argument("--token-from-env", action="store_true", help="read the write token from $HF_TOKEN instead of prompting")
    p.add_argument("--strict", action="store_true", help="treat warnings as failures")
    p.add_argument("--check-read-token", action="store_true", help="also check the read-only token you use in Colab")
    p.add_argument("--no-env", action="store_true", help="do not write /etc/moneyball.env on this machine")
    p.add_argument("--env-file", default=ENV_FILE)
    p.add_argument("--gh-repo", help="GitHub repo OWNER/NAME")
    p.add_argument("--no-gh", action="store_true")
    p.add_argument("--delay", type=float, default=None, help="Actions DELAY variable (set only if given or missing; default 2.0)")
    p.add_argument("--pi-delay", type=float, default=None, help="Pi DELAY (set only if given or missing; default 2.5)")
    p.set_defaults(fn=cmd_pair_secrets)

    p = sub.add_parser("probe-studio", parents=[common], help="test CSS selectors against a real gallery and save the winner selector")
    p.add_argument("url", nargs="?", help="gallery URL, project URL")
    p.add_argument("--html", help="a saved .html file instead of a URL")
    p.add_argument("--selector", help="start with this selector")
    p.add_argument("--expect", type=int, help="winner badges you counted on page 1")
    p.add_argument("--suggest", action="store_true", help="list big finished hackathons to probe")
    p.add_argument("--pages", type=int, default=20, help="API pages to read for --suggest")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--pipeline", help="path to pipeline.py for the parity check")
    p.add_argument("--report", action="store_true", help="print one report and exit (exit 0 only on PASS)")
    p.add_argument("--gh-repo")
    p.add_argument("--apply-css", metavar="CSS", help="write a selector you verified elsewhere, without testing it here")
    p.set_defaults(fn=cmd_probe_studio)

    p = sub.add_parser("dashboard", parents=[common], help="live ledger view with circuit-breaker alerts")
    p.add_argument("--repo", help="dataset repo USER/NAME (a read token is enough)")
    p.add_argument("--local", help="read a local folder with ledger/ and data/ instead of Hugging Face")
    p.add_argument("--cache", help="where to keep downloaded files (default ~/.cache/moneyball_ops)")
    p.add_argument("--refresh", type=int, default=120, help="seconds between refreshes, 60 or more (default 120)")
    p.add_argument("--once", action="store_true", help="print one snapshot; exit code 0 ok, 2 warnings, 3 breaker or no winners")
    p.add_argument("--json", action="store_true", help="print the analysis as JSON and exit")
    p.add_argument("--nshards", type=int, default=5)
    p.add_argument("--min-registrations", type=int, default=0, help="must match what the workers use")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--no-totals", action="store_true", help="skip the hackathons and cards tables (no percentages, much less to download)")
    p.add_argument("--no-gha", action="store_true", help="do not ask GitHub for run annotations")
    p.add_argument("--gh-repo")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("canary", parents=[common], help="is this machine, or GitHub Actions, being blocked right now")
    p.add_argument("--gha", type=int, metavar="N", default=0, help="dispatch N Actions canary runs and wait for them")
    p.add_argument("--gh-repo")
    p.add_argument("--delay", type=float, default=2.0)
    p.set_defaults(fn=cmd_canary)

    p = sub.add_parser("timer", parents=[common], help="enable, disable or show the nightly Pi timer")
    p.add_argument("action", choices=["enable", "disable", "status"])
    p.add_argument("--force", action="store_true", help="enable even if canary or probe have not passed")
    p.set_defaults(fn=cmd_timer)

    p = sub.add_parser("selftest", parents=[common], help="offline checks of this tool, no network and no root")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except OpsError as e:
        console.print(f"\n[bold red]error:[/] {esc(e)}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
