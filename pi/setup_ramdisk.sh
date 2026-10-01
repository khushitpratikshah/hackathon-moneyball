#!/usr/bin/env bash
# Moneyball Pi 5 setup: RAM disk, venv, secrets file, systemd units. Idempotent; safe to re-run.
#
# Run as the normal Pi user (not root) from inside a checkout of the repo:
#   bash pi/setup_ramdisk.sh
# Options (environment variables):
#   SIZE=1g               tmpfs size ceiling (RAM is only used as files accumulate). Use SIZE=512m for a smaller cap.
#   VOLATILE_JOURNAL=1    keep the systemd journal in RAM so logs never touch the card (logs vanish on reboot)
#   DISABLE_SWAP=1        turn off dphys-swapfile so nothing is swapped to the card
# It does NOT enable the timer. Run the canary and probe checks from RUNBOOK.md section 3 first.
set -euo pipefail

MOUNT=/mnt/moneyball-ram
SIZE="${SIZE:-1g}"
APP=/opt/moneyball
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ME="$(id -un)"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" -ne 0 ] || die "run as your normal user, not root (the script calls sudo where needed)"
command -v sudo >/dev/null || die "sudo is required"
[ -f "$SRC/pipeline.py" ] || die "run this from a checkout of the repo (pipeline.py not found next to pi/)"

echo "== 1/6 packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv git >/dev/null

echo "== 2/6 tmpfs RAM disk at $MOUNT (size ceiling $SIZE)"
sudo mkdir -p "$MOUNT"
FSTAB_LINE="tmpfs $MOUNT tmpfs rw,nosuid,nodev,noexec,noatime,size=$SIZE,mode=0700,uid=$(id -u),gid=$(id -g) 0 0"
if grep -qE "^[^#]*[[:space:]]$MOUNT[[:space:]]+tmpfs" /etc/fstab; then
  sudo sed -i -E "s|^[^#]*[[:space:]]$MOUNT[[:space:]]+tmpfs.*$|$FSTAB_LINE|" /etc/fstab
else
  echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null
fi
sudo systemctl daemon-reload
if findmnt -n "$MOUNT" >/dev/null 2>&1; then
  sudo mount -o "remount,size=$SIZE" "$MOUNT"
else
  sudo mount "$MOUNT"
fi
[ "$(findmnt -n -o FSTYPE "$MOUNT")" = "tmpfs" ] || die "$MOUNT is not a tmpfs"
touch "$MOUNT/.w" && rm "$MOUNT/.w" || die "$MOUNT is not writable by $ME"
findmnt "$MOUNT"

if [ "${VOLATILE_JOURNAL:-0}" = "1" ]; then
  echo "   journal -> RAM"
  sudo mkdir -p /etc/systemd/journald.conf.d
  printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=50M\n' | sudo tee /etc/systemd/journald.conf.d/volatile.conf >/dev/null
  sudo systemctl restart systemd-journald
fi
if [ "${DISABLE_SWAP:-0}" = "1" ]; then
  echo "   swap off"
  sudo systemctl disable --now dphys-swapfile 2>/dev/null || true
fi

echo "== 3/6 code at $APP"
sudo mkdir -p "$APP"
sudo chown "$ME": "$APP"
if [ "$SRC" != "$APP" ]; then
  cp -a "$SRC"/. "$APP"/
fi

echo "== 4/6 venv and dependencies"
cd "$APP"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt pytest
.venv/bin/python -c "import curl_cffi, selectolax, pyarrow, huggingface_hub; print('   imports ok')"

echo "== 5/6 shell helper and secrets file"
cat > "$HOME/.moneyball_shell" <<'SH'
export PYTHONDONTWRITEBYTECODE=1
export WORKDIR=/mnt/moneyball-ram/work
export HF_HOME=/mnt/moneyball-ram/hf_home
export XDG_CACHE_HOME=/mnt/moneyball-ram/cache
export TMPDIR=/mnt/moneyball-ram/tmp
mkdir -p "$WORKDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
cd /opt/moneyball
SH
if [ ! -f /etc/moneyball.env ]; then
  sudo tee /etc/moneyball.env >/dev/null <<'ENVF'
HF_REPO=YOUR_HF_USER/hackathon-moneyball
HF_TOKEN=PASTE_THE_WRITE_TOKEN_HERE
DELAY=2.5
WINNER_CSS=
ENVF
  echo "   created /etc/moneyball.env with placeholders: edit it with: sudo nano /etc/moneyball.env"
fi
sudo chown root:root /etc/moneyball.env
sudo chmod 600 /etc/moneyball.env

echo "== 6/6 systemd units (timer left DISABLED)"
sed "s/__PI_USER__/$ME/" "$APP/pi/moneyball.service" | sudo tee /etc/systemd/system/moneyball.service >/dev/null
sudo cp "$APP/pi/moneyball.timer" /etc/systemd/system/moneyball.timer
sudo systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/moneyball.service && echo "   unit ok"

echo "== offline tests (writes only to the RAM disk)"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$MOUNT/pytest" tests
rm -rf "$MOUNT/pytest"

cat <<MSG

Done. Next:
  1. sudo nano /etc/moneyball.env            (set HF_REPO and HF_TOKEN)
  2. source ~/.moneyball_shell && .venv/bin/python pipeline.py canary
  3. probe a real hackathon (RUNBOOK.md 3.4), then:
  4. sudo systemctl enable --now moneyball.timer
MSG
