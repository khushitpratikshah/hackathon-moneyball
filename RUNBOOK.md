# Runbook: deploy and run the Moneyball Devpost pipeline from scratch

Order matters. Do the steps in sequence and do not move on until the sanity check passes.
Placeholders to replace everywhere: `YOUR_HF_USER`, `YOUR_GH_USER`.
Times: GitHub's cron is UTC. The workflow fires at 02:17 UTC, which is 07:47 IST. The Pi timer fires at 01:40 Pi-local time.

## Final directory layout

```
devpost-moneyball/                 # public GitHub repo, also cloned to /opt/moneyball on the Pi
  pipeline.py                      # all stages: discover gallery detail run canary probe status compact
  requirements.txt
  PLAN.md  RUNBOOK.md
  .gitignore
  .github/workflows/ingest.yml     # canary | discover | run (4 shards)
  pi/moneyball.service  pi/moneyball.timer
  colab/analysis.py                # cell-delimited ("# %%") analysis script
  pi/setup_ramdisk.sh              # one-shot Pi setup (does steps 2.1 to 2.6 below)
  tests/                           # 18 offline tests, no network
  tools/build_export.py  tools/restore_from_export.py   # build / restore PROJECT_EXPORT.md
  ARCHITECTURE_AND_SPECS.md        # portable spec: design, sampling math, schemas, findings
Hugging Face dataset (private) YOUR_HF_USER/hackathon-moneyball
  data/hackathons|cards|projects|projects_raw/*.parquet
  ledger/discover|gallery|detail/*.parquet
```

---

## 1. Initial setup and secrets

### 1.1 Hugging Face dataset repo (private) and tokens

1. Create the repo in the browser: https://huggingface.co/new-dataset
   - Owner: your user. Name: `hackathon-moneyball`. Visibility: **Private**. Create.
2. Create the write token: https://huggingface.co/settings/tokens, "Create new token", type **Fine-grained**.
   - Name: `moneyball-write`
   - Under "Repositories permissions", add `YOUR_HF_USER/hackathon-moneyball` and tick **Write access to contents/settings of selected repos**.
   - Leave every other permission unticked (no account-wide access, no inference, no repo creation).
   - Create, then copy the `hf_...` value now. It is shown once.
3. Create a second fine-grained token the same way, named `moneyball-read`, with **Read access to contents of selected repos** on the same repo. Colab will use this one.

Sanity check, from any machine with Python (do not paste the token into a shell history you sync anywhere; `read -s` keeps it out of the command line):

```bash
python3 -m venv /tmp/hfcheck && /tmp/hfcheck/bin/pip -q install huggingface_hub
export HF_REPO=YOUR_HF_USER/hackathon-moneyball
read -rs HF_TOKEN && export HF_TOKEN
/tmp/hfcheck/bin/python - <<'EOF'
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
print("repo private:", a.repo_info(os.environ["HF_REPO"], repo_type="dataset").private)
a.upload_file(path_or_fileobj=b"ok", path_in_repo="healthcheck.txt", repo_id=os.environ["HF_REPO"], repo_type="dataset")
a.delete_file("healthcheck.txt", repo_id=os.environ["HF_REPO"], repo_type="dataset")
print("write access OK")
EOF
```

Pass: `repo private: True` and `write access OK`. A 403 on upload means the token lacks the write tick for this repo.

### 1.2 GitHub repo, secrets, and safe ignore rules

```bash
unzip moneyball.zip -d devpost-moneyball && cd devpost-moneyball
git init -b main
git add .
git status --short | head -30          # sanity: no .parquet, .html, .env, .zip, probe/ files listed
git grep -nE "hf_[A-Za-z0-9]{20,}" || echo "no token strings in the tree"
git commit -m "Initial pipeline"
gh auth login                           # once, if you have not
gh repo create devpost-moneyball --public --source=. --push
```

The `.gitignore` already excludes secrets (`.env`, `*.env`), scraped data (`*.parquet`, `*.html`, `*.gz`, `*.npy`, `probe/`, `work/`, `ledger/`), and the venv. The repo is public, so nothing scraped may ever be committed.

Secrets and variables:

```bash
gh secret set HF_TOKEN                                  # paste the moneyball-write token when prompted
gh variable set HF_REPO --body "YOUR_HF_USER/hackathon-moneyball"
gh secret list && gh variable list                      # sanity: HF_TOKEN and HF_REPO both appear
```

Lock the public repo down in the browser (Settings of the repo):
- Actions, General, "Fork pull request workflows from outside collaborators": **Require approval for all outside collaborators**.
- Actions, General, "Workflow permissions": **Read repository contents** (the workflow file also pins `permissions: contents: read`).

Notes: GitHub masks the secret in logs, and the pipeline never prints it. Workflow logs on a public repo are public, and they contain only counts and warnings, not scraped content.

---

## 2. Raspberry Pi 5 (shard 4)

Assumes Raspberry Pi OS 64-bit (Bookworm) with SSH access.

Shortcut: clone the repo anywhere on the Pi and run `bash pi/setup_ramdisk.sh` (options `SIZE=512m`, `VOLATILE_JOURNAL=1`, `DISABLE_SWAP=1`). It performs 2.1 to 2.6, creates `/etc/moneyball.env` with placeholders, runs the offline tests inside the RAM disk, and leaves the timer disabled. The manual steps below show what it does.

### 2.1 Packages

```bash
sudo apt update && sudo apt install -y python3-venv git
python3 --version        # 3.11 or newer
```

### 2.2 Isolated tmpfs RAM disk

```bash
sudo mkdir -p /mnt/moneyball-ram
echo "tmpfs /mnt/moneyball-ram tmpfs rw,nosuid,nodev,noexec,noatime,size=1g,mode=0700,uid=$(id -u),gid=$(id -g) 0 0" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload
sudo mount /mnt/moneyball-ram
findmnt /mnt/moneyball-ram                     # FSTYPE must say tmpfs, SIZE 1G
touch /mnt/moneyball-ram/x && rm /mnt/moneyball-ram/x && echo "tmpfs writable"
```

`size=1g` is a ceiling, not a reservation. RAM is used only as files accumulate (a few tens of MB during normal runs). The mount is re-created empty on every boot, which is what you want.

Optional, to keep log writes off the card (logs are then lost on reboot):

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=50M\n' | sudo tee /etc/systemd/journald.conf.d/volatile.conf
sudo systemctl restart systemd-journald
```

Optional, if you want zero swap writes (tmpfs pages can be swapped under memory pressure; a 4 GB or 8 GB Pi will not need it):

```bash
sudo systemctl disable --now dphys-swapfile
```

What "100% in RAM" covers: everything the pipeline writes (work dir, Hugging Face downloads and cache, temp files, Parquet buffers) lands on the tmpfs. The code and venv sit on the card and are only read. The service sets `PYTHONDONTWRITEBYTECODE=1` so Python does not write `.pyc` files there. Installing packages and cloning the repo in the next steps write to the card once.

### 2.3 Code and venv

```bash
sudo git clone https://github.com/YOUR_GH_USER/devpost-moneyball /opt/moneyball
sudo chown -R "$USER": /opt/moneyball
cd /opt/moneyball
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -c "import curl_cffi, selectolax, pyarrow, huggingface_hub; print('imports ok')"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/mnt/moneyball-ram/pytest tests
rm -rf /mnt/moneyball-ram/pytest
```

Pass: `imports ok` and `18 passed`. If `curl_cffi` fails to install on the Pi, stop and fix that first (it is the HTTP layer); `pip install --upgrade pip` and retry usually does it.

### 2.4 Shell helper for manual commands on the Pi

Every command you run by hand should also keep its writes in RAM. Create this once:

```bash
cat > ~/.moneyball_shell <<'EOF'
export PYTHONDONTWRITEBYTECODE=1
export WORKDIR=/mnt/moneyball-ram/work
export HF_HOME=/mnt/moneyball-ram/hf_home
export XDG_CACHE_HOME=/mnt/moneyball-ram/cache
export TMPDIR=/mnt/moneyball-ram/tmp
mkdir -p "$WORKDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
cd /opt/moneyball
EOF
source ~/.moneyball_shell        # run this at the start of every SSH session
```

### 2.5 Secrets file for the service

```bash
sudo tee /etc/moneyball.env >/dev/null <<'EOF'
HF_REPO=YOUR_HF_USER/hackathon-moneyball
HF_TOKEN=PASTE_THE_WRITE_TOKEN_HERE
DELAY=2.5
WINNER_CSS=
EOF
sudo chown root:root /etc/moneyball.env && sudo chmod 600 /etc/moneyball.env
sudo ls -l /etc/moneyball.env      # -rw------- root root
```

Edit the placeholders with `sudo nano /etc/moneyball.env`. `DELAY` is the base seconds between requests for this worker. `WINNER_CSS` stays empty unless step 3.4 tells you to set it.

### 2.6 systemd service and timer

```bash
cd /opt/moneyball
sed "s/__PI_USER__/$USER/" pi/moneyball.service | sudo tee /etc/systemd/system/moneyball.service >/dev/null
sudo cp pi/moneyball.timer /etc/systemd/system/moneyball.timer
sudo systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/moneyball.service && echo "unit ok"
timedatectl | grep "Time zone"     # the timer uses this zone (01:40 local)
```

**Do not enable the timer yet.** Finish section 3 first.

---

## 3. Pre-flight: canary and probe

### 3.1 Offline tests (laptop or Pi)

```bash
pip install -r requirements.txt pytest && pytest -q tests      # 18 passed
```

### 3.2 Canary from the Pi

```bash
source ~/.moneyball_shell
.venv/bin/python pipeline.py canary; echo "exit=$?"
```

Pass: three lines `ok 200` (`api`, `gallery`, `project`) and `exit=0`.
- `blocked 403/429/503`, or exit 3: this IP is being challenged. Do not start the run from it. Wait a few hours, retry; if it persists, use Actions only (section 4.6, case C).
- `error exc:...`: network problem on the Pi, not a block.

The canary counts a 403, 429, 503, a `cf-mitigated: challenge` header, or a "Just a moment..." page as blocked. It tests three representative URLs, so a pass means "not blocked right now", not "will never be blocked".

### 3.3 Canary from GitHub Actions (different runner IPs each time)

Run it three times, one after another, so you sample several runner IPs (the concurrency group would cancel overlapping dispatches):

```bash
for i in 1 2 3; do
  gh workflow run ingest.yml -f stage=canary
  sleep 10
  gh run watch "$(gh run list --workflow=ingest.yml --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status \
    && echo "canary $i PASSED" || echo "canary $i FAILED"
done
```

Pass: 3 of 3 pass. If Actions fails consistently, GitHub's IP ranges are blocked: go to 4.6 case C. If it fails once in three, expect occasional circuit-breaker trips and keep going.

### 3.4 Probe a real completed hackathon (the step that must be right before the long run)

The winner badge's HTML was not verifiable from my side, so you confirm it once here.

**a) Get candidate URLs** (about 20 polite requests; picks large, finished hackathons that announced winners):

```bash
source ~/.moneyball_shell
.venv/bin/python - <<'EOF'
import json, pipeline as p
cl = p.Client(1.0); out = []
for pg in range(1, 21):
    k, c, t = cl.get(p.API, params={"status[]": "ended", "order_by": "recently-added", "page": pg})
    if k != "ok": break
    out += [h for h in json.loads(t)["hackathons"]
            if h["winners_announced"] and h["registrations_count"] >= 300 and not h["invite_only"]]
for h in sorted(out, key=lambda h: -h["registrations_count"])[:8]:
    print(h["registrations_count"], h["submission_gallery_url"])
EOF
```

**b) Pick one and look at it in your browser.** Open its gallery URL, confirm that some cards show a "Winner" badge, and count the winner badges visible on page 1. Open one winning project and one non-winning project; note their `/software/<slug>` URLs.

**c) Probe the gallery** (page 1 only):

```bash
.venv/bin/python pipeline.py probe "https://SOME-HACKATHON.devpost.com/project-gallery"
```

Pass: the line `expected=N cards=M winners=K` shows `K` equal to the number of winner badges you counted on page 1, with `0 < K < M`. The first five parsed cards are printed under it.
- `K = 0` but you saw badges: find the real class (next block), then set `WINNER_CSS`.
- `K = M` (everything flagged): a selector is overmatching. Do not set `WINNER_CSS` to a class that appears on every card.

Find the badge class in the saved HTML:

```bash
f=$(ls -t "$WORKDIR"/probe/*.html | head -1)
grep -o 'class="[^"]*\(winner\|prize\|badge\|award\)[^"]*"' "$f" | sort | uniq -c | sort -rn | head
grep -o -i '.\{80\}winner.\{80\}' "$f" | head -5
```

Then test a candidate selector without changing any file, and repeat until `winners=` matches what you counted:

```bash
WINNER_CSS=".the-badge-class" .venv/bin/python pipeline.py probe "https://SOME-HACKATHON.devpost.com/project-gallery"
```

Make it permanent once it matches:

```bash
sudo sed -i 's/^WINNER_CSS=.*/WINNER_CSS=.the-badge-class/' /etc/moneyball.env
gh variable set WINNER_CSS --body ".the-badge-class"
```

**d) Probe a winning project, then a non-winning one:**

```bash
.venv/bin/python pipeline.py probe "https://devpost.com/software/WINNING-SLUG"
.venv/bin/python pipeline.py probe "https://devpost.com/software/NON-WINNING-SLUG"
```

Check against the page in your browser:

| Field | Pass condition |
|---|---|
| `built_with` | Same tags as the "Built With" list on the page (lowercase). Cross-check: `grep -o 'built-with/[^"]*' "$f" \| sort -u` |
| `is_winner_page` / `winner_texts` | true with text for the winner, false and `[]` for the non-winner |
| `story_root_found` | `true`. If `false`, run `grep -o 'id="app-details[^"]*"' "$f"` and tell me the ids; word counts would include page chrome |
| `team_size` | Matches the number of people under "Created by" |
| `video_platform`, `video_id` | Set when the page has a video; null when it has none |
| `likes`, `comments`, `n_updates`, `first_update_text` | Match what the page shows |
| `tagline`, `try_links` | Match the subtitle and the "Try it out" links |

Clean up: `rm -rf "$WORKDIR"/probe`

Do not continue until 3.2, 3.3 and 3.4 pass.

---

## 4. Running and monitoring

### 4.1 Make sure the repo on GitHub is current

Scheduled workflows run from the default branch only.

```bash
git add -A && git commit -m "Tune settings" && git push      # only if you changed files
```

### 4.2 Discovery (once)

```bash
gh workflow run ingest.yml -f stage=discover
sleep 10
gh run watch "$(gh run list --workflow=ingest.yml --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
```

About 1,540 requests at 2 s each, so roughly an hour. Then check the ledger from the Pi (or any machine with the venv):

```bash
source ~/.moneyball_shell
export HF_REPO=YOUR_HF_USER/hackathon-moneyball; read -rs HF_TOKEN; export HF_TOKEN
.venv/bin/python pipeline.py status
```

Expected shape (numbers will differ):

```
hackathons discovered: 13xxx | with winners announced: N | detail sample: 0
discover  done 1537 | empty=1, ok=1536
gallery   no ledger yet
detail    no ledger yet
```

If `discover` shows `blocked` or `error` counts, rerun the discover dispatch; finished pages are skipped. If `winners announced` is 0, stop and re-check the API fields.

Size the job now. Detail-stage requests ≈ `N x (winners + up to 40)` plus about 1 to 3 gallery requests per hackathon. If that is more than you want, skip tiny events:

```bash
sed -i 's/--nonwinners-per-hack 40/--nonwinners-per-hack 40 --min-registrations 20/' .github/workflows/ingest.yml
sudo sed -i 's/--shard 4 --nshards 5 --max-minutes 180/--shard 4 --nshards 5 --max-minutes 180 --min-registrations 20/' /etc/systemd/system/moneyball.service
sudo systemctl daemon-reload
git commit -am "Skip hackathons under 20 registrations" && git push
```

Use the same flags on every worker, or shards will disagree about the work list.

### 4.3 Start the 4 Actions shards and the Pi shard

The workers do not need to start in sync. Each one only touches keys where `sha1(key) % 5` equals its shard, so there is no overlap whatever the start order.

```bash
gh workflow run ingest.yml -f stage=run                   # shards 0..3
sudo systemctl start moneyball.service                    # shard 4 (runs up to 3 hours)
journalctl -u moneyball -f                                # Ctrl-C stops following, not the job
```

First minute on the Pi should show something like `[gallery] shard 4/5: 1500 hackathons to do`. Then enable the nightly schedule:

```bash
sudo systemctl enable --now moneyball.timer
systemctl list-timers moneyball.timer          # NEXT shows tonight's run
```

Actions runs by itself nightly from the cron in the workflow. Each run does gallery for its shard first, then detail, inside its time budget, and exits cleanly when time is up.

### 4.4 Monitoring

```bash
.venv/bin/python pipeline.py status                       # progress per stage, ok-today per worker
gh run list --workflow=ingest.yml --limit 6
gh run view RUN_ID --log | grep -E "gallery\]|detail\]|warning"     # one run's key lines
systemctl status moneyball.service                        # Pi; also: journalctl -u moneyball --since today
```

Healthy signs: `done` percentages rise every day, `ok today` is in the thousands per Actions shard (about 1,400 per hour at the default pace) and roughly 4,000 for the Pi's 3 hours, and `blocked`/`error` stay small relative to `ok`. Status downloads the ledger files each time, so run it a few times a day, not in a loop.

### 4.5 How the remote ledger handles state and resumption

- Every worker buffers rows in memory and ships data plus ledger entries together as one Hugging Face commit, every 300 rows or 10 minutes, and once more on exit.
- A ledger row is `(stage, key, status, note, ts, worker)`. Keys are `page:N` (discover), hackathon id (gallery), project slug (detail).
- On start, a worker downloads `ledger/<stage>/` and treats a key as done if its status is `ok`, `empty` or `gone`. Everything else is retried.
- A key that failed 5 times is skipped (poison pill). Retry those with `--max-attempts 10` once the cause is fixed.
- The gallery stage writes a hackathon's cards atomically. If a block hits halfway through a gallery, nothing from it is saved and it retries later.
- `systemctl stop moneyball`, a cancelled Actions run, or Ctrl-C triggers a flush and exit code 130. A power cut or kill -9 loses at most the last unflushed batch (under 10 minutes), and the next run redoes only that.
- Resharding is safe at any time, because done-ness is tracked by key, not by shard.
- Weekly, while no worker is running, merge the small ledger files: `.venv/bin/python pipeline.py compact`.

### 4.6 If the circuit breaker trips (429 and friends)

What it looks like: the client doubles its delay on each 403, 429 or 503 (up to 16x), and after 8 blocks in a row it flushes, prints `::warning::blocked, cooling down until next run`, and exits 0. In Actions that shows as a yellow annotation on a green job; on the Pi it is in the journal. Nothing is lost.

1. **Do not rerun straight away.** Run `status` and confirm the ledger's last write is recent, so the flush landed.
2. **Wait a few hours**, then run `canary` from that environment (3.2 for the Pi, 3.3 for Actions).
3. **Canary passes: slow down.**
   - Actions: `gh variable set DELAY --body 4.0`
   - Pi: `sudo sed -i 's/^DELAY=.*/DELAY=5.0/' /etc/moneyball.env` (read fresh on every start, no daemon reload needed)
4. **Case A, one IP keeps tripping early in every run while the others are fine.** Remove that worker and give its shard to Actions.
   - Pi: `sudo systemctl disable --now moneyball.timer`, then `sed -i 's/shard: \[0, 1, 2, 3\]/shard: [0, 1, 2, 3, 4]/' .github/workflows/ingest.yml && git commit -am "Shard 4 to Actions" && git push`
5. **Case B, repeated 429 even at 5 s or more.** Stop everything for 24 to 48 hours (`gh workflow disable ingest.yml`, stop the Pi timer). Then resume at a delay of 6 or more. If it keeps happening, consider asking Devpost for permission or a data export instead of pushing harder.
6. **Case C, Actions IPs blocked, Pi fine.** Run the Pi as the only worker and stretch its window:
   - `sudo sed -i 's/--shard 4 --nshards 5 --max-minutes 180/--shard 0 --nshards 1 --max-minutes 600/; s/TimeoutStartSec=4h/TimeoutStartSec=11h/' /etc/systemd/system/moneyball.service && sudo systemctl daemon-reload`
   - `gh workflow disable ingest.yml`
   - Expect weeks instead of days (about 0.4 requests per second is roughly 4,000 requests per 3 hours).
7. After the cause is fixed, retry poisoned keys by adding `--max-attempts 10` to the run commands.

Never add workers or lower the delay to "push through" a block.

---

## 5. Colab analysis walkthrough

### 5.1 Set up the notebook

1. https://colab.research.google.com, New notebook. Runtime, Change runtime type, **T4 GPU**.
2. Left sidebar, key icon (Secrets). Add three, each with "Notebook access" switched on:
   - `HF_TOKEN`: the `moneyball-read` token
   - `HF_REPO`: `YOUR_HF_USER/hackathon-moneyball`
   - `YT_API_KEY` (optional, for video length): console.cloud.google.com, new project, APIs and Services, Library, "YouTube Data API v3", Enable, then Credentials, Create credentials, API key, restrict it to that API. The default quota is 10,000 units per day and, as far as I know, no billing account is needed. Skip this secret and the script just leaves `video_min` empty.

### 5.2 Run it

Option A, one cell, runs everything top to bottom:

```python
!git clone https://github.com/YOUR_GH_USER/devpost-moneyball /content/repo
%run /content/repo/colab/analysis.py
```

Option B, cell by cell (better for the first run). On your laptop:

```bash
pip install jupytext
jupytext --to ipynb colab/analysis.py -o analysis.ipynb
```

Upload `analysis.ipynb` through Colab's File, Upload notebook. Each `# %%` block becomes a cell. Run them in order.

To skip the text-embedding cell on a first pass or if memory is tight, run this before the script: `import os; os.environ["RUN_TEXT"] = "0"`.

### 5.3 What each cell does and what to check

| Cell | Does | Check |
|---|---|---|
| 0 | Loads Colab secrets into env vars | No error |
| 1 | Installs duckdb, lightgbm, shap, bertopic, sentence-transformers, statsmodels | Finishes without a resolver error |
| 2 | `snapshot_download` of `data/hackathons`, `data/cards`, `data/projects`, `enrich/` into `/content/hf/data`. It does not pull raw HTML | Files appear under `/content/hf/data/*/` |
| 3 | DuckDB views over the Parquet files with `union_by_name=true`, deduped by key (newest `parser_version` wins) | Prints counts for hackathons, cards, winner cards, detailed projects |
| 4 | Parses `period_text` into start and end dates | `hk[["period_text","start","end"]].head()` looks right |
| 5 | Video lengths from the YouTube and Vimeo APIs (only if `YT_API_KEY` is set) | `vm.minutes.describe()` shows sane minutes |
| 6 | Builds the gold table and saves `/content/out/gold.parquet` | `rows; N winners` line, N in the hundreds or more |
| 7 | Conditional logit with hackathon fixed effects, saves `conditional_logit.csv` | Table of odds ratios per standard deviation |
| 8 | LightGBM with 5-fold GroupKFold by hackathon, then SHAP, saves `shap_summary.png` | Pooled and within-hackathon AUC printed, top-15 SHAP features |
| 9 | MiniLM embeddings, BERTopic, topic win-rate lift, saves `topic_lift.csv` | Topics table (needs the GPU runtime for speed) |
| 10 | Comments only: how to re-parse raw HTML with a newer parser | Nothing to run |

Paste this sanity cell after cell 3 and read it before trusting any model output:

```python
print(con.sql("""SELECT count(*) cards, count(DISTINCT hackathon_id) hackathons,
                 avg(is_winner::INT) win_rate, avg(detail_selected::INT) sampled FROM cards""").df())
print(con.sql("""SELECT count(*) projects, avg(story_root_found::INT) root_found,
                 avg((video_url IS NOT NULL)::INT) has_video, avg(len(built_with)) avg_tags,
                 avg(team_size) avg_team FROM projects""").df())
print(con.sql("""SELECT count(*) AS selected_but_not_fetched FROM cards c
                 LEFT JOIN projects p USING (slug) WHERE c.detail_selected AND p.slug IS NULL""").df())
print(con.sql("""SELECT count(*) AS usable_hackathons FROM (
                   SELECT hackathon_id FROM cards WHERE detail_selected
                   GROUP BY 1 HAVING sum(is_winner::INT) > 0 AND sum((NOT is_winner)::INT) > 0)""").df())
```

What to look for (my expectations, not measured values):
- `root_found` close to 1.0. If it is well below, the story container selector is wrong for many pages and `story_words` and the heading features are polluted.
- `win_rate` is not a real-world base rate. It is inflated by design, because every winner is kept and non-winners are capped at 40 per hackathon. Use it only for sanity.
- `avg_tags` of a few and `avg_team` of roughly 2 to 4.
- `selected_but_not_fetched` goes to 0 as the crawl completes. Mid-crawl it is fine: detail pages are fetched in hash-random order, so a partial set is an unbiased sample.
- `usable_hackathons` is what the conditional logit can learn from. Under a few hundred means the results will be noisy.

### 5.4 Reading the models

Conditional logit (`conditional_logit.csv`): `odds_ratio_per_sd` above 1 means that, inside the same hackathon, a project one standard deviation higher on that feature is more likely to win. If `lo` to `hi` spans 1.0, you cannot distinguish it from no effect. `story_words` is log-transformed before scaling. A feature that is constant in your sample shows an infinite interval; drop it.

LightGBM: trust the **within-hackathon AUC** more than the pooled one. Around 0.5 means the scraped features carry almost no signal for picking winners inside an event. Clearly above 0.6 is a real signal. Features are percentiles within each hackathon plus absolute values, and `likes_post`, `comments_post` and `updates_post` are excluded by an assertion because they accumulate after judging.

SHAP plots: view the summary in Colab and look at one shape:

```python
from IPython.display import Image; display(Image("/content/out/shap_summary.png"))
shap.dependence_plot("story_words", sv, sample)      # swap in any column name from the summary
```

Features that rank high in SHAP and also have an odds ratio away from 1 in the logit are the ones to believe. A SHAP-only feature is a non-linearity or interaction worth checking; a logit-only feature with low SHAP is real but small.

### 5.5 Save the results

```python
!cd /content/out && zip -r results.zip conditional_logit.csv shap_summary.png topic_lift.csv gold.parquet
from google.colab import files; files.download("/content/out/results.zip")
```

`gold.parquet` contains writeup text. Keep it private and out of the GitHub repo. Share only aggregate tables and plots.

If Colab runs out of memory: set `RUN_TEXT=0`, or restart the runtime and rerun without cell 9. The Parquet files are read by DuckDB, so the raw tables themselves are not the problem.

---

## Operations cheat sheet

```bash
# Pi (source ~/.moneyball_shell first)
.venv/bin/python pipeline.py canary
.venv/bin/python pipeline.py status
sudo systemctl start moneyball.service          # run a burst now
journalctl -u moneyball -f
sudo systemctl disable --now moneyball.timer    # pause the Pi

# GitHub
gh workflow run ingest.yml -f stage=canary|discover|run
gh run list --workflow=ingest.yml --limit 6
gh variable set DELAY --body 4.0                # slow all Actions shards
gh workflow disable ingest.yml                  # pause Actions (enable to resume)
```

Done when `status` shows `gallery` and `detail` at 100% (or only `gone` and poisoned keys left) and a run prints `0 projects to do` for every shard. Then disable the timer and the workflow, run `compact` once, and do the full Colab pass.
