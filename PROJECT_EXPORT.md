# Moneyballing Hackathons: complete project export

Every file below is the exact content of the real file. Restore all of them with
`python tools/restore_from_export.py PROJECT_EXPORT.md <target_dir>` (it checks every sha256).

## Resume prompt for a new chat

Paste this, then attach or paste the files from this export:

> I am continuing a project called "Moneyballing Hackathons". Read ARCHITECTURE_AND_SPECS.md first, then RUNBOOK.md.
> The code is already written and tested offline (18 tests). Nothing has run against live Devpost or Hugging Face yet.
> Do not redesign the architecture. Help me execute the runbook, starting with the preflight checks (canary, then
> probing a real winning hackathon to confirm the winner-badge selector), and help me interpret ledger status
> output, circuit-breaker trips, and the Colab results.

## Manifest

| path | lines | sha256 |
|---|---|---|
| `ARCHITECTURE_AND_SPECS.md` | 315 | `1a25f6b59c137e1b` |
| `pipeline.py` | 769 | `6795af7b417bb330` |
| `.github/workflows/ingest.yml` | 84 | `cb1b98ad772d6c21` |
| `pi/setup_ramdisk.sh` | 110 | `b751232e5c829454` |
| `pi/moneyball.service` | 26 | `235764cfd02aa103` |
| `pi/moneyball.timer` | 15 | `29b854cb9e0127ce` |
| `colab/analysis.py` | 219 | `20e4dd7512a3d09f` |
| `tests/test_pipeline.py` | 272 | `f9339c710eda52fb` |
| `tests/make_synthetic.py` | 32 | `e819f03a0759c4dc` |
| `requirements.txt` | 4 | `531e1cbc6674f317` |
| `.gitignore` | 28 | `ab4de792979c08d5` |
| `PLAN.md` | 162 | `b402a7add2edf6f2` |
| `RUNBOOK.md` | 535 | `4b207e74c762404f` |
| `tools/build_export.py` | 65 | `ff8760719f636cee` |
| `tools/restore_from_export.py` | 46 | `ee2d04ea9eb7e14f` |

## File 1: ARCHITECTURE_AND_SPECS.md

### `ARCHITECTURE_AND_SPECS.md`

<!-- FILE: ARCHITECTURE_AND_SPECS.md sha256=1a25f6b59c137e1b7ac8470113ec6c2a28b197f4e93dc563cf54cb0d62d9aaeb -->
````markdown
# Moneyballing Hackathons: architecture and specs (portable export)

This file is self-contained. A new chat or machine needs only this file plus the code files from the same export. Everything marked VERIFIED was checked against live Devpost pages on 1 Oct 2026. Everything marked UNVERIFIED was not, and is listed again in section 10.

## 1. Context and objective

Goal: a one-off, $0, no-credit-card research pipeline that collects as many completed Devpost hackathons and their project galleries as possible, labels which projects won, and builds a table for statistical modeling. The research question is what correlates with winning: tech stack, theme, video presence and length, team size, writeup structure. Analysis runs in Google Colab with a conditional logit, LightGBM and SHAP.

Hard constraints:
- Budget is exactly $0 and nothing may need a credit card.
- Compute: one Raspberry Pi 5 on a home network (runs 24/7, residential IP), GitHub Actions on a public repo (short bursts), Google Colab free tier.
- No heavy or long-term writes on the Pi's SD card or NVMe.
- Storage must be free and card-free. Chosen: a private Hugging Face dataset repo with Parquet files.
- Lifecycle: a one-off batch over days to weeks, in bursts of minutes to hours per day.

## 2. Sampling design and why odds ratios survive it

Fetching every project page is the expensive part (one request each). The gallery page already gives the label (winner or not) for every project, so the design is:

- Phase 2 (gallery) labels every project in a hackathon.
- Phase 3 (detail) fetches the project page only for a sample: every winner, plus up to 40 non-winners per hackathon.

The non-winners are picked deterministically: sort that hackathon's non-winners by `sha1("<seed>:<slug>")` and take the first `k = nonwinners_per_hack` (default 40, seed default 42). If a hackathon has fewer non-winners than k, all are kept.

Why this preserves odds ratios. Let project i in hackathon h have features x and outcome y. Assume the usual logistic model P(y=1 | x, h) = sigmoid(a_h + b'x), where a_h is a hackathon-specific intercept. Selection S depends only on the outcome and the hackathon, never on x:

- P(S=1 | y=1, x, h) = 1  (every winner is kept)
- P(S=1 | y=0, x, h) = pi_h = min(1, 40 / n0_h), where n0_h is the number of non-winners in h

The selection key is a hash of the slug, which is independent of every feature. Then, among selected projects, the odds of winning are

  P(y=1 | x, h, S=1) / P(y=0 | x, h, S=1) = [P(y=1|x,h) * 1] / [P(y=0|x,h) * pi_h] = exp(a_h + b'x) / pi_h

so the model among selected projects is logistic with the same slope b and a shifted intercept a_h - ln(pi_h). This is the classic case-control (outcome-dependent) sampling result: slopes are unaffected, only intercepts move. Two consequences:

1. A conditional logit with one stratum per hackathon conditions the intercept away completely, so b is estimated consistently with no correction. That is why `colab/analysis.py` uses statsmodels `ConditionalLogit(groups=hackathon_id)` as the primary inference tool.
2. Anything that uses absolute probabilities is miscalibrated: pooled logistic intercepts, and LightGBM `predict_proba` values. They describe the sample, not real win rates. For calibrated probabilities, weight winners 1 and non-winners 1/pi_h. The shipped script does not apply weights because it reports ranking quality, not probabilities.

Additional notes:
- Top-k by hash is a simple random sample of size k without replacement within a hackathon. Inclusion probability is k/n0 for every non-winner there, independent of x.
- Within-hackathon AUC is unchanged in expectation by this sampling, because it compares winners to uniformly sampled non-winners in the same event. Pooled AUC is not comparable across different sampling fractions.
- The guarantee breaks if missingness is related to features: detail pages that return 404 or stay blocked drop out of the sample. Check `status` counts for `gone` and for poisoned keys before interpreting results.
- Fetch order is hash-random, so a half-finished crawl is still an unbiased sample, not a pile of winners.

## 3. Infrastructure and sharding layout

Work is split into 5 shards by a consistent hash, with no coordinator and no locks:

  shard_of(key, n) = int(sha1(str(key)).hexdigest()[:8], 16) % n

All workers must use the same `--nshards 5`.

| Shard | Runs on | Schedule and budget | Pace |
|---|---|---|---|
| 0, 1, 2, 3 | GitHub Actions matrix, one job per shard | cron `17 2 * * *` (02:17 UTC, 07:47 IST); job timeout 355 min; work budget `--max-minutes 330` | `DELAY` 2.0 s base, about 0.5 requests per second |
| 4 | Raspberry Pi 5, systemd timer | 01:40 Pi-local time plus up to 20 min random delay; `--max-minutes 180`; `TimeoutStartSec=4h` | `DELAY` 2.5 s base, about 0.4 requests per second |
| discover (one-off) | one GitHub Actions job, not a matrix | manual dispatch, `--max-minutes 100`, timeout 120 min | `DELAY` 2.0 s |

What a shard owns:
- Gallery stage: hackathons where `shard_of(hackathon_id, 5) == shard`.
- Detail stage: project slugs where `shard_of(slug, 5) == shard`. The two hashes are independent, so the detail work for one hackathon is spread across all shards. That is intended: any worker's progress is an unbiased slice.
- Resharding is safe at any time (change `--nshards` or move a shard to another machine), because completion is tracked per key in the ledger, not per shard.
- Actions concurrency group `ingest` with `cancel-in-progress: false` serializes dispatches, so do not fire overlapping dispatches.
- Pi work directory and every cache live on a tmpfs at `/mnt/moneyball-ram`; the service wipes it after each run.

Throughput planning (estimates, not measurements): 0.4 to 0.5 requests per second is about 1,400 to 1,800 per hour per worker. Four Actions shards at 5.5 h plus the Pi at 3 h is about 40,000 requests per day. Phase 1 is about 1,540 requests. Phases 2 and 3 total about `hackathons_with_winners x (gallery pages + winners + up to 40)`; compute it from the `hackathons` table after discovery.

## 4. Storage and state spec

### 4.1 Hugging Face dataset repo (private)

Fine-grained token with write access to this one repo only (write token for scrapers, read token for Colab).

```
data/hackathons/<worker>-<UTC ts>-<6hex>.parquet     one row per hackathon from the API
data/cards/...                                       one row per (hackathon, project) with winner flag and sample flag
data/projects/...                                    parsed project pages, tagged with parser_version
data/projects_raw/...                                gzipped raw HTML of every fetched project page
ledger/discover|gallery|detail/...                   tiny state files, one row per attempted key
```

All files are append-only Parquet, zstd compressed, never modified. Write pattern: each worker buffers rows and ledger entries in memory and ships all of them in a single `create_commit` per flush. A flush happens when 300 rows are buffered across all tables (a detailed project adds a `projects` row and a `projects_raw` row, so about 150 projects), or when 600 seconds have passed with ledger entries pending, and always once at exit. Hugging Face throttles commits per repo per hour, so do not flush per row. If a commit fails it retries 4 times with 10, 20, 40, 80 s sleeps; if it still fails the rows stay in memory and go out with the next flush.

### 4.2 Raw HTML cache and schema evolution

Every fetched project page is stored gzipped in `projects_raw`. Parsed columns in `projects` carry `parser_version`. To add a feature, bump `PARSER_VERSION` in `pipeline.py`, re-parse `projects_raw` in Colab (no network needed), and write a new `projects` file. In DuckDB, `read_parquet(..., union_by_name=true)` fills old files with NULL for new columns, and the view keeps the highest `parser_version` per slug. A parsing mistake never requires re-crawling.

### 4.3 Ledger synchronization logic

Ledger row: `(stage, key, status, note, ts, worker)`. Keys: `page:N` for discover, `hackathon_id` for gallery, project `slug` for detail.

- Statuses: `ok`, `empty`, `gone` (404 or 410) are final. `blocked` and `error` are retryable.
- On start a worker downloads `ledger/<stage>/*` with `snapshot_download`, builds `{key: status}` for final statuses and a failure counter for everything else.
- A key with 5 failed attempts is skipped (poison pill). `--max-attempts N` raises the limit to retry those after the cause is fixed.
- The gallery stage writes a hackathon's cards atomically: a block or error halfway through discards that hackathon's partial cards and logs the failure, so it retries cleanly.
- The ledger note for a finished gallery is JSON: `{"cards": n, "winners": w, "expected": N}` where `expected` is the "of N" total printed on the gallery page, used as a completeness check.
- `pipeline.py status` reports final status per key per stage, the share done, and per-worker ok counts for the current UTC day. `pipeline.py compact` merges small ledger files into one and must be run only while no worker is active.
- SIGTERM and Ctrl-C trigger a final flush and exit code 130. A power cut loses at most the unflushed batch, and the next run redoes only that.

### 4.4 Parquet schemas (generated from `pipeline.py`)

**`hackathons`**

| column | type |
|---|---|
| `hackathon_id` | int64 |
| `title` | string |
| `url` | string |
| `gallery_url` | string |
| `open_state` | string |
| `period_text` | string |
| `themes` | list<item: string> |
| `prize_text` | string |
| `prize_value` | double |
| `prize_currency` | string |
| `prizes_cash` | int32 |
| `prizes_other` | int32 |
| `registrations` | int32 |
| `organization` | string |
| `winners_announced` | bool |
| `invite_only` | bool |
| `managed_by_devpost` | bool |
| `location` | string |
| `is_online` | bool |
| `fetched_at` | string |

**`cards`**

| column | type |
|---|---|
| `hackathon_id` | int64 |
| `slug` | string |
| `is_winner` | bool |
| `winner_text` | string |
| `likes` | int32 |
| `comments` | int32 |
| `profile_links` | int32 |
| `card_text` | string |
| `page` | int32 |
| `detail_selected` | bool |
| `n_cards_in_hackathon` | int32 |
| `n_winners_in_hackathon` | int32 |
| `fetched_at` | string |

**`projects`**

| column | type |
|---|---|
| `slug` | string |
| `parser_version` | int32 |
| `fetched_at` | string |
| `title` | string |
| `tagline` | string |
| `og_image` | string |
| `built_with` | list<item: string> |
| `story_text` | string |
| `story_chars` | int32 |
| `story_words` | int32 |
| `story_root_found` | bool |
| `headings` | list<item: string> |
| `n_h2` | int32 |
| `n_h3` | int32 |
| `n_img` | int32 |
| `n_li` | int32 |
| `n_code` | int32 |
| `n_links` | int32 |
| `n_bold` | int32 |
| `n_photos` | int32 |
| `video_url` | string |
| `video_platform` | string |
| `video_id` | string |
| `video_source` | string |
| `team_handles` | list<item: string> |
| `team_size` | int32 |
| `likes` | int32 |
| `comments` | int32 |
| `n_updates` | int32 |
| `first_update_text` | string |
| `try_links` | list<item: string> |
| `submitted_to` | list<item: string> |
| `winner_texts` | list<item: string> |
| `is_winner_page` | bool |

**`projects_raw`**

| column | type |
|---|---|
| `slug` | string |
| `fetched_at` | string |
| `html_gz` | binary |

**`ledger (all stages)`**

| column | type |
|---|---|
| `stage` | string |
| `key` | string |
| `status` | string |
| `note` | string |
| `ts` | string |
| `worker` | string |

### 4.5 Gold table (built in Colab, one row per sampled hackathon-project, saved as `gold.parquet`)

Keys and label: `hackathon_id`, `slug`, `y` (1 if winner).
Event context: `registrations`, `log_prize` (ln(1+prize_value)), `online`, `n_sub` (gallery size), `year`, `days`, `themes`.
Project raw features: `team_size`, `built_with`, `story_words`, `n_h2`, `n_img`, `n_li`, `n_code`, `n_links`, `n_photos`, `has_video`, `video_platform`, `video_min` (from YouTube Data API or Vimeo oEmbed, optional), `tagline_len`, `title_len`, `try_links`, `headings`, `tagline`, `story_text`.
Derived: `has_github`, `has_demo`, `n_try_links`, `n_tech`, `has_inspiration_head`, `has_challenges_head`, `video_min_clip` (capped at 10), `<feature>_pct` (within-hackathon percentile rank) for `team_size`, `n_tech`, `story_words`, `n_h2`, `n_img`, `n_code`, `n_links`, `tagline_len`, and `tech_<tag>` multi-hot columns for the 120 most common tags.
Post-outcome fields kept for reference only and never used as predictors (a script assertion enforces this): `likes_post`, `comments_post`, `updates_post`.
There is no separate "media ratio" column; media is captured by `has_video`, `video_min`, `n_img` and `n_photos`.
The gold table has no handle columns. It does contain `story_text`, so keep it private.

## 5. Ethical scraper rules (as implemented)

HTTP client:
- `curl_cffi` session with `impersonate="chrome"` (the library's current Chrome TLS and HTTP/2 fingerprint), header `Accept-Language: en-US,en;q=0.9`, timeout 30 s, redirects followed. One session per worker so cookies persist. `httpx` is not used.
- Exactly one request in flight per worker.

Pacing:
- Gap before each request: `gap = base * mult * (1 + U(-0.4, 0.4))`, measured from the previous request's start.
- `base` is the `--delay` value (env `DELAY`, default 2.0 s; the Pi sets 2.5 s). That is about 0.5 and 0.4 requests per second per worker, and about 2 requests per second across all five workers on five different IPs.
- `mult` starts at 1.0, doubles on every block (capped at 16), and is multiplied by 0.9 on every success (floor 1.0).

Block detection: HTTP 403, 429 or 503; or header `cf-mitigated: challenge`; or a 200 whose first 3,000 characters contain "Just a moment...". A challenge is never solved, bypassed or retried aggressively.

Retry rules (up to 4 attempts per request):
- Block: honor `Retry-After` (capped at 300 s), then sleep `min(120, 10 * 2^attempt)` s, so 10, 20, 40, 80.
- Network exception: `mult` doubles, sleep `min(60, 5 * 2^attempt)`.
- Other 5xx or unexpected status: `mult` doubles, sleep `min(60, 3 * 2^attempt)`.
- 404 or 410 returns `gone` at once and is final.
- After 4 attempts the call returns `blocked` (if the last status was a block code) or `error`, and the ledger records it for a later retry.

Circuit breaker: a counter of consecutive blocked responses, reset by any success or 404. At 8 in a row the client raises `Blocked`; the worker flushes state, prints `::warning::blocked, cooling down until next run`, and exits 0 so the schedule continues. Never respond to a block by adding workers or lowering the delay.

Scope and etiquette: only public pages, no login, `robots.txt` has no Disallow for generic agents (VERIFIED). Devpost's Terms of Service were NOT verified: read them before running, and stop if they forbid automated access.

Privacy as built:
- The Hugging Face dataset is private and the GitHub repo is public but `.gitignore` excludes every data, probe and secret path.
- `projects.team_handles` holds raw Devpost handles and `projects.submitted_to` holds hackathon subdomains; `projects_raw` holds full HTML (names, avatar URLs). These exist only in the private dataset.
- The gold table contains no handles. No hashing or pseudonymization is implemented. If you ever share data, hash handles first and share only aggregates and model outputs.
- The HF token lives only in GitHub repository secrets and in `/etc/moneyball.env` (mode 600, root). It is never printed.

## 6. Devpost findings (VERIFIED 1 Oct 2026 unless noted)

- Hackathon list endpoint: `https://devpost.com/api/hackathons?status[]=ended&order_by=recently-added&page=N` returns JSON. 9 hackathons per page. `meta.total_count` was 13,809, so about 1,535 pages. `per_page` was not tested as a parameter.
- Hackathon fields: `id`, `title`, `url`, `submission_gallery_url`, `start_a_submission_url`, `open_state`, `submission_period_dates` (strings like "Sep 25 - 30, 2026"), `themes[{id,name}]`, `prize_amount` (HTML-wrapped, e.g. a span around the number), `prizes_counts{cash,other}`, `registrations_count`, `organization_name`, `winners_announced`, `invite_only`, `displayed_location{icon,location}`, `managed_by_devpost_badge`, `featured`, `thumbnail_url`.
- The list includes junk and test entries (for example "TEST ROWDYHACKS"). Filter with `winners_announced` and `--min-registrations`.
- `robots.txt`: `User-agent: *` with an empty `Disallow`; a few named bots (BLEXBot, Omgilibot, Omgili, ImagesiftBot, Bytespider) are disallowed.
- Gallery: `https://<sub>.devpost.com/project-gallery`, server-rendered. Projects link as `https://devpost.com/software/<slug>`. The page prints "1 - 11 of 11" (used as `expected`). Each card ends with likes then comments (card text "3 1" matched "Like 3" and "Comment 1" on the project page). The pagination parameter `?page=N` and the page size are UNVERIFIED; the loop stops when a page has no new slugs, and the per-hackathon `cards` versus `expected` comparison exposes a wrong guess.
- Project page, server-rendered: tagline in `og:description` (VERIFIED); built-with tags are links to `/software/built-with/<tag>` (VERIFIED); section headings "Built With", "Try it out", "Submitted to", "Created by", "Updates", "Submission history"; "Like N" and "Comment N" labels; first update line "started this project - <date time tz>" (the separator in the page is a dash character, matched by `[\u2014\u2013-]`); update links `/software/<slug>/updates/<id>`.
- UNVERIFIED: the winner badge's HTML. The parser flags a winner when a card or project page contains an element whose class contains "winner", an image whose alt or title contains "winner", or an element whose text is exactly "Winner". `WINNER_CSS` (env or GitHub variable) adds an extra selector. RUNBOOK section 3.4 is the procedure to confirm this before the long run.
- UNVERIFIED: whether the Pi's home IP and GitHub runner IPs get challenged; the `canary` command tests this.
- The fetch tool used from the build sandbox was not challenged on any page; that says little about your IPs.

Probe logic: `pipeline.py probe <url>` fetches one page, saves the HTML under `$WORKDIR/probe/`, and prints either parsed project fields or, for a gallery, `expected`, number of cards, number of winners and the first five cards. Compare these with what you see in a browser.

## 7. Analysis spec (colab/analysis.py)

1. Pull the dataset with `snapshot_download` (everything except raw HTML) into `/content/hf/data`, then create DuckDB views with `union_by_name=true`. The script does not use `hf://` paths.
2. Parse `period_text` into start and end dates (handles "Sep 25 - 30, 2026", "Aug 27 - Sep 26, 2026", single dates, and year wrap).
3. Optional video length enrichment (YouTube Data API 50 ids per call, Vimeo oEmbed).
4. Build the gold table and within-hackathon percentiles.
5. Conditional logit with hackathon fixed effects; odds ratio per standard deviation with confidence intervals; only hackathons with at least one winner and one non-winner contribute.
6. LightGBM with 5-fold `GroupKFold` grouped by hackathon; report pooled out-of-fold AUC and the mean within-hackathon AUC (the honest number).
7. SHAP TreeExplainer on up to 20,000 rows; summary plot and top features.
8. Text: `all-MiniLM-L6-v2` embeddings of tagline plus the first 1,000 characters, BERTopic (`min_topic_size=60`), win-rate lift per topic with Wilson intervals.
9. Method cautions: likes, comments and updates are post-outcome and excluded; "winner" mixes sponsor tracks; survivorship (deleted or unsubmitted projects are invisible); many tiny beginner events; many comparisons, so read intervals and keep untouched hackathons as a holdout.

Tested: the script ran end to end on synthetic data with planted effects; the conditional logit recovered writeup length and team size as top odds ratios and SHAP ranked the same two first. The text cell (BERTopic) and the YouTube enrichment were not run.

## 8. Session decisions and rejected alternatives

- Storage: Hugging Face datasets chosen. Supabase free (500 MB) and Mongo Atlas M0 (512 MB) cannot hold text plus raw HTML. GitHub releases require a public repo and would publish scraped handles (emergency fallback only).
- Coordination: hash sharding instead of a queue server (no locks, no coordinator).
- Bot handling: Chrome-fingerprinted client, slow pacing, adaptive backoff, circuit breaker, never solve challenges. Playwright on the Pi is a last resort for failing pages only.
- Cost lever: label everything from gallery cards, fetch details only for the sample.
- Video length is not on the page (only an embed), so it is enriched in Colab through the free YouTube Data API (10,000 quota units per day by default).
- Pi: tmpfs at `/mnt/moneyball-ram`, size ceiling 1g by default (the export outline asked for 512 MB; `SIZE=512m` works), journald and swap optional.
- Residential Pi IP is the safer worker; datacenter runner IPs are more likely to be blocked, hence the canary.

## 9. Deviations of the as-built system from the export outline

| Outline said | As built |
|---|---|
| CLI `worker` | There is no `worker` command. `run` does gallery then detail for one shard; `gallery` and `detail` run alone. Also `discover`, `canary`, `probe`, `status`, `compact` |
| `curl_cffi` / `httpx` | `curl_cffi` only |
| Discovery job matrix dispatch | Discovery is a single job. The matrix (shards 0 to 3) belongs to the `run` stage |
| `pi/setup_ramdisk.sh`, 512 MB | Script included; default size ceiling is 1g, override with `SIZE=512m` |
| `devpost-worker.service` / `.timer` | Files are named `moneyball.service` and `moneyball.timer`; all docs refer to those names |
| `tests/test_pipeline.py` | Provided (renamed from `test_offline.py`) and extended with limiter, breaker and resumption tests |
| DuckDB reads directly from the Hub | Script downloads with `snapshot_download`, then DuckDB reads the local Parquet |
| Handle anonymization guarantees | Handles are excluded from the gold table and kept in a private dataset; no hashing is implemented |
| Verified CSS selectors for winner badges | Tags, tagline, likes, comments, team and updates are verified; the winner badge is not (see section 6) |

## 10. Open items and first actions

Open and unverified: winner badge markup, gallery pagination parameter and size, API deep-paging behavior, IP reputation of the Pi and of GitHub runners, Devpost Terms of Service, Hugging Face upload and download paths (no credentials or network in the build sandbox), the Colab text and YouTube cells.

First actions in a new environment: unzip the export, follow RUNBOOK.md in order (secrets, Pi setup, canary, probe, discover, run, monitor, Colab), and do not start the long run until the probe in RUNBOOK 3.4 passes.

## 11. File inventory

| File | Purpose |
|---|---|
| `pipeline.py` | All stages: discover, gallery, detail, run, canary, probe, status, compact |
| `.github/workflows/ingest.yml` | Actions workflow: canary, discover, run (4 shards) |
| `pi/setup_ramdisk.sh` | One-shot Pi setup: tmpfs, venv, secrets file, systemd units |
| `pi/moneyball.service`, `pi/moneyball.timer` | Shard 4 nightly burst, all writes in RAM |
| `colab/analysis.py` | Cell-delimited Colab analysis |
| `tests/test_pipeline.py`, `tests/make_synthetic.py` | 18 offline tests; synthetic data generator for the analysis script |
| `requirements.txt`, `.gitignore` | Dependencies; secret and data exclusions |
| `PLAN.md`, `RUNBOOK.md` | Original plan; step-by-step deployment guide |
| `ARCHITECTURE_AND_SPECS.md` | This file |
| `tools/build_export.py`, `tools/restore_from_export.py` | Build `PROJECT_EXPORT.md` (every file in a fenced block with sha256) and restore all files from it |
| `PROJECT_EXPORT.md` | Generated single-file copy of everything above; regenerate with `python tools/build_export.py` |
````


## File 2: pipeline.py

### `pipeline.py`

<!-- FILE: pipeline.py sha256=6795af7b417bb3300369fc9a658f48604e2c226535627bb0ef169de6996a8894 -->
```python
#!/usr/bin/env python3
"""Moneyballing Hackathons: resumable, shardable Devpost ingestion.

Stages (each is ledger-backed, so a crash or a block costs at most one flush):
  discover   list ended hackathons from Devpost's public JSON endpoint
  gallery    walk each hackathon's project gallery, label winners, pick a detail sample
  detail     fetch + parse the sampled project pages (also keeps gzipped raw HTML)
  run        gallery then detail for one shard inside one time budget (what CI calls)
  canary     one request per endpoint; exit 3 if blocked (fail fast from a new IP)
  probe URL  fetch one page, save it, print what the parsers extract (verify selectors)
  compact    merge small ledger files (run while no worker is active)
  status     progress report from the remote ledger

State lives in a Hugging Face dataset repo (HF_REPO) as Parquet shards plus tiny ledger
files. Workers read nothing from local disk between runs, so any worker can die anytime.
Use --local-only to run everything against a local folder with no HF account.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq
from selectolax.parser import HTMLParser

API = "https://devpost.com/api/hackathons"
WINNER_CSS = os.environ.get("WINNER_CSS", "").strip()  # extra CSS selector that marks a winner badge
PARSER_VERSION = 1
DONE_STATUSES = {"ok", "empty", "gone"}
MAX_ATTEMPTS = 5  # poison-pill cutoff per key

# ----------------------------------------------------------------------------- schemas
_s, _i32, _i64, _b, _f = pa.string(), pa.int32(), pa.int64(), pa.bool_(), pa.float64()
_ls = pa.list_(pa.string())

SCHEMAS = {
    "hackathons": pa.schema([
        ("hackathon_id", _i64), ("title", _s), ("url", _s), ("gallery_url", _s),
        ("open_state", _s), ("period_text", _s), ("themes", _ls), ("prize_text", _s),
        ("prize_value", _f), ("prize_currency", _s), ("prizes_cash", _i32),
        ("prizes_other", _i32), ("registrations", _i32), ("organization", _s),
        ("winners_announced", _b), ("invite_only", _b), ("managed_by_devpost", _b),
        ("location", _s), ("is_online", _b), ("fetched_at", _s),
    ]),
    "cards": pa.schema([
        ("hackathon_id", _i64), ("slug", _s), ("is_winner", _b), ("winner_text", _s),
        ("likes", _i32), ("comments", _i32), ("profile_links", _i32), ("card_text", _s),
        ("page", _i32), ("detail_selected", _b), ("n_cards_in_hackathon", _i32),
        ("n_winners_in_hackathon", _i32), ("fetched_at", _s),
    ]),
    "projects": pa.schema([
        ("slug", _s), ("parser_version", _i32), ("fetched_at", _s), ("title", _s),
        ("tagline", _s), ("og_image", _s), ("built_with", _ls), ("story_text", _s),
        ("story_chars", _i32), ("story_words", _i32), ("story_root_found", _b),
        ("headings", _ls), ("n_h2", _i32), ("n_h3", _i32), ("n_img", _i32),
        ("n_li", _i32), ("n_code", _i32), ("n_links", _i32), ("n_bold", _i32),
        ("n_photos", _i32), ("video_url", _s), ("video_platform", _s), ("video_id", _s),
        ("video_source", _s), ("team_handles", _ls), ("team_size", _i32),
        ("likes", _i32), ("comments", _i32), ("n_updates", _i32),
        ("first_update_text", _s), ("try_links", _ls), ("submitted_to", _ls),
        ("winner_texts", _ls), ("is_winner_page", _b),
    ]),
    "projects_raw": pa.schema([("slug", _s), ("fetched_at", _s), ("html_gz", pa.binary())]),
}
LEDGER_SCHEMA = pa.schema([("stage", _s), ("key", _s), ("status", _s), ("note", _s),
                           ("ts", _s), ("worker", _s)])


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def shard_of(key, n: int) -> int:
    return int(sha(str(key))[:8], 16) % n


# ----------------------------------------------------------------------------- http
class Blocked(Exception):
    """Raised when the circuit breaker trips. Caller flushes state and exits cleanly."""


class Budget:
    def __init__(self, minutes: float):
        self.end = time.monotonic() + minutes * 60

    def over(self) -> bool:
        return time.monotonic() >= self.end


class Pacer:
    """Jittered spacing between requests, with a multiplier that backs off when blocked
    and decays slowly back to the base rate after successes."""

    def __init__(self, base: float, jitter: float = 0.4):
        self.base, self.jitter, self.mult, self.last = base, jitter, 1.0, 0.0

    def wait(self):
        import random
        gap = self.base * self.mult * (1 + random.uniform(-self.jitter, self.jitter))
        dt = time.monotonic() - self.last
        if dt < gap:
            time.sleep(gap - dt)
        self.last = time.monotonic()

    def ok(self):
        self.mult = max(1.0, self.mult * 0.9)

    def slow(self, retry_after: int = 0):
        self.mult = min(16.0, self.mult * 2)
        if retry_after:
            time.sleep(min(retry_after, 300))


class Client:
    """One in-flight request per worker. Returns (kind, status, text) where kind is
    ok | gone | blocked | error. A challenge page is treated as a block, never solved."""

    def __init__(self, base_delay: float, breaker: int = 8):
        from curl_cffi import requests as cr
        self.s = cr.Session(impersonate="chrome", timeout=30,
                            headers={"Accept-Language": "en-US,en;q=0.9"})
        self.pacer, self.consec, self.breaker = Pacer(base_delay), 0, breaker

    def get(self, url, params=None):
        last = None
        for attempt in range(4):
            self.pacer.wait()
            try:
                r = self.s.get(url, params=params, allow_redirects=True)
            except Exception as e:  # network hiccup, TLS reset, timeout
                last = f"exc:{type(e).__name__}"
                self.pacer.slow()
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            code, last = r.status_code, r.status_code
            body = r.text if code == 200 else ""
            blocked = (code in (403, 429, 503)
                       or r.headers.get("cf-mitigated") == "challenge"
                       or (code == 200 and "Just a moment..." in body[:3000]))
            if code == 200 and not blocked:
                self.consec = 0
                self.pacer.ok()
                return "ok", code, body
            if code in (404, 410):
                self.consec = 0
                return "gone", code, None
            if blocked:
                self.consec += 1
                ra = r.headers.get("retry-after") or ""
                self.pacer.slow(int(ra) if ra.isdigit() else 0)
                if self.consec >= self.breaker:
                    raise Blocked(f"{self.consec} consecutive blocks, last status {code}")
                time.sleep(min(120, 10 * 2 ** attempt))
                continue
            self.pacer.slow()  # 5xx and friends
            time.sleep(min(60, 3 * 2 ** attempt))
        return ("blocked" if last in (403, 429, 503) else "error"), last, None


# ----------------------------------------------------------------------------- parsing
SOFT = re.compile(r"^(?:https?://devpost\.com)?/software/([^/?#]+)/?$")
PROFILE = re.compile(r"^(?:https?://devpost\.com)?/([A-Za-z0-9_\-]+)/?$")
RESERVED = {"software", "hackathons", "settings", "portfolio", "submit-to", "users", "about"}
SUBDOMAIN = re.compile(r"^https?://([a-z0-9\-]+)\.devpost\.com/?$")
NON_HACK_SUB = {"www", "secure", "info", "help"}
VIDEO = [
    ("youtube", re.compile(r"youtube\.com/embed/([\w\-]{11})|youtu\.be/([\w\-]{11})|youtube\.com/watch\?v=([\w\-]{11})")),
    ("vimeo", re.compile(r"vimeo\.com/(?:video/)?(\d{6,})")),
    ("loom", re.compile(r"loom\.com/(?:embed|share)/(\w+)")),
    ("youku", re.compile(r"youku\.com/\S*?(\w{10,})")),
]
STORY_STOP = re.compile(r"\n(?:Built With|Try it out|Updates|Submission history)\n")
NON_STORY_H2 = {"built with", "try it out", "updates", "submission history"}


def _a(n, k):
    return (n.attributes.get(k) or "") if n is not None else ""


def _txt(n):
    return n.text(deep=True, separator=" ", strip=True) if n is not None else ""


def _i(v):
    return int(v) if v is not None else None


def parse_gallery(html: str):
    """Return (cards, expected_total). A card is the smallest ancestor of a project link
    that contains exactly one distinct project link, so it survives class renames."""
    t = HTMLParser(html)
    first, order = {}, []
    for a in t.css("a[href]"):
        m = SOFT.match(_a(a, "href").strip())
        if m and m.group(1) != "built-with" and m.group(1) not in first:
            first[m.group(1)] = a
            order.append(m.group(1))

    def slugs_in(node):
        out = set()
        for x in node.css("a[href]"):
            m = SOFT.match(_a(x, "href").strip())
            if m:
                out.add(m.group(1))
        return out

    cards = []
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
        win_txt = []
        for n in node.css("[class]"):
            if "winner" in _a(n, "class").lower():
                w = _txt(n)
                if w:
                    win_txt.append(w[:80])
        for n in node.css("img[alt], [title]"):
            v = _a(n, "alt") or _a(n, "title")
            if "winner" in v.lower():
                win_txt.append(v[:80])
        for n in node.css("span, small, b, strong, div, p"):
            s = _txt(n)
            if s.lower() == "winner":
                win_txt.append(s)
        if WINNER_CSS:
            for n in node.css(WINNER_CSS):
                win_txt.append(_txt(n)[:80] or "winner")
        m = re.search(r"(\d+)\s+(\d+)\s*$", text)  # trailing "likes comments"
        profiles = {_a(x, "href") for x in node.css("a[href]")
                    if (pm := PROFILE.match(_a(x, "href").strip())) and pm.group(1) not in RESERVED}
        cards.append({
            "slug": slug, "is_winner": bool(win_txt), "winner_text": (win_txt[0] if win_txt else None),
            "likes": int(m.group(1)) if m else None, "comments": int(m.group(2)) if m else None,
            "profile_links": len(profiles), "card_text": text[:600],
        })
    body = t.body.text(deep=True, separator=" ", strip=True) if t.body else ""
    m = re.search(r"(\d+)\s*[\u2013\-]\s*(\d+)\s*of\s*([\d,]+)", body)
    expected = int(m.group(3).replace(",", "")) if m else None
    return cards, expected


def _next_list_after(h, steps=8):
    for start in (h, h.parent):
        n = start.next if start is not None else None
        for _ in range(steps):
            if n is None:
                break
            if n.tag == "ul":
                return n
            n = n.next
    return None


def _heading(t, prefix):
    for h in t.css("h2, h3, h4, h5"):
        if _txt(h).lower().startswith(prefix):
            return h
    return None


def _count_label(t, word):
    rx = re.compile(rf"^{word}\s*(\d+)$")
    for n in t.css("a, button, span"):
        m = rx.match(_txt(n))
        if m:
            return int(m.group(1))
    return None


def parse_project(html: str) -> dict:
    t = HTMLParser(html)
    for n in t.css("script, style, noscript"):
        n.decompose()

    def meta(p):
        n = t.css_first(f'meta[property="{p}"]') or t.css_first(f'meta[name="{p}"]')
        return _a(n, "content") or None

    h1 = t.css_first("h1")
    title = _txt(h1) or meta("og:title")
    tagline = meta("og:description")

    built = []
    for a in t.css('a[href*="/software/built-with/"]'):
        tag = unquote(_a(a, "href").rstrip("/").rsplit("/", 1)[-1]).lower()
        if tag and tag not in built:
            built.append(tag)

    root = t.css_first("#app-details-left") or t.css_first("#app-details")
    found = root is not None
    root = root or t.body or t.root
    full = root.text(deep=True, separator="\n", strip=True) if root is not None else ""
    parts = STORY_STOP.split("\n" + full + "\n", maxsplit=1)
    story = parts[0].strip()
    heads = [_txt(h) for h in root.css("h2, h3")] if root is not None else []
    story_heads = [h for h in heads if h.lower() not in NON_STORY_H2]
    n_h2 = len([h for h in root.css("h2") if _txt(h).lower() not in NON_STORY_H2]) if root is not None else 0
    n_links = len([a for a in root.css("a[href]") if "/software/built-with/" not in _a(a, "href")]) if root is not None else 0

    v_url = v_plat = v_id = v_src = None
    cands = [(_a(n, "src") or _a(n, "data-src"), "iframe") for n in t.css("iframe")]
    cands += [(_a(n, "href"), "link") for n in t.css("a[href]")]
    for url, src in cands:
        for plat, rx in VIDEO:
            m = rx.search(url)
            if m and not v_url:
                v_url, v_plat, v_src = url, plat, src
                v_id = next((g for g in m.groups() if g), None)
        if v_url and v_src == "iframe":
            break

    handles = []
    h = _heading(t, "created by")
    ul = _next_list_after(h) if h is not None else None
    for a in (ul.css("a[href]") if ul is not None else t.css("#app-team a[href]")):
        pm = PROFILE.match(_a(a, "href").strip())
        if pm and pm.group(1) not in RESERVED and pm.group(1) not in handles:
            handles.append(pm.group(1))

    tries = []
    h = _heading(t, "try it out")
    ul = _next_list_after(h) if h is not None else None
    if ul is not None:
        tries = [_a(a, "href") for a in ul.css("a[href]") if _a(a, "href")]

    subs = []
    for a in t.css("a[href]"):
        m = SUBDOMAIN.match(_a(a, "href").strip())
        if m and m.group(1) not in NON_HACK_SUB and _a(a, "href").strip() not in subs:
            subs.append(_a(a, "href").strip())

    upd = {m.group(0) for a in t.css("a[href]") if (m := re.search(r"/software/[^/]+/updates/\d+", _a(a, "href")))}
    page_text = _txt(t.body) if t.body else ""
    m = re.search(r"started this project\s*[\u2014\u2013\-]\s*([A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M [A-Z]{2,4})", page_text)

    wins = []
    for n in t.css("[class]"):
        if "winner" in _a(n, "class").lower():
            w = _txt(n)[:120]
            if w and w not in wins:
                wins.append(w)
    for n in t.css("img[alt]"):
        if "winner" in _a(n, "alt").lower() and _a(n, "alt") not in wins:
            wins.append(_a(n, "alt")[:120])

    if WINNER_CSS:
        for n in t.css(WINNER_CSS):
            w = _txt(n)[:120] or "winner"
            if w not in wins:
                wins.append(w)
    photos = {_a(n, "src") for n in t.css("img[src]") if "software_photos" in _a(n, "src")}
    return {
        "slug": None, "parser_version": PARSER_VERSION, "fetched_at": now(), "title": title,
        "tagline": tagline, "og_image": meta("og:image"), "built_with": built,
        "story_text": story, "story_chars": len(story), "story_words": len(story.split()),
        "story_root_found": found, "headings": story_heads, "n_h2": n_h2,
        "n_h3": len(root.css("h3")) if root is not None else 0,
        "n_img": len(root.css("img")) if root is not None else 0,
        "n_li": len(root.css("li")) if root is not None else 0,
        "n_code": len(root.css("pre, code")) if root is not None else 0,
        "n_links": n_links, "n_bold": len(root.css("b, strong")) if root is not None else 0,
        "n_photos": len(photos), "video_url": v_url, "video_platform": v_plat,
        "video_id": v_id, "video_source": v_src, "team_handles": handles,
        "team_size": len(handles), "likes": _count_label(t, "Like"),
        "comments": _count_label(t, "Comment"), "n_updates": len(upd),
        "first_update_text": m.group(1) if m else None, "try_links": tries,
        "submitted_to": subs, "winner_texts": wins[:5], "is_winner_page": bool(wins),
    }


def hack_row(h: dict) -> dict:
    pt = re.sub(r"<[^>]+>", "", h.get("prize_amount") or "").strip()
    m = re.search(r"([\d,]+(?:\.\d+)?)", pt)
    loc = h.get("displayed_location") or {}
    pc = h.get("prizes_counts") or {}
    return {
        "hackathon_id": h.get("id"), "title": h.get("title"), "url": h.get("url"),
        "gallery_url": h.get("submission_gallery_url"), "open_state": h.get("open_state"),
        "period_text": h.get("submission_period_dates"),
        "themes": [x.get("name") for x in (h.get("themes") or []) if x.get("name")],
        "prize_text": pt, "prize_value": float(m.group(1).replace(",", "")) if m else None,
        "prize_currency": re.sub(r"[\d,\.\s]", "", pt) or None,
        "prizes_cash": pc.get("cash"), "prizes_other": pc.get("other"),
        "registrations": h.get("registrations_count"), "organization": h.get("organization_name"),
        "winners_announced": h.get("winners_announced"), "invite_only": h.get("invite_only"),
        "managed_by_devpost": h.get("managed_by_devpost_badge"),
        "location": loc.get("location"), "is_online": loc.get("icon") == "globe",
        "fetched_at": now(),
    }


# ----------------------------------------------------------------------------- state
class Store:
    """Buffers rows and ledger entries, then ships them as ONE commit per flush."""

    def __init__(self, work, repo, token, wid, local_only, flush_rows=300, flush_secs=600):
        self.work, self.repo, self.token, self.wid, self.local = Path(work), repo, token, wid, local_only
        self.out, self.cache = self.work / "out", self.work / "hf"
        self.buf = {t: [] for t in SCHEMAS}
        self.led, self.t0 = [], time.time()
        self.flush_rows, self.flush_secs = flush_rows, flush_secs
        self.out.mkdir(parents=True, exist_ok=True)
        if not local_only:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)
            try:
                self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            except Exception:  # fine-grained token without create-repo rights: just confirm access
                self.api.repo_info(repo, repo_type="dataset")

    def add(self, table, row):
        self.buf[table].append(row)

    def log(self, stage, key, status, note=""):
        self.led.append({"stage": stage, "key": str(key), "status": status, "note": str(note)[:500],
                         "ts": now(), "worker": self.wid})

    def maybe_flush(self):
        n = sum(len(v) for v in self.buf.values())
        if n >= self.flush_rows or (self.led and time.time() - self.t0 >= self.flush_secs):
            self.flush()

    def _write(self, rel, schema, rows):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), p, compression="zstd")
        return rel, p

    def flush(self):
        ts = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        files = [self._write(f"data/{t}/{self.wid}-{ts}.parquet", SCHEMAS[t], rows)
                 for t, rows in self.buf.items() if rows]
        stages = {}
        for r in self.led:
            stages.setdefault(r["stage"], []).append(r)
        files += [self._write(f"ledger/{s}/{self.wid}-{ts}.parquet", LEDGER_SCHEMA, rows)
                  for s, rows in stages.items()]
        if not files:
            return
        if not self.local:
            from huggingface_hub import CommitOperationAdd
            ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)) for rel, p in files]
            sent = False
            for i in range(4):
                try:
                    self.api.create_commit(self.repo, repo_type="dataset", operations=ops,
                                           commit_message=f"{self.wid} {ts}")
                    sent = True
                    break
                except Exception as e:
                    print(f"[store] commit failed ({e}); retry {i + 1}", file=sys.stderr)
                    time.sleep(10 * 2 ** i)
            for _, p in files:
                if sent:
                    p.unlink(missing_ok=True)
                else:
                    p.unlink(missing_ok=True)  # rewritten on the next flush
            if not sent:
                print("[store] flush failed; rows stay in memory for the next attempt", file=sys.stderr)
                return
        self.buf = {t: [] for t in SCHEMAS}
        self.led, self.t0 = [], time.time()

    def _pull(self, prefix):
        if self.local:
            return sorted((self.out / prefix).glob("*.parquet"))
        from huggingface_hub import snapshot_download
        d = snapshot_download(self.repo, repo_type="dataset", allow_patterns=[f"{prefix}/*"],
                              local_dir=str(self.cache), token=self.token)
        return sorted((Path(d) / prefix).glob("*.parquet"))

    def done(self, stage):
        """Return ({key: status} for finished keys, Counter of failed attempts per key)."""
        ok, bad = {}, Counter()
        paths = self._pull(f"ledger/{stage}")
        if paths:
            d = pa.concat_tables([pq.read_table(p) for p in paths]).to_pydict()
            for k, s in zip(d["key"], d["status"]):
                if s in DONE_STATUSES:
                    ok[k] = s
                else:
                    bad[k] += 1
        return ok, bad

    def read(self, table, columns=None):
        rows = []
        for p in self._pull(f"data/{table}"):
            rows += pq.read_table(p, columns=columns).to_pylist()
        return rows


# ----------------------------------------------------------------------------- stages
def cmd_discover(a, st, cl, bud):
    ok, _ = st.done("discover")
    page = 1
    while not bud.over():
        key = f"page:{page}"
        if key in ok and not a.refresh:
            if ok[key] == "empty":
                break
            page += 1
            continue
        kind, code, text = cl.get(API, params={"status[]": "ended", "order_by": "recently-added", "page": page})
        if kind != "ok":
            st.log("discover", key, kind, code)
            page += 1
            continue
        try:
            hs = json.loads(text).get("hackathons") or []
        except ValueError:
            st.log("discover", key, "error", "bad json")
            page += 1
            continue
        if not hs:
            st.log("discover", key, "empty")
            break
        for h in hs:
            st.add("hackathons", hack_row(h))
        st.log("discover", key, "ok", len(hs))
        st.maybe_flush()
        page += 1
    st.flush()


def pick_sample(rows, k, seed):
    non = sorted((c for c in rows if not c["is_winner"]), key=lambda c: sha(f"{seed}:{c['slug']}"))
    chosen = {c["slug"] for c in rows if c["is_winner"]} | {c["slug"] for c in non[:k]}
    for c in rows:
        c["detail_selected"] = c["slug"] in chosen


def cmd_gallery(a, st, cl, bud):
    ok, bad = st.done("gallery")
    cols = ["hackathon_id", "gallery_url", "winners_announced", "registrations"]
    hs = {r["hackathon_id"]: r for r in st.read("hackathons", cols)}
    todo = [h for h in hs.values()
            if h["winners_announced"] and h["gallery_url"]
            and (h["registrations"] or 0) >= a.min_registrations
            and shard_of(h["hackathon_id"], a.nshards) == a.shard
            and str(h["hackathon_id"]) not in ok and bad[str(h["hackathon_id"])] < a.max_attempts]
    todo.sort(key=lambda h: -(h["registrations"] or 0))
    print(f"[gallery] shard {a.shard}/{a.nshards}: {len(todo)} hackathons to do", flush=True)
    for h in todo:
        if bud.over():
            break
        hid, cards, expected, status, code, page = h["hackathon_id"], {}, None, "ok", None, 1
        while page <= a.max_gallery_pages:
            kind, code, text = cl.get(h["gallery_url"], params={"page": page})
            if kind != "ok":
                status = kind
                break
            cs, exp = parse_gallery(text)
            expected = exp or expected
            new = [c for c in cs if c["slug"] not in cards]
            if not new:
                break
            for c in new:
                c["page"] = page
                cards[c["slug"]] = c
            if expected and len(cards) >= expected:
                break
            page += 1
        if status != "ok":  # drop partial work; the ledger entry makes it retry later
            st.log("gallery", hid, status, code)
            continue
        rows = list(cards.values())
        pick_sample(rows, a.nonwinners_per_hack, a.seed)
        nw = sum(c["is_winner"] for c in rows)
        for c in rows:
            c.update(hackathon_id=hid, n_cards_in_hackathon=len(rows), n_winners_in_hackathon=nw, fetched_at=now())
            st.add("cards", c)
        st.log("gallery", hid, "ok" if rows else "empty",
               json.dumps({"cards": len(rows), "winners": nw, "expected": expected}))
        st.maybe_flush()
    st.flush()


def cmd_detail(a, st, cl, bud):
    ok, bad = st.done("detail")
    seen, todo = set(), []
    for c in st.read("cards", ["slug", "detail_selected"]):
        s = c["slug"]
        if (c["detail_selected"] and s not in seen and s not in ok and bad[s] < a.max_attempts
                and shard_of(s, a.nshards) == a.shard):
            seen.add(s)
            todo.append(s)
    # random order keeps a partially finished run an unbiased sample of winners and non-winners
    todo.sort(key=lambda s: sha(f"{a.seed}:{s}"))
    print(f"[detail] shard {a.shard}/{a.nshards}: {len(todo)} projects to do", flush=True)
    for slug in todo:
        if bud.over():
            break
        kind, code, text = cl.get(f"https://devpost.com/software/{slug}")
        if kind == "ok":
            row = parse_project(text)
            row["slug"] = slug
            st.add("projects", row)
            st.add("projects_raw", {"slug": slug, "fetched_at": row["fetched_at"],
                                    "html_gz": gzip.compress(text.encode())})
            st.log("detail", slug, "ok")
        else:
            st.log("detail", slug, kind, code)
        st.maybe_flush()
    st.flush()


def cmd_run(a, st, cl, bud):
    cmd_gallery(a, st, cl, bud)
    if not bud.over():
        cmd_detail(a, st, cl, bud)


def cmd_canary(a, cl):
    tests = [("api", API, {"status[]": "ended", "order_by": "recently-added", "page": 1}),
             ("gallery", "https://cmu-hackathon.devpost.com/project-gallery", None),
             ("project", "https://devpost.com/software/coterie-57fv6e", None)]
    bad = False
    for name, url, params in tests:
        try:
            kind, code, _ = cl.get(url, params=params)
        except Blocked as e:
            kind, code = "blocked", str(e)
        print(f"[canary] {name}: {kind} {code}")
        bad |= kind != "ok"
    sys.exit(3 if bad else 0)


def cmd_probe(a, cl):
    kind, code, text = cl.get(a.url)
    print(f"[probe] {kind} {code}")
    if kind != "ok":
        return
    d = Path(a.workdir) / "probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / (sha(a.url)[:10] + ".html")).write_text(text)
    if re.search(r"/software/[^/?#]+/?$", a.url) and "built-with" not in a.url:
        r = parse_project(text)
        r["story_text"] = r["story_text"][:300]
        print(json.dumps(r, indent=2, default=str))
    else:
        cards, exp = parse_gallery(text)
        print(f"expected={exp} cards={len(cards)} winners={sum(c['is_winner'] for c in cards)}")
        for c in cards[:5]:
            print(c)


def cmd_status(a, st):
    """Progress report from the remote ledger: final status per key, per stage."""
    cols = ["hackathon_id", "winners_announced"]
    hk = st.read("hackathons", cols)
    eligible = {str(r["hackathon_id"]) for r in hk if r["winners_announced"]}
    sel = {r["slug"] for r in st.read("cards", ["slug", "detail_selected"]) if r["detail_selected"]}
    totals = {"discover": None, "gallery": len(eligible), "detail": len(sel)}
    print(f"hackathons discovered: {len(hk)} | with winners announced: {len(eligible)} | detail sample: {len(sel)}")
    for stage in ("discover", "gallery", "detail"):
        paths = st._pull(f"ledger/{stage}")
        if not paths:
            print(f"{stage:9s} no ledger yet")
            continue
        d = pa.concat_tables([pq.read_table(p) for p in paths]).to_pylist()
        d.sort(key=lambda r: r["ts"])
        final = {r["key"]: r["status"] for r in d}
        c = Counter(final.values())
        done = sum(v for k, v in c.items() if k in DONE_STATUSES)
        tot = totals[stage]
        pct = f" ({100 * done / tot:.1f}% of {tot})" if tot else ""
        print(f"{stage:9s} done {done}{pct} | " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
        last = d[-1]
        per = Counter(r["worker"] for r in d if r["ts"] >= now()[:10] and r["status"] == "ok")
        print(f"          last write {last['ts']} by {last['worker']} | ok today (UTC): " + (", ".join(f"{k}={v}" for k, v in sorted(per.items())) or "none"))


def cmd_compact(a, st):
    if st.local:
        print("local mode: nothing to compact")
        return
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    ts = time.strftime("%Y%m%dT%H%M%S")
    for stage in ("discover", "gallery", "detail"):
        paths = st._pull(f"ledger/{stage}")
        if len(paths) < 20:
            continue
        rel = f"ledger/{stage}/compact-{ts}.parquet"
        _, p = st._write(rel, LEDGER_SCHEMA, pa.concat_tables([pq.read_table(x) for x in paths]).to_pylist())
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p))]
        ops += [CommitOperationDelete(path_in_repo=f"ledger/{stage}/{x.name}") for x in paths]
        st.api.create_commit(st.repo, repo_type="dataset", operations=ops, commit_message=f"compact {stage}")
        print(f"[compact] {stage}: {len(paths)} files -> 1")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["discover", "gallery", "detail", "run", "canary", "probe", "compact", "status"])
    ap.add_argument("url", nargs="?")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--delay", type=float, default=float(os.environ.get("DELAY", "2.0")), help="base seconds between requests per worker")
    ap.add_argument("--workdir", default=os.environ.get("WORKDIR", "/tmp/moneyball"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nonwinners-per-hack", type=int, default=40)
    ap.add_argument("--min-registrations", type=int, default=0)
    ap.add_argument("--max-gallery-pages", type=int, default=60)
    ap.add_argument("--refresh", action="store_true", help="discover: ignore ledger and re-walk all pages")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                    help="skip keys that already failed this many times (raise to retry poisoned keys)")
    ap.add_argument("--local-only", action="store_true")
    a = ap.parse_args()

    wid = os.environ.get("WORKER_ID") or platform.node() or "worker"
    repo, token = os.environ.get("HF_REPO"), os.environ.get("HF_TOKEN")
    cl = Client(a.delay)
    if a.cmd == "canary":
        return cmd_canary(a, cl)
    if a.cmd == "probe":
        return cmd_probe(a, cl)
    if not a.local_only and not (repo and token):
        sys.exit("Set HF_REPO and HF_TOKEN, or pass --local-only")
    st = Store(a.workdir, repo, token, f"{wid}", a.local_only)
    if a.cmd == "compact":
        return cmd_compact(a, st)
    if a.cmd == "status":
        return cmd_status(a, st)
    bud = Budget(a.max_minutes)

    def _term(*_):  # systemctl stop / Actions cancel: flush what we have, then exit
        raise KeyboardInterrupt

    import signal
    signal.signal(signal.SIGTERM, _term)
    fn = {"discover": cmd_discover, "gallery": cmd_gallery, "detail": cmd_detail, "run": cmd_run}[a.cmd]
    try:
        fn(a, st, cl, bud)
    except Blocked as e:
        st.flush()
        print(f"::warning::blocked, cooling down until next run ({e})")
    except KeyboardInterrupt:
        st.flush()
        print("[main] interrupted; state flushed", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
```


## File 3: github-actions-ingest.yml (lives at .github/workflows/ingest.yml)

### `.github/workflows/ingest.yml`

<!-- FILE: .github/workflows/ingest.yml sha256=cb1b98ad772d6c212c168dc4c4c275a4d66115bd34cc4798190a4533dda5328e -->
```yaml
name: ingest

# Daily burst. Four shards run in parallel, each with its own IP and its own slice of the work.
# Shard 4 of 5 belongs to the Raspberry Pi, so the Actions matrix only covers 0..3.
on:
  schedule:
    - cron: "17 2 * * *"          # 02:17 UTC, off the hour on purpose
  workflow_dispatch:
    inputs:
      stage:
        description: "canary (one job, tests IP reputation), discover (run once), run (gallery + detail, sharded)"
        default: "run"
        type: choice
        options: [run, discover, canary]
      max_minutes:
        description: "time budget per job"
        default: "330"
      delay:
        description: "base seconds between requests (raise after a block)"
        default: ""

permissions:
  contents: read

concurrency:
  group: ingest
  cancel-in-progress: false

env:
  HF_REPO: ${{ vars.HF_REPO }}          # repository variable, e.g. youruser/hackathon-moneyball
  HF_TOKEN: ${{ secrets.HF_TOKEN }}     # fine-grained token, write access to that one dataset repo only
  WINNER_CSS: ${{ vars.WINNER_CSS }}    # optional, set after running probe if the default badge check misses
  DELAY: ${{ inputs.delay || vars.DELAY || '2.0' }}
  PYTHONDONTWRITEBYTECODE: "1"

jobs:
  canary:
    if: ${{ github.event_name == 'workflow_dispatch' && inputs.stage == 'canary' }}
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - run: python pipeline.py canary

  discover:
    if: ${{ github.event_name == 'workflow_dispatch' && inputs.stage == 'discover' }}
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - run: python pipeline.py canary
      - run: python pipeline.py discover --max-minutes 100
        env: { WORKER_ID: gha-discover }

  scrape:
    if: ${{ github.event_name == 'schedule' || inputs.stage == 'run' }}
    runs-on: ubuntu-latest
    timeout-minutes: 355
    strategy:
      fail-fast: false
      matrix:
        shard: [0, 1, 2, 3]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - name: Canary (fail fast if this runner's IP range is blocked)
        run: python pipeline.py canary
      - name: Gallery then detail for this shard
        env:
          WORKER_ID: gha-${{ matrix.shard }}
          WORKDIR: ${{ runner.temp }}/moneyball
        run: >
          python pipeline.py run
          --shard ${{ matrix.shard }} --nshards 5
          --max-minutes ${{ inputs.max_minutes || '330' }}
          --nonwinners-per-hack 40
```


## File 4: Raspberry Pi setup script and systemd units

### `pi/setup_ramdisk.sh`

<!-- FILE: pi/setup_ramdisk.sh sha256=b751232e5c829454cc6ef032561344b6077fdf89090bcb30b5bc18cff237fa93 -->
```bash
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
```

### `pi/moneyball.service`

<!-- FILE: pi/moneyball.service sha256=235764cfd02aa103c8860baf2a7c04af7a3a98a9790a05edc964fa29db875a42 -->
```ini
# /etc/systemd/system/moneyball.service
# The Pi owns shard 4 of 5. Everything the job writes lands on the tmpfs at /mnt/moneyball-ram.
# Code and the venv are only read from disk.
[Unit]
Description=Moneyball Devpost ingestion burst (shard 4)
Wants=network-online.target
After=network-online.target
RequiresMountsFor=/mnt/moneyball-ram

[Service]
Type=oneshot
User=__PI_USER__
WorkingDirectory=/opt/moneyball
EnvironmentFile=/etc/moneyball.env
Environment=WORKER_ID=pi5
Environment=WORKDIR=/mnt/moneyball-ram/work
Environment=HF_HOME=/mnt/moneyball-ram/hf_home
Environment=XDG_CACHE_HOME=/mnt/moneyball-ram/cache
Environment=TMPDIR=/mnt/moneyball-ram/tmp
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStartPre=/usr/bin/mkdir -p /mnt/moneyball-ram/work /mnt/moneyball-ram/hf_home /mnt/moneyball-ram/cache /mnt/moneyball-ram/tmp
ExecStart=/opt/moneyball/.venv/bin/python pipeline.py run --shard 4 --nshards 5 --max-minutes 180
ExecStopPost=/bin/sh -c 'rm -rf /mnt/moneyball-ram/work /mnt/moneyball-ram/hf_home /mnt/moneyball-ram/cache /mnt/moneyball-ram/tmp'
TimeoutStartSec=4h
Nice=10
MemoryMax=2G
```

### `pi/moneyball.timer`

<!-- FILE: pi/moneyball.timer sha256=29b854cb9e0127ce202fbc8122b581cbebae1993f7192132aad540f935d6025d -->
```ini
# /etc/systemd/system/moneyball.timer
# Enable with: sudo systemctl enable --now moneyball.timer
# /etc/moneyball.env (chmod 600, owner root) holds:
#   HF_REPO=youruser/hackathon-moneyball
#   HF_TOKEN=hf_xxx
[Unit]
Description=Run the moneyball burst nightly

[Timer]
OnCalendar=*-*-* 01:40:00
RandomizedDelaySec=20min
Persistent=true

[Install]
WantedBy=timers.target
```


## File 5: colab/analysis.py

### `colab/analysis.py`

<!-- FILE: colab/analysis.py sha256=20e4dd7512a3d09f1acd89c6ce51be136b80f81a95adb5d816db15cdc131a2ab -->
```python
# Moneyballing Hackathons: Colab analysis. Cells are separated by "# %%" (Colab and VS Code both honor it).
# Free Colab is enough: DuckDB reads Parquet out of core, LightGBM runs on CPU, embeddings use the T4.

# %% 0. Config. In Colab, add secrets HF_TOKEN (read access), HF_REPO and optionally YT_API_KEY
#      (key icon in the left sidebar, then switch on "Notebook access").
import os, re, sys, json, time, glob, gzip, subprocess
from pathlib import Path
try:
    from google.colab import userdata
    for _k in ("HF_TOKEN", "HF_REPO", "YT_API_KEY"):
        try: os.environ.setdefault(_k, userdata.get(_k))
        except Exception: pass
except ImportError:
    pass

HF_REPO = os.environ.get("HF_REPO", "YOUR_USER/hackathon-moneyball")
DATA = os.environ.get("DATA_DIR", "/content/hf/data")     # where snapshot_download lands
OUT = Path(os.environ.get("OUT_DIR", "/content/out")); OUT.mkdir(parents=True, exist_ok=True)
SKIP_PULL = os.environ.get("SKIP_PULL") == "1"
RUN_TEXT = os.environ.get("RUN_TEXT", "1") == "1"
YT_KEY = os.environ.get("YT_API_KEY")                      # free key, Google account only

# %% 1. Install (Colab only; skipped elsewhere)
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "duckdb", "lightgbm", "shap", "bertopic",
                    "sentence-transformers", "statsmodels", "huggingface_hub", "selectolax", "curl_cffi"], check=True)

# %% 2. Pull the dataset. Raw HTML is NOT pulled here; it is only needed when you re-parse.
if not SKIP_PULL:
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(Path(DATA).parent),
                      allow_patterns=["data/hackathons/*", "data/cards/*", "data/projects/*", "enrich/*"],
                      token=os.environ.get("HF_TOKEN"))

# %% 3. DuckDB views. union_by_name is the schema-evolution mechanism: new parser columns
#      show up as NULL in old files, and the newest parser_version wins per slug.
import duckdb, numpy as np, pandas as pd
con = duckdb.connect()
con.sql(f"""CREATE OR REPLACE VIEW hackathons AS
  SELECT * FROM read_parquet('{DATA}/hackathons/*.parquet', union_by_name=true)
  QUALIFY row_number() OVER (PARTITION BY hackathon_id ORDER BY fetched_at DESC) = 1""")
con.sql(f"""CREATE OR REPLACE VIEW cards AS
  SELECT * FROM read_parquet('{DATA}/cards/*.parquet', union_by_name=true)
  QUALIFY row_number() OVER (PARTITION BY hackathon_id, slug ORDER BY fetched_at DESC) = 1""")
con.sql(f"""CREATE OR REPLACE VIEW projects AS
  SELECT * FROM read_parquet('{DATA}/projects/*.parquet', union_by_name=true)
  QUALIFY row_number() OVER (PARTITION BY slug ORDER BY parser_version DESC, fetched_at DESC) = 1""")
print(con.sql("""SELECT (SELECT count(*) FROM hackathons) hackathons,
                        (SELECT count(*) FROM cards) cards,
                        (SELECT count(*) FROM cards WHERE is_winner) winner_cards,
                        (SELECT count(*) FROM projects) detailed""").df())

# %% 4. Period text -> dates. Devpost gives strings like "Sep 25 - 30, 2026" or "Aug 27 - Sep 26, 2026".
from datetime import date
MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}

def parse_period(s):
    if not s: return (None, None)
    yrs = re.findall(r"\d{4}", s)
    mons = re.findall(r"[A-Z][a-z]{2}", s)
    days = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", re.sub(r"\d{4}", "", s))
    if not yrs or not mons or not days or mons[0] not in MON or mons[-1] not in MON: return (None, None)
    ey = int(yrs[-1]); sy = int(yrs[0]) if len(yrs) > 1 else ey
    sm, em = MON[mons[0]], MON[mons[-1]]
    if len(yrs) == 1 and sm > em: sy -= 1
    try: return date(sy, sm, int(days[0])), date(ey, em, int(days[-1]))
    except ValueError: return (None, None)

hk = con.sql("SELECT * FROM hackathons").df()
hk[["start", "end"]] = hk.period_text.apply(lambda s: pd.Series(parse_period(s)))
hk["year"] = pd.to_datetime(hk["end"]).dt.year
hk["days"] = (pd.to_datetime(hk["end"]) - pd.to_datetime(hk["start"])).dt.days + 1
con.register("hk", hk)

# %% 5. Video length (optional enrichment). YouTube Data API: 1 quota unit per 50 ids, 10k units/day free.
import requests
def iso_to_min(d):
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    return None if not m else (int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + int(m[3] or 0)) / 60

def enrich_video():
    vids = con.sql("SELECT DISTINCT video_platform p, video_id i FROM projects WHERE video_id IS NOT NULL").df()
    cache = OUT / "video_minutes.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["video_id", "minutes"])
    todo = vids[~vids.i.isin(have.video_id)]
    rows = []
    yt = todo[todo.p == "youtube"].i.tolist()
    for k in range(0, len(yt), 50):
        r = requests.get("https://www.googleapis.com/youtube/v3/videos",
                         params={"part": "contentDetails", "id": ",".join(yt[k:k + 50]), "key": YT_KEY}, timeout=30)
        got = {x["id"]: iso_to_min(x["contentDetails"]["duration"]) for x in r.json().get("items", [])}
        rows += [(i, got.get(i)) for i in yt[k:k + 50]]          # unavailable videos stay NaN
    for i in todo[todo.p == "vimeo"].i.tolist():
        try:
            j = requests.get("https://vimeo.com/api/oembed.json", params={"url": f"https://vimeo.com/{i}"}, timeout=20).json()
            rows.append((i, j.get("duration", 0) / 60))
        except Exception:
            rows.append((i, None))
        time.sleep(0.5)
    out = pd.concat([have, pd.DataFrame(rows, columns=["video_id", "minutes"])]).drop_duplicates("video_id", keep="last")
    out.to_parquet(cache)
    return out

vm = enrich_video() if YT_KEY else pd.DataFrame(columns=["video_id", "minutes"])
con.register("vm", vm)

# %% 6. Gold table: one row per (hackathon, project) that got a detail fetch.
base = con.sql("""
  SELECT c.hackathon_id, c.slug, c.is_winner::INT AS y, c.n_cards_in_hackathon AS n_sub,
         h.registrations, ln(1 + coalesce(h.prize_value, 0)) AS log_prize, h.is_online::INT AS online,
         h.year, h.days, h.themes,
         p.team_size, p.built_with, p.story_words, p.n_h2, p.n_img, p.n_li, p.n_code, p.n_links, p.n_photos,
         (p.video_url IS NOT NULL)::INT AS has_video, p.video_platform, vm.minutes AS video_min,
         length(p.tagline) AS tagline_len, length(p.title) AS title_len, p.try_links, p.headings,
         p.likes AS likes_post, p.comments AS comments_post, p.n_updates AS updates_post,
         p.tagline, p.story_text
  FROM cards c JOIN hk h USING (hackathon_id) JOIN projects p USING (slug)
  LEFT JOIN vm ON vm.video_id = p.video_id
  WHERE p.story_root_found
""").df()
print(len(base), "rows;", base.y.sum(), "winners")

b = base
b["has_github"] = b.try_links.apply(lambda l: int(any("github.com" in x for x in (l if l is not None else []))))
b["has_demo"] = b.try_links.apply(lambda l: int(any(("github.com" not in x and "gitlab.com" not in x) for x in (l if l is not None else []))))
b["n_try_links"] = b.try_links.apply(lambda l: len(l) if l is not None else 0)
b["n_tech"] = b.built_with.apply(lambda l: len(l) if l is not None else 0)
b["has_inspiration_head"] = b.headings.apply(lambda l: int(any("inspir" in h.lower() for h in (l if l is not None else []))))
b["has_challenges_head"] = b.headings.apply(lambda l: int(any("challeng" in h.lower() for h in (l if l is not None else []))))
b["video_min_clip"] = b.video_min.clip(upper=10)
b["video_platform"] = b.video_platform.fillna("none").astype("category")

# Within-hackathon percentiles remove the "big hackathon vs small hackathon" confound.
REL = ["team_size", "n_tech", "story_words", "n_h2", "n_img", "n_code", "n_links", "tagline_len"]
for c in REL:
    b[c + "_pct"] = b.groupby("hackathon_id")[c].rank(pct=True)

# Top tech tags as multi-hot columns (name-normalized by the parser).
from collections import Counter
cnt = Counter(t for l in b.built_with if l is not None for t in l)
TOP = [t for t, _ in cnt.most_common(120)]
tags = pd.DataFrame({f"tech_{t}": b.built_with.apply(lambda l, t=t: int(l is not None and t in l)) for t in TOP}, index=b.index)
b = pd.concat([b, tags], axis=1)
b.to_parquet(OUT / "gold.parquet")

# %% 7. Inference: conditional logit (hackathon fixed effects). Uniform within-hackathon sampling of
#      non-winners keeps odds ratios valid here, which a pooled logit would not guarantee.
from statsmodels.discrete.conditional_models import ConditionalLogit
CORE = ["team_size", "n_tech", "story_words", "n_img", "n_code", "n_links", "has_video", "has_github", "has_demo",
        "tagline_len", "has_inspiration_head", "has_challenges_head"]
d = b.dropna(subset=CORE + ["y"]).copy()
grp = d.groupby("hackathon_id").y.agg(["sum", "count"])
d = d[d.hackathon_id.isin(grp[(grp["sum"] > 0) & (grp["sum"] < grp["count"])].index)]   # groups with variation
X = d[CORE].astype(float)
X = (X - X.mean()) / X.std().replace(0, 1)
X["story_words"] = np.log1p(d.story_words.astype(float)); X["story_words"] = (X.story_words - X.story_words.mean()) / X.story_words.std()
res = ConditionalLogit(d.y.values, X.values, groups=d.hackathon_id.values).fit(method="bfgs", maxiter=300, disp=False)
ci = np.exp(res.conf_int())
odds = pd.DataFrame({"feature": CORE, "odds_ratio_per_sd": np.exp(res.params), "lo": ci[:, 0], "hi": ci[:, 1], "p": res.pvalues})
print(odds.sort_values("odds_ratio_per_sd", ascending=False).round(3).to_string(index=False))
odds.to_csv(OUT / "conditional_logit.csv", index=False)

# %% 8. Prediction + SHAP. GroupKFold by hackathon so no event leaks across folds.
import lightgbm as lgb, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
LEAKY = {"likes_post", "comments_post", "updates_post"}      # accrue after judging; never use as predictors
FEATS = (CORE + [c + "_pct" for c in REL] + ["registrations", "log_prize", "online", "n_sub", "days", "year",
         "n_h2", "n_photos", "video_min_clip", "video_platform"] + [f"tech_{t}" for t in TOP])
FEATS = list(dict.fromkeys(FEATS))
assert not LEAKY & set(FEATS)
Xg, yg, gg = b[FEATS], b.y.values, b.hackathon_id.values
oof = np.zeros(len(b))
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_estimators=400)
for tr, te in GroupKFold(5).split(Xg, yg, gg):
    m = lgb.LGBMClassifier(**params).fit(Xg.iloc[tr], yg[tr])
    oof[te] = m.predict_proba(Xg.iloc[te])[:, 1]
within = []
for _, g in b.assign(p=oof).groupby("hackathon_id"):
    if 0 < g.y.sum() < len(g): within.append(roc_auc_score(g.y, g.p))
print(f"pooled OOF AUC {roc_auc_score(yg, oof):.3f} | mean within-hackathon AUC {np.mean(within):.3f} over {len(within)} events")

final = lgb.LGBMClassifier(**params).fit(Xg, yg)
sample = Xg.sample(min(20000, len(Xg)), random_state=0)
sv = shap.TreeExplainer(final).shap_values(sample)
sv = sv[1] if isinstance(sv, list) else sv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
shap.summary_plot(sv, sample, max_display=25, show=False); plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=150); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False)
print(imp.head(15).round(4))

# %% 9. Text: embeddings + topics, then win-rate lift per topic with confidence intervals.
if RUN_TEXT:
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from statsmodels.stats.proportion import proportion_confint
    docs = (b.tagline.fillna("") + ". " + b.story_text.fillna("").str.slice(0, 1000)).tolist()
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, batch_size=256, show_progress_bar=True)
    np.save(OUT / "emb.npy", emb.astype("float16"))
    tm = BERTopic(min_topic_size=60, calculate_probabilities=False, verbose=False)
    b["topic"] = tm.fit_transform(docs, emb)[0]
    base_rate = b.y.mean()
    t = b[b.topic >= 0].groupby("topic").y.agg(["sum", "count"])
    t["rate"] = t["sum"] / t["count"]
    t[["lo", "hi"]] = [proportion_confint(s, n, method="wilson") for s, n in zip(t["sum"], t["count"])]
    t["lift"] = t.rate / base_rate
    t["name"] = [tm.get_topic(i)[0][0] + "/" + tm.get_topic(i)[1][0] for i in t.index]
    print(t[t["count"] >= 100].sort_values("lift", ascending=False).head(15).round(3))
    t.to_csv(OUT / "topic_lift.csv")

# %% 10. Schema evolution: re-parse stored raw HTML with a newer parser, no re-crawling.
# Clone the repo, bump PARSER_VERSION in pipeline.py, then:
#   from pipeline import parse_project
#   for f in sorted(glob.glob(f"{DATA}/projects_raw/*.parquet")):   # pull these first, they are the big files
#       t = pq.read_table(f).to_pylist()
#       rows = [dict(parse_project(gzip.decompress(r["html_gz"]).decode()), slug=r["slug"]) for r in t]
#       write rows with pa.Table.from_pylist(rows, schema=pipeline.SCHEMAS["projects"]) to data/projects/ and upload
# The view in cell 3 already prefers the highest parser_version per slug.
```


## File 6: tests/test_pipeline.py

### `tests/test_pipeline.py`

<!-- FILE: tests/test_pipeline.py sha256=f9339c710eda52fbcbe8ed7a4400dc1375f2c7b2c7d72349f13ee4fe40c76368 -->
```python
"""Offline checks: parsers on fixture HTML, plus discover -> gallery -> detail end to end
in --local-only mode with a fake HTTP client. No network needed."""
import gzip, json, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as p

GALLERY = """<html><body><nav><a href="https://devpost.com/software">Projects</a></nav>
<div class="gallery"><ul>
<li class="gallery-item"><a href="https://devpost.com/software/alpha"><h5>Alpha</h5></a>
 <span class="winner">Winner</span><a href="https://devpost.com/ann">Ann</a><a href="https://devpost.com/bob">Bob</a> 3 1</li>
<li class="gallery-item"><a href="https://devpost.com/software/beta"><h5>Beta</h5></a>
 <a href="https://devpost.com/cy">Cy</a> 2 0</li>
</ul></div><p><b>1</b> \u2013 <b>2</b> of <b>2</b></p></body></html>"""

PROJECT = """<html><head><meta property="og:title" content="Alpha"><meta property="og:description" content="Does a thing">
<meta property="og:image" content="https://x/y.png"></head><body><h1>Alpha</h1>
<a href="/login">Like <span>3</span></a><a href="#updates">Comment 1</a>
<div id="app-details-left"><h2>Inspiration</h2><p>We were inspired by lots of words here.</p><h3>Sub</h3><img src="a.png">
<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>
<h2>Built With</h2><ul><li><a href="https://devpost.com/software/built-with/python">python</a></li>
<li><a href="https://devpost.com/software/built-with/html5">html5</a></li></ul>
<h2>Try it out</h2><ul><li><a href="https://github.com/a/b">github.com</a></li></ul></div>
<aside><h4>Submitted to</h4><a href="https://cmu-hackathon.devpost.com/">Central Hacks</a>
<h4>Created by</h4><ul><li><a href="https://devpost.com/ann">Ann</a></li><li><a href="https://devpost.com/bob">Bob</a></li></ul></aside>
<div><span class="winner label">Winner Best Hack</span></div>
<p><a href="https://devpost.com/software/alpha/updates/1">Ann</a> started this project \u2014 Apr 06, 2024 09:30 AM EDT</p>
</body></html>"""

def test_gallery():
    cards, exp = p.parse_gallery(GALLERY)
    assert exp == 2 and [c["slug"] for c in cards] == ["alpha", "beta"]
    a, b = cards
    assert a["is_winner"] and not b["is_winner"]
    assert (a["likes"], a["comments"]) == (3, 1) and a["profile_links"] == 2

def test_project():
    r = p.parse_project(PROJECT)
    assert r["title"] == "Alpha" and r["tagline"] == "Does a thing"
    assert r["built_with"] == ["python", "html5"]
    assert r["video_platform"] == "youtube" and r["video_id"] == "dQw4w9WgXcQ"
    assert r["team_size"] == 2 and r["team_handles"] == ["ann", "bob"]
    assert r["likes"] == 3 and r["comments"] == 1 and r["n_updates"] == 1
    assert r["first_update_text"] == "Apr 06, 2024 09:30 AM EDT"
    assert r["try_links"] == ["https://github.com/a/b"]
    assert r["submitted_to"] == ["https://cmu-hackathon.devpost.com/"]
    assert r["is_winner_page"] and r["story_root_found"] and r["n_h2"] == 1
    assert "Built With" not in r["story_text"] and "inspired" in r["story_text"]

class Fake:
    def get(self, url, params=None):
        if "api/hackathons" in url:
            pg = params["page"]
            hs = [] if pg > 2 else [{"id": pg * 10 + i, "title": f"H{pg}{i}", "url": f"https://h{pg}{i}.devpost.com/",
                  "submission_gallery_url": f"https://h{pg}{i}.devpost.com/project-gallery", "open_state": "ended",
                  "submission_period_dates": "Sep 25 - 30, 2026", "themes": [{"id": 1, "name": "Web"}],
                  "prize_amount": "$<span data-currency-value>1,500</span>", "prizes_counts": {"cash": 1, "other": 0},
                  "registrations_count": 50, "organization_name": "o", "winners_announced": True, "invite_only": False,
                  "managed_by_devpost_badge": False, "displayed_location": {"icon": "globe", "location": "Online"}}
                 for i in range(2)]
            return "ok", 200, json.dumps({"hackathons": hs, "meta": {"total_count": 4, "per_page": 2}})
        if "project-gallery" in url:
            return "ok", 200, GALLERY
        return "ok", 200, PROJECT

def test_end_to_end(tmp_path):
    a = types.SimpleNamespace(shard=0, nshards=1, refresh=False, min_registrations=0, seed=1,
                              nonwinners_per_hack=1, max_gallery_pages=5, max_attempts=5)
    st = p.Store(tmp_path, None, None, "t", True)
    bud = p.Budget(5)
    p.cmd_discover(a, st, Fake(), bud)
    p.cmd_discover(a, st, Fake(), bud)               # resume: must not duplicate or loop
    assert len(st.read("hackathons")) == 4
    p.cmd_run(a, st, Fake(), bud)
    cards = st.read("cards")
    assert len(cards) == 8 and sum(c["detail_selected"] for c in cards) == 8
    p.cmd_run(a, st, Fake(), bud)                    # resume: nothing left to do
    ok, _ = st.done("detail")
    assert set(ok) == {"alpha", "beta"}
    rows = st.read("projects")
    assert len(rows) == 2 and gzip.decompress(st.read("projects_raw")[0]["html_gz"]).startswith(b"<html")

def test_pacer_and_shards():
    assert 0 <= p.shard_of("abc", 5) < 5
    pc = p.Pacer(1.0); pc.mult = 4; pc.ok(); assert pc.mult < 4

def test_winner_css_override(monkeypatch):
    html = GALLERY.replace('<span class="winner">Winner</span>', '<i class="trophy-x">x</i>')
    assert not p.parse_gallery(html)[0][0]["is_winner"]
    monkeypatch.setattr(p, "WINNER_CSS", ".trophy-x")
    assert p.parse_gallery(html)[0][0]["is_winner"]

def test_status_runs(tmp_path, capsys):
    a = types.SimpleNamespace(shard=0, nshards=1, refresh=False, min_registrations=0, seed=1,
                              nonwinners_per_hack=1, max_gallery_pages=5, max_attempts=5)
    st = p.Store(tmp_path, None, None, "t", True)
    bud = p.Budget(5)
    p.cmd_discover(a, st, Fake(), bud); p.cmd_run(a, st, Fake(), bud)
    p.cmd_status(a, st)
    out = capsys.readouterr().out
    assert "discover" in out and "gallery" in out and "detail" in out and "done 2" in out


# ----------------------------------------------------------------------------- rate limiter, backoff, breaker
class Resp:
    def __init__(self, code, text="", headers=None):
        self.status_code, self.text, self.headers = code, text, headers or {}


class FakeSession:
    """Stands in for the curl_cffi session: replays a scripted list of responses."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def get(self, url, params=None, allow_redirects=True):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_client(monkeypatch, script, breaker=8):
    sleeps = []
    monkeypatch.setattr(p.time, "sleep", lambda s: sleeps.append(s))
    cl = p.Client(0.01, breaker=breaker)
    cl.s = FakeSession(script)
    monkeypatch.setattr(cl.pacer, "wait", lambda: None)
    return cl, sleeps


def test_ok_and_gone(monkeypatch):
    cl, _ = make_client(monkeypatch, [Resp(200, "<html>fine</html>")])
    assert cl.get("u") == ("ok", 200, "<html>fine</html>")
    cl, _ = make_client(monkeypatch, [Resp(404)])
    assert cl.get("u") == ("gone", 404, None)


def test_challenge_page_counts_as_block(monkeypatch):
    cl, _ = make_client(monkeypatch, [Resp(200, "<title>Just a moment...</title>")], breaker=99)
    kind, code, text = cl.get("u")
    assert kind == "error"          # 200 with a challenge body: counted as blocks, reported as retryable error
    assert cl.consec == 4 and text is None
    cl, _ = make_client(monkeypatch, [Resp(403, "", {"cf-mitigated": "challenge"})], breaker=99)
    assert cl.get("u")[0] == "blocked"


def test_backoff_doubles_and_caps_and_decays(monkeypatch):
    cl, sleeps = make_client(monkeypatch, [Resp(429, "", {"retry-after": "7"})], breaker=99)
    assert cl.get("u") == ("blocked", 429, None)
    assert cl.pacer.mult == 16.0                      # 2x per block, capped at 16x after 4 blocks
    assert 7 in sleeps                                # Retry-After honored
    assert [s for s in sleeps if s >= 10][:4] == [10, 20, 40, 80]   # 10 * 2**attempt
    cl, _ = make_client(monkeypatch, [Resp(200, "ok")])
    cl.pacer.mult = 8.0
    cl.get("u")
    assert cl.pacer.mult == 7.2                       # decays by 10% per success
    cl.pacer.mult = 1.05
    cl.get("u")
    assert cl.pacer.mult == 1.0                       # never below the base rate


def test_circuit_breaker_trips_on_8th_consecutive_block(monkeypatch):
    cl, _ = make_client(monkeypatch, [Resp(429)], breaker=8)
    assert cl.get("u")[0] == "blocked" and cl.consec == 4        # 4 attempts per call, not yet tripped
    with __import__("pytest").raises(p.Blocked):
        cl.get("u")                                               # blocks 5..8 -> trips
    assert cl.consec == 8


def test_success_resets_the_breaker_counter(monkeypatch):
    cl, _ = make_client(monkeypatch, [Resp(429), Resp(429), Resp(200, "ok")], breaker=8)
    assert cl.get("u")[0] == "ok" and cl.consec == 0


def test_5xx_and_network_errors_retry_then_report_error(monkeypatch):
    cl, sleeps = make_client(monkeypatch, [Resp(500)])
    assert cl.get("u") == ("error", 500, None) and cl.s.calls == 4
    cl, _ = make_client(monkeypatch, [ConnectionError("reset"), Resp(200, "ok")])
    assert cl.get("u")[0] == "ok" and cl.s.calls == 2


def test_pacer_spacing_is_jittered_within_bounds(monkeypatch):
    waits = []
    monkeypatch.setattr(p.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(p.time, "monotonic", lambda: 100.0)
    pc = p.Pacer(2.0, jitter=0.4)
    pc.last = 100.0
    for _ in range(50):
        pc.wait()
    assert waits and all(1.2 - 1e-9 <= w <= 2.8 + 1e-9 for w in waits)   # 2.0 s +/- 40%


# ----------------------------------------------------------------------------- resumption and state
class CountingFake(Fake):
    def __init__(self):
        self.detail_calls = []

    def get(self, url, params=None):
        if "/software/" in url and "project-gallery" not in url:
            self.detail_calls.append(url.rsplit("/", 1)[-1])
        return super().get(url, params)


def ns(**kw):
    base = dict(shard=0, nshards=1, refresh=False, min_registrations=0, seed=1, nonwinners_per_hack=1,
                max_gallery_pages=5, max_attempts=5)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_poison_pill_and_retry_override(tmp_path):
    st = p.Store(tmp_path, None, None, "t", True)
    fake, bud = CountingFake(), p.Budget(5)
    p.cmd_discover(ns(), st, fake, bud)
    p.cmd_gallery(ns(), st, fake, bud)
    for _ in range(5):
        st.log("detail", "alpha", "error", 500)
    st.flush()
    p.cmd_detail(ns(), st, fake, bud)
    assert set(fake.detail_calls) == {"beta"}                  # alpha failed 5 times: skipped
    fake.detail_calls.clear()
    p.cmd_detail(ns(max_attempts=6), st, fake, bud)
    assert fake.detail_calls == ["alpha"]                      # override retries only the poisoned key


def test_gone_is_final_but_blocked_is_retried(tmp_path):
    st = p.Store(tmp_path, None, None, "t", True)
    st.log("detail", "a", "gone"); st.log("detail", "b", "blocked"); st.log("detail", "c", "ok")
    st.flush()
    ok, bad = st.done("detail")
    assert set(ok) == {"a", "c"} and bad["b"] == 1


def test_blocked_mid_gallery_discards_partial_and_retries(tmp_path):
    class MidBlock(Fake):
        def get(self, url, params=None):
            if "project-gallery" in url:
                if params and params["page"] >= 2:
                    return "blocked", 429, None
                return "ok", 200, GALLERY.replace("of <b>2</b>", "of <b>4</b>")   # promises 4, shows 2
            return super().get(url, params)
    st = p.Store(tmp_path, None, None, "t", True)
    bud = p.Budget(5)
    p.cmd_discover(ns(), st, MidBlock(), bud)
    p.cmd_gallery(ns(), st, MidBlock(), bud)
    assert st.read("cards") == []                               # nothing partial was written
    ok, bad = st.done("gallery")
    assert ok == {} and sum(bad.values()) == 4                  # 4 hackathons logged blocked, all retryable
    p.cmd_gallery(ns(), st, Fake(), bud)                        # healthy retry succeeds
    assert len(st.read("cards")) == 8


def test_sampling_keeps_all_winners_and_caps_non_winners():
    rows = [{"slug": f"s{i}", "is_winner": i < 3} for i in range(100)]
    p.pick_sample(rows, 40, seed=7)
    sel = [r for r in rows if r["detail_selected"]]
    assert len(sel) == 3 + 40 and all(r["detail_selected"] for r in rows if r["is_winner"])
    again = [dict(r) for r in rows]
    p.pick_sample(again, 40, seed=7)
    assert [r["detail_selected"] for r in rows] == [r["detail_selected"] for r in again]   # deterministic
    small = [{"slug": f"t{i}", "is_winner": i == 0} for i in range(10)]
    p.pick_sample(small, 40, seed=7)
    assert all(r["detail_selected"] for r in small)                                         # under the cap: keep all


def test_shards_are_disjoint_and_cover_everything():
    keys = [f"slug-{i}" for i in range(2000)]
    owners = [[k for k in keys if p.shard_of(k, 5) == s] for s in range(5)]
    assert sum(len(o) for o in owners) == len(keys)
    assert len(set().union(*map(set, owners))) == len(keys)
    assert min(len(o) for o in owners) > 300                                                # roughly balanced
```


## Supporting files

### `tests/make_synthetic.py`

<!-- FILE: tests/make_synthetic.py sha256=e819f03a0759c4dc6ef035c71be852dbcaea737f2e327a72e3461072d05abe0c -->
```python
import random, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pyarrow as pa, pyarrow.parquet as pq, pipeline as p
random.seed(1)
root = pathlib.Path("/tmp/synth/data")
for t in ("hackathons","cards","projects"): (root/t).mkdir(parents=True, exist_ok=True)
H, C, P = [], [], []
periods = ["Sep 25 - 30, 2025","Aug 27 - Sep 26, 2025","Sep 28, 2024","Dec 28 - Jan 3, 2025"]
for h in range(1, 121):
    H.append(dict(hackathon_id=h, title=f"H{h}", url=f"https://h{h}.devpost.com/", gallery_url="x", open_state="ended",
      period_text=random.choice(periods), themes=["Web"], prize_text="$1,000", prize_value=1000.0, prize_currency="$",
      prizes_cash=1, prizes_other=0, registrations=random.randint(20,500), organization="o", winners_announced=True,
      invite_only=False, managed_by_devpost=False, location="Online", is_online=True, fetched_at="2026-10-01T00:00:00Z"))
    n = 30
    for k in range(n):
        slug = f"p{h}-{k}"; team = random.randint(1,5); words = random.randint(80,900)
        score = 0.35*team/5 + 0.5*min(words,600)/600 + random.random()*0.9
        win = k < 3 and score > 0.5 or score > 1.55
        C.append(dict(hackathon_id=h, slug=slug, is_winner=win, winner_text="Winner" if win else None, likes=1, comments=0,
          profile_links=team, card_text="", page=1, detail_selected=True, n_cards_in_hackathon=n, n_winners_in_hackathon=0, fetched_at="2026-10-01T00:00:00Z"))
        P.append(dict(slug=slug, parser_version=1, fetched_at="2026-10-01T00:00:00Z", title="t"*random.randint(5,20),
          tagline="a tagline about "+random.choice(["health","games","finance"]), og_image=None,
          built_with=random.sample(["python","react","node.js","openai","flutter","firebase"], random.randint(1,4)),
          story_text="story "*words, story_chars=words*6, story_words=words, story_root_found=True,
          headings=["Inspiration","Challenges we ran into"][:random.randint(0,2)], n_h2=random.randint(0,6), n_h3=0,
          n_img=random.randint(0,6), n_li=2, n_code=random.randint(0,2), n_links=random.randint(0,5), n_bold=1,
          n_photos=random.randint(0,5), video_url="u" if random.random()<.7 else None, video_platform="youtube", video_id=f"v{h}{k}"[:11].ljust(11,"x"),
          video_source="iframe", team_handles=[], team_size=team, likes=3, comments=1, n_updates=1, first_update_text=None,
          try_links=["https://github.com/a/b"] if random.random()<.8 else [], submitted_to=[], winner_texts=[], is_winner_page=win))
for t, rows in (("hackathons",H),("cards",C),("projects",P)):
    pq.write_table(pa.Table.from_pylist(rows, schema=p.SCHEMAS[t]), root/t/"a.parquet")
print("synthetic written", len(H), len(C), len(P))
```

### `requirements.txt`

<!-- FILE: requirements.txt sha256=531e1cbc6674f317ef8d69ab79d7eeaaa7c5f0b9c47c0fcb4fa5b873b3e674c5 -->
```text
curl_cffi>=0.7
selectolax>=0.3
pyarrow>=15
huggingface_hub>=0.24
```

### `.gitignore`

<!-- FILE: .gitignore sha256=ab4de792979c08d5bd4bc382167b2e64a45bca071ebeba5fcb9a7748c4278782 -->
```
# secrets
.env
*.env
.env.*
moneyball.env
hf_token*
# python
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
# scraped or generated data must never reach the public repo
*.parquet
*.npy
*.html
*.gz
probe/
out/
hf/
hf_home/
work/
data/
ledger/
colab/out/
*.zip
# notebooks converted from colab/analysis.py may hold outputs
*.ipynb_checkpoints/
```

### `PLAN.md`

<!-- FILE: PLAN.md sha256=b402a7add2edf6f2d480b2587963215482e392dac5cb2e939ff0ff68c215fbb8 -->
````markdown
# Moneyballing Hackathons: architecture and runbook

Goal: a one-off, $0, no-credit-card pipeline that pulls completed Devpost hackathons and their galleries, labels winners, and produces a clean table for statistical modeling in Colab.

Everything below was checked against Devpost on 1 Oct 2026 unless it is listed under "Not verified".

## 0. What I verified, and what I did not

Verified live:
- `https://devpost.com/api/hackathons?status[]=ended&order_by=recently-added&page=N` returns JSON. 9 hackathons per page, `meta.total_count` was 13,809 (about 1,535 pages). Each item has: `id`, `title`, `url`, `submission_gallery_url`, `submission_period_dates`, `themes[]`, `prize_amount` (HTML-wrapped), `prizes_counts{cash,other}`, `registrations_count`, `organization_name`, `winners_announced`, `invite_only`, `displayed_location`, `managed_by_devpost_badge`.
- `robots.txt` has no Disallow for generic agents (it blocks a few named bots only).
- Gallery pages (`<sub>.devpost.com/project-gallery`) are server-rendered and link every project as `devpost.com/software/<slug>`. Each card ends with likes then comments (checked against a project page: card "3 1" matched "Like 3", "Comment 1"). The page shows "1 - 11 of 11", which the scraper uses as a completeness check.
- Project pages are server-rendered. `og:description` is the tagline. Built-with tags are links to `/software/built-with/<tag>`. Sections "Built With", "Try it out", "Submitted to", "Created by", "Updates" exist. The first update line carries the start timestamp.
- The fetch tool I used from my sandbox was not challenged on any of these pages. That says little about your IPs.

Not verified (the code is defensive, and `probe` exists to settle these in two minutes):
- The exact HTML of the winner badge on gallery cards and project pages. The parser looks for any element whose class contains "winner", an image alt with "winner", or an element whose text is exactly "Winner". Run `probe` on a gallery with known winners before the big run.
- Gallery pagination parameter (`?page=N`) and page size. The loop stops on a page with no new slugs, so a wrong guess shows up as a low card count against the "of N" total, which is logged per hackathon.
- Whether the API allows deep paging to page 1,500 without throttling.
- Whether your Raspberry Pi's home IP and GitHub's runner IPs get challenged. Plain datacenter ranges are more likely to be flagged than a residential IP, so the Pi is the safer worker. The `canary` command tells you in seconds.
- Devpost's Terms of Service. I could not confirm what they say about automated access. Read them before you start. See section 7.

## 1. Architecture and division of labor

```
        discover (once)             gallery -> detail (nightly bursts)               analysis
   +--------------------+       +----------------------------------------+     +-------------------+
   | GitHub Actions     |       | GitHub Actions: 4 matrix shards (0..3) |     | Google Colab      |
   | 1 job, ~1 hour     |       | Raspberry Pi 5: shard 4, 3h at night   |     | DuckDB, LightGBM, |
   | hackathon list     |       | each: ~0.4 req/s, one request at a time|     | SHAP, BERTopic    |
   +---------+----------+       +-------------------+--------------------+     +---------+---------+
             |                                      |                                    ^
             |             one commit per flush     v                                    |
             +----------------------> Hugging Face dataset repo (private) <--------------+
                                      data/<table>/*.parquet   ledger/<stage>/*.parquet
```

| Where | Does | Why there |
|---|---|---|
| GitHub Actions | Phase 1 once; then four parallel shards of gallery + detail, one 5.5 h job per shard per day | Free minutes on a public repo, four different IPs per day, no hardware to babysit |
| Pi 5 | Shard 4 of 5, 3 h nightly via systemd timer, working in `/dev/shm` | Residential IP is the least likely to be challenged. Zero SD card writes except code and the venv |
| Colab | Everything analytical: joins, features, video durations, models, embeddings, re-parsing raw HTML | Open internet, T4 for embeddings, nothing you scrape ever has to fit on the Pi |

Why sharding by hash, not a queue server: a worker computes `sha1(key) % 5` and only touches its own keys, so there is no coordinator, no locks, and no double fetching. Any worker can take over a dead worker's shard by passing `--shard`.

### Storage: Hugging Face dataset repo, Parquet, private

Chosen over the alternatives:
- Supabase free (500 MB) cannot hold the writeup text, let alone raw HTML.
- Mongo Atlas M0 (512 MB) has the same problem and needs a driver on every worker.
- GitHub releases need a public repo, which would publish scraped usernames. Keep as an emergency fallback only.
- HF datasets: free, no card, large quota, and DuckDB/pandas read Parquet from it directly.

Layout (a bronze/silver idea without the jargon):

| Path | Content | Mutable? |
|---|---|---|
| `data/hackathons/` | one row per hackathon from the API | append-only |
| `data/cards/` | one row per (hackathon, project): winner flag, likes, comments, `detail_selected` | append-only |
| `data/projects_raw/` | gzipped raw HTML of every detailed project page | append-only, the source of truth |
| `data/projects/` | parsed fields, tagged with `parser_version` | append-only, regenerated from raw |
| `ledger/<stage>/` | tiny files: `(stage, key, status, note, ts, worker)` | append-only |

Schema evolution: parquet files never change. When you want a new feature (say, "has a table in the writeup"), bump `PARSER_VERSION`, re-parse `projects_raw` in Colab with no network, write a new `projects` file. DuckDB's `read_parquet(..., union_by_name=true)` fills old files with NULL for new columns, and the view keeps the highest `parser_version` per slug. You never re-crawl for a parsing mistake.

Write pattern: each worker buffers in memory and ships data plus ledger in one `create_commit` every 300 rows or 10 minutes. That keeps commit count low (HF throttles commits per repo per hour, so do not flush per row) and makes state and data land together or not at all. Run `pipeline.py compact` occasionally, with no worker active, to merge small ledger files.

## 2. Anti-bot and throttling

Be honest about the goal: stay under the threshold that triggers protection, and stop when it triggers anyway. Do not try to solve interactive challenges.

Layered, cheapest first:
1. **Do not look like a script.** `curl_cffi` with `impersonate="chrome"` sends a real Chrome TLS and HTTP/2 fingerprint, which is what most bot filters actually key on. Python `requests` is the thing that gets blocked. One session per worker, so cookies persist.
2. **Be slow.** One request in flight per worker. Base gap 2.0 s with plus or minus 40% jitter (Pi: 2.5 s). Five workers is about 2 requests per second across five IPs, and under 0.5 per second from any one IP.
3. **Adapt.** On 403, 429, 503, a `cf-mitigated: challenge` header, or a "Just a moment..." body, the pacing multiplier doubles (cap 16x), `Retry-After` is honored, and the request retries with exponential sleep. Success decays the multiplier by 10% each time.
4. **Circuit breaker.** 8 consecutive blocks raise `Blocked`; the worker flushes, prints a warning, exits 0, and tries again at the next scheduled run, from a different Actions IP. Do not loop through blocks.
5. **IP diversity for free.** Actions gives a different runner IP every run. The Pi is the residential fallback. Run `canary` first from each. If Actions ranges are blocked, drop to Pi-only (set `--nshards 1`, a longer window, and accept it takes weeks).
6. **Last resort, still free:** Playwright on the Pi for only the pages that fail. It is slow and RAM-heavy, so do not make it the default.

Pacing numbers: 0.4 req/s is about 1,440 requests per hour. Four Actions shards at 5.5 h plus the Pi at 3 h is about 40,000 requests per day. Phase 1 is about 1,540 requests (roughly 50 minutes). The total for phases 2 and 3 is `hackathons_with_winners x (gallery pages + winners + up to 40 sampled non-winners)`. Compute it from the `hackathons` table after phase 1 rather than trusting a guess; for example 4,000 hackathons averaging 30 requests each is 120,000 requests, which is about three days.

## 3. Ingestion flow

State rule for every stage: a key is done when the ledger has `ok`, `empty` or `gone` for it. `blocked` and `error` stay retryable, up to 5 attempts, after which the key is skipped (poison pill). Nothing is ever deleted, so a crash at any point loses at most one unflushed batch.

### Phase 1, discover (`pipeline.py discover`, once, from Actions)
Walk the API pages until a page comes back empty. Ledger key `page:N`. Write all fields into `hackathons`. New hackathons added while you crawl can shift the "recently-added" ordering, so dedupe by id (the Colab views do) and re-run with `--refresh` at the end for a cheap second pass.

Filter at selection time, not at fetch time: the gallery stage only takes hackathons with `winners_announced = true`. There are junk entries ("TEST ROWDYHACKS"), so also use `--min-registrations` (try 20) to skip dead events.

### Phase 2, gallery and card extraction (`gallery`)
Per hackathon: fetch `project-gallery?page=1..N` until no new slugs or the "of N" total is reached. A card is the smallest ancestor that contains exactly one project link, which survives CSS class renames. Extract slug, winner flag, likes, comments. Writes are atomic per hackathon: a failure mid-gallery discards the partial set and the hackathon retries later.

Sampling is decided here, once, deterministically: every winner plus up to 40 non-winners per hackathon, ordered by `sha1(seed:slug)`. This is the key cost lever. You get the label for every project from the gallery, and pay for the expensive detail page only on the sample. It is also statistically fine: uniform sampling of non-winners within each hackathon preserves odds ratios in a conditional logit (section 5).

### Phase 3, detail (`detail`)
For each selected slug in this shard, in hash-random order (so a half-finished run is still an unbiased sample, not a pile of winners), fetch `devpost.com/software/<slug>`, store the gzipped HTML in `projects_raw`, and parse into `projects`:
- title, tagline (`og:description`), built-with tags
- writeup text, character and word counts, h2/h3/image/list/code/link counts, heading texts
- video URL, platform, id (iframe first, then links)
- team handles and size, likes, comments, update count, first update timestamp
- try-it-out links, "submitted to" subdomains, winner badge texts

`run` chains gallery then detail inside one time budget. Both stages end gracefully when the budget runs out.

Resume test (done offline): the end-to-end test runs discover, gallery and detail against a fake client, then runs them again and confirms nothing is refetched and nothing loops.

## 4. Target schema

Tables are the ones in section 1; the gold table (built in Colab, one row per hackathon-project) holds:

| Group | Columns |
|---|---|
| Keys and label | `hackathon_id`, `slug`, `y` (winner flag), `winner_texts` (prize names, for finer labels) |
| Event context | `registrations`, `log_prize`, `online`, `n_sub` (gallery size), `days`, `year`, `themes[]` |
| Team | `team_size` |
| Tech | `built_with[]`, `n_tech`, top-120 tag multi-hot |
| Video | `has_video`, `video_platform`, `video_min` (from YouTube API or Vimeo oEmbed in Colab) |
| Writeup | `story_words`, `n_h2`, `n_img`, `n_li`, `n_code`, `n_links`, section-heading flags (inspiration, challenges), `tagline_len`, `title_len` |
| Links | `has_github`, `has_demo`, `n_try_links` |
| Relative | within-hackathon percentile of each count feature (removes big-event vs small-event confounding) |
| Post-outcome (never predictors) | `likes_post`, `comments_post`, `updates_post` |

Why video length is not in the scraper: the page only holds an embed, not a duration. The YouTube Data API returns it for 50 ids per quota unit and 10,000 units per day are free with just a Google account, so Colab can enrich 100k videos in a few runs.

## 5. Colab analytics roadmap (`colab/analysis.py`)

1. **Load**: `snapshot_download` of everything except raw HTML; DuckDB views with `union_by_name`.
2. **Gold table**: SQL joins plus pandas for list features. Percentiles per hackathon.
3. **Inference first**: conditional logit with hackathon fixed effects (statsmodels `ConditionalLogit`). Output is an odds ratio per standard deviation with CI for each feature. This answers "what correlates with winning inside the same event", which is the real question.
4. **Prediction second**: LightGBM, 5-fold `GroupKFold` by hackathon. Report both pooled AUC and mean within-hackathon AUC. The second one is the honest number.
5. **SHAP**: TreeExplainer on up to 20k rows; summary plot and top features. Use it to find non-linear shapes (for example writeup length with diminishing returns) and interactions, then confirm them in the logit.
6. **Text**: `all-MiniLM-L6-v2` embeddings of tagline plus the first 1,000 characters (T4, a few minutes for 100k), BERTopic with `min_topic_size=60`, then win rate per topic with Wilson intervals and lift over base rate. Only trust topics with at least 100 projects.
7. **Re-parse** raw HTML with a new `PARSER_VERSION` whenever you want a new feature.

I ran this script end to end on synthetic data with planted effects (writeup length and team size): the conditional logit recovered both as the top odds ratios and SHAP ranked the same two first. The text cell and the YouTube enrichment were not run, because they need model downloads and a key.

Method cautions that will bite if ignored:
- **Leakage**: likes, comments and updates keep accruing after judging, and winning causes likes. They are excluded from predictors by an assertion in the script.
- **"Winner" is not one thing**: sponsor tracks give many winners per event. Use `winner_texts` to build a "grand prize" label as a robustness check.
- **Survivorship**: you only see projects that submitted and were not deleted.
- **Many hackathons are tiny or beginner events**: weight or stratify by event size before generalizing.
- **Multiple comparisons**: 120 tech tags and many topics. Use the CI, not the p-value, and hold out a set of hackathons you never looked at.

## 6. Runbook

1. Create a private HF dataset repo and a fine-grained token with write access to that repo only.
2. Push this folder to a public GitHub repo. Add secret `HF_TOKEN` and repository variable `HF_REPO`.
3. Locally: `pip install -r requirements.txt pytest && pytest -q tests` (offline, 18 tests).
4. `python pipeline.py probe https://<a-known-winning-hackathon>.devpost.com/project-gallery` and one `https://devpost.com/software/<slug>` of a winner. Check `winners=` is nonzero and `winner_texts` is filled. If not, adjust the winner check in `parse_gallery` and `parse_project`. This is the one thing that must be right before you spend days crawling.
5. Actions, "ingest", run workflow with `stage=discover`.
6. Pi: copy to `/opt/moneyball`, create the venv, drop the two files from `pi/` into `/etc/systemd/system/`, create `/etc/moneyball.env`, `systemctl enable --now moneyball.timer`.
7. Let the nightly cron run. Watch the ledger counts in Colab (`SELECT status, count(*) ... GROUP BY 1`). Run `compact` weekly.
8. When `detail` has nothing left (the log prints "0 projects to do" for every shard), run `colab/analysis.py`.

## 7. Risks to settle before the long run

- **Terms of Service**: unverified. Public pages, a robots.txt that allows access, low request rate and no login are the favorable facts. Whether the ToS forbids automated collection is a separate question, so read it, and consider emailing Devpost for research permission. If they say no, stop.
- **Personal data**: handles and names are in the data. Keep the dataset private, hash handles in the gold table, never publish raw rows, and publish only aggregates and model outputs.
- **Secrets**: the Actions repo is public, so the HF token lives only in repository secrets, is never printed, and does not reach fork pull requests.
- **Being a good neighbor**: the rates above are deliberately low. If you see repeated 429s, lower the rate rather than adding workers.
- **Untested here**: the Hugging Face upload and download paths (no credentials or network in my sandbox) and the live Devpost requests. Everything else, including resume behavior, was exercised offline.
````

### `RUNBOOK.md`

<!-- FILE: RUNBOOK.md sha256=4b207e74c762404ff4496c53a8f19d9ed9295327382abd85cf707ae4afe8c46d -->
````markdown
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
````

### `tools/build_export.py`

<!-- FILE: tools/build_export.py sha256=ff8760719f636cee8a3ab79975a723879b02cf7adfe9710c939dc980e4637a0f -->
```python
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
```

### `tools/restore_from_export.py`

<!-- FILE: tools/restore_from_export.py sha256=ee2d04ea9eb7e14f8a4cf91fae2876f6e93179784a679e23ced900423a472403 -->
```python
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
```
