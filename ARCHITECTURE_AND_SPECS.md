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
