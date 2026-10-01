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
