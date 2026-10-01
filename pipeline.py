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
