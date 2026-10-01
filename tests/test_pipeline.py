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
