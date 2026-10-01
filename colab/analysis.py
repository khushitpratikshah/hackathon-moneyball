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
