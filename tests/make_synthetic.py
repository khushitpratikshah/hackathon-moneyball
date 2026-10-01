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
