import json

recs = [json.loads(l) for l in open("paper/meld_test_scores.jsonl", encoding="utf-8") if l.strip()]

def am(d):  # argmax over a score dict
    return max(d, key=lambda k: d[k])

def analyze(rows, title):
    n = len(rows)
    both = vonly = tonly = neither = agree = agree_ok = 0
    for r in rows:
        g, v, t = r["gold"], am(r["voice"]), am(r["text_scores"])
        vc, tc = v == g, t == g
        both += vc and tc
        vonly += vc and not tc
        tonly += tc and not vc
        neither += not vc and not tc
        if v == t:
            agree += 1
            agree_ok += v == g
    va = sum(am(r["voice"]) == r["gold"] for r in rows) / n
    ta = sum(am(r["text_scores"]) == r["gold"] for r in rows) / n
    print(f"\n== {title} (n={n}) ==")
    print(f"  voice acc        : {va:6.2%}")
    print(f"  text acc         : {ta:6.2%}")
    print(f"  ORACLE (either)  : {(both+vonly+tonly)/n:6.2%}   <- fusion ceiling")
    print(f"  both right       : {both/n:6.2%}")
    print(f"  voice-only right : {vonly/n:6.2%}   <- voice's unique value")
    print(f"  text-only right  : {tonly/n:6.2%}")
    print(f"  neither          : {neither/n:6.2%}")
    print(f"  agree rate       : {agree/n:6.2%}  (acc | agree: {agree_ok/max(agree,1):6.2%})")

analyze(recs, "ALL")
for name, f in {"short<=3": lambda n: n <= 3,
                "mid 4-8": lambda n: 4 <= n <= 8,
                "long 9+": lambda n: n >= 9}.items():
    analyze([r for r in recs if f(r["n_words"])], name)