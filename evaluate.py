"""Validate with the provided sample questions.

questions.txt - one per line:   question | expected PDF pages | expected keywords (separated by ;)
  e.g.  How much power does iMac use in sleep at 115V? | 12 | 0.57W
Pages and keywords are optional. Then:  python evaluate.py [--rerank]
Reports retrieval hit (expected page among the sources) and answer hit (keywords found in the answer).
"""
import sys
import time

import config
from rag import RAG

rerank = "--rerank" in sys.argv
rows = []
for line in (config.BASE_DIR / "questions.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = [p.strip() for p in line.split("|")]
    pages = {int(x) for x in parts[1].replace(" ", "").split(",") if x} if len(parts) > 1 else set()
    kws = [k.strip().lower() for k in parts[2].split(";") if k.strip()] if len(parts) > 2 else []
    rows.append((parts[0], pages, kws))

rag = RAG()
out, r_hit, r_n, a_hit, a_n = ["# Evaluation results\n"], 0, 0, 0, 0
for q, pages, kws in rows:
    r = rag.answer(q, rerank=rerank)
    marks = []
    if pages:
        r_n += 1
        ok = bool(pages & set(r["retrieved_pages"]))
        r_hit += ok
        marks.append("retrieval HIT" if ok else f"retrieval MISS (expected {sorted(pages)})")
    if kws:
        a_n += 1
        ok = all(k in r["answer"].lower() for k in kws)
        a_hit += ok
        marks.append("answer HIT" if ok else f"answer MISS (missing {[k for k in kws if k not in r['answer'].lower()]})")
    types = ", ".join(f"p{s['page']}:{s['type']}" for s in r["sources"])
    print(f"\nQ: {q}\nA: {r['answer']}\n   sources: {types}\n   {' | '.join(marks)}")
    out.append(f"## {q}\n\n{r['answer']}\n\n*Sources: {types}*  \n*{' | '.join(marks)}*\n")
    time.sleep(4)
summary = []
if r_n: summary.append(f"Retrieval hit rate: {r_hit}/{r_n}")
if a_n: summary.append(f"Answer keyword hit rate: {a_hit}/{a_n}")
if summary:
    print("\n" + " | ".join(summary)); out.append("\n**" + " | ".join(summary) + "**")
(config.BASE_DIR / "results.md").write_text("\n".join(out), encoding="utf-8")
print("Saved results.md")
