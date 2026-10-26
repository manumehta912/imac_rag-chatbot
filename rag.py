"""Retrieval for mixed-content documents (text, tables, charts, footnotes).

Query -> optional rewrite of follow-ups
      -> hybrid retrieval: Gemini dense embeddings + BM25, fused with RRF
      -> content-type awareness: numeric / "table" / "chart" / "footnote" intent boosts the right chunk types;
         explicit "footnote 7", configuration names (two ports, 256GB ...) boost matching chunks
      -> optional content-type filter from the UI, optional LLM reranking (precision mode)
      -> context assembly: every retrieved chunk is shipped together with the footnotes it cites,
         tables as markdown + row facts, page + type labels for citations
"""
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
from google import genai
from google.genai import types

import config

STOP = set("a an and are as at be by for from has have in is it its of on or that the their this to was were which with will what how does do did about into than then these those who whom can could should would".split())

TYPE_GROUPS = {
    "tables": ["table"],
    "charts": ["chart", "vision"],
    "footnotes": ["footnote", "definition"],
    "text": ["text", "callout"],
}

SYSTEM_PROMPT = """You answer questions about Apple's "iMac Product Environmental Report" (October 2024).
Use ONLY the numbered context excerpts. Each excerpt header shows its page and content type (Text, Table, Chart, Footnote, Definition, Figure/table (vision)).
Rules:
- If the excerpts do not contain the answer, say you could not find it in the report. Never guess or use outside knowledge.
- Numbers from tables and charts must be quoted exactly, with units and the product configuration they belong to
  (the report mixes "iMac (two ports) 256GB" and "iMac (four ports) 512GB" - always say which one).
- Excerpts may be followed by "Footnotes cited here". Use them: mention qualifiers that change the meaning of a number
  (e.g. how a reduction is calculated, what a percentage is measured against, which configuration it applies to).
- If excerpts disagree with each other (for example about a configuration), point it out instead of picking silently.
- Be concise. End with the sources you used, formatted like: (Sources: p.10 table; p.11 footnote 7)."""


def tokenize(text):
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.inv = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.inv[w].append(i)

    def scores(self, query):
        s = np.zeros(len(self.tf), dtype="float32")
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i in self.inv[w]:
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(rank_lists, k=60):
    out = defaultdict(float)
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            out[i] += 1.0 / (k + r + 1)
    return out


TYPE_LABEL = {"text": "Text", "callout": "Text (callout)", "table": "Table", "chart": "Chart",
              "footnote": "Footnote", "definition": "Definition", "vision": "Figure/table (vision)"}


class RAG:
    def __init__(self):
        if not config.API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
        self.client = genai.Client(api_key=config.API_KEY)
        self.vectors = np.load(config.INDEX_DIR / "embeddings.npy")
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.notes = {int(k): v for k, v in json.loads((config.INDEX_DIR / "footnotes.json").read_text(encoding="utf-8")).items()}
        self.bm25 = BM25([c["breadcrumb"] + " " + c["text"] for c in self.chunks])

    # ---------- LLM helpers ----------
    def _generate(self, prompt, system=None, temperature=0.1):
        res = self.client.models.generate_content(
            model=config.CHAT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature))
        return (res.text or "").strip()

    def embed_query(self, q):
        res = self.client.models.embed_content(
            model=config.EMBED_MODEL, contents=[q],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=config.EMBED_DIM))
        v = np.array(res.embeddings[0].values, dtype="float32")
        return v / np.linalg.norm(v)

    def rewrite(self, question, history):
        if not history:
            return question
        convo = "\n".join(f"{h['role']}: {h['content'][:500]}" for h in history[-4:])
        prompt = ("Rewrite the user's last question as a standalone search question using the conversation for context. "
                  "Keep product configurations (e.g. 'two ports 256GB'), numbers and report terms. Output only the question.\n\n"
                  f"Conversation:\n{convo}\n\nLast question: {question}\nStandalone question:")
        try:
            return self._generate(prompt, temperature=0) or question
        except Exception:
            return question

    # ---------- retrieval ----------
    def _boosts(self, query):
        q = query.lower()
        boost = defaultdict(float)
        type_boost = defaultdict(float)
        numeric_intent = False
        if re.search(r"\b(how much|how many|what percent|percentage|percent|kg|co2|watt|power|energy|consum|emission|footprint|total|number|share|%|reduction|recycled)\b", q):
            numeric_intent = True
            for t, b in (("table", .005), ("chart", .005), ("vision", .004), ("callout", .002)):
                type_boost[t] += b
        if re.search(r"\b(table|row|column|mode|voltage|sleep|idle)\b", q):
            type_boost["table"] += .012
        if re.search(r"\b(chart|graph|figure|bar|diagram|infographic|baseline|segment)\b", q):
            type_boost["chart"] += .012; type_boost["vision"] += .008
        fn_ids = {int(m.group(1)) for m in re.finditer(r"\b(?:footnote|endnote|note)s?\s*#?(\d{1,2})\b", q)}
        if fn_ids or re.search(r"\b(footnote|endnote|note|calculated|calculation|how is|defined|definition|means?|methodology|assum|exception|exempt)", q):
            type_boost["footnote"] += .006; type_boost["definition"] += .006
        else:                               # long footnotes would otherwise win keyword ranking on generic questions
            type_boost["footnote"] -= .006; type_boost["definition"] -= .004
        configs = [s for s in ("four ports", "two ports", "512gb", "256gb") if s in q]
        for c in self.chunks:
            i = c["id"]
            boost[i] += type_boost.get(c["type"], 0)
            if numeric_intent and re.search(r"\d+\s?(%|percent|kg|W\b)", c["text"]):
                boost[i] += 0.003                      # chunks that actually contain figures
            if fn_ids:
                if c["type"] == "footnote" and c.get("footnote_id") in fn_ids:
                    boost[i] += 0.12
                elif fn_ids & set(c["footnotes"]):
                    boost[i] += 0.015
            low = c["text"].lower()
            for s in configs:
                if s in low:
                    boost[i] += 0.008
        return boost

    def retrieve(self, query, content_type=None, k=config.FINAL_K, rerank=False):
        allowed_types = set(TYPE_GROUPS.get(content_type, [])) if content_type else None
        allowed = np.array([(allowed_types is None or c["type"] in allowed_types) for c in self.chunks])
        dense = np.where(allowed, self.vectors @ self.embed_query(query), -1e9)
        sparse = np.where(allowed, self.bm25.scores(query), -1e9)
        d_rank = [int(i) for i in np.argsort(-dense)[:config.CANDIDATES]]
        s_rank = [int(i) for i in np.argsort(-sparse)[:config.CANDIDATES] if sparse[i] > 0]
        fused = rrf([d_rank, s_rank])
        for i, b in self._boosts(query).items():
            if allowed[i]:
                fused[i] = fused.get(i, 0) + b
        ranked = sorted(fused, key=lambda i: -fused[i])
        pool = ranked[:config.RERANK_POOL if rerank else k]
        if rerank:
            pool = self.rerank(query, pool, k)
        return [{**self.chunks[i], "score": round(float(fused[i]), 4)} for i in pool]

    def rerank(self, query, ids, k):
        listing = "\n".join(f"[{i}] ({self.chunks[i]['type']}, p.{self.chunks[i]['page']}) {self.chunks[i]['text'][:300]}" for i in ids)
        prompt = (f"Question: {query}\n\nPassages:\n{listing}\n\n"
                  f"Return a JSON list of the ids of the {k} passages most useful for answering the question, best first. "
                  "Output only the JSON list, e.g. [12, 4, 30].")
        try:
            out = self._generate(prompt, temperature=0)
            picked = list(dict.fromkeys(int(x) for x in re.findall(r"\d+", out) if int(x) in ids))[:k]
            return (picked + [i for i in ids if i not in picked])[:k] if picked else ids[:k]
        except Exception:
            return ids[:k]

    # ---------- context ----------
    def build_context(self, hits):
        blocks, sources, size, shown_notes = [], [], 0, set()
        for n, h in enumerate(hits, start=1):
            body = h["text"]
            notes = []
            for fid in h["footnotes"]:
                if fid in self.notes and fid not in shown_notes and h["type"] != "footnote":
                    notes.append(f"  [footnote {fid}] {self.notes[fid][:650]}")
                    shown_notes.add(fid)
            label = TYPE_LABEL[h["type"]]
            block = f"[{n}] p.{h['page']} | {label}" + (f" | {h['topic']}" if h.get("topic") else "") + f"\n{body}"
            if notes:
                block += "\nFootnotes cited here:\n" + "\n".join(notes)
            if size + len(block) > config.MAX_CONTEXT_CHARS and blocks:
                break
            blocks.append(block)
            size += len(block)
            sources.append({"id": h["id"], "page": h["page"], "type": h["type"], "label": label,
                            "topic": h.get("topic"), "footnotes": h["footnotes"], "text": h["text"][:360]})
        return "\n\n".join(blocks), sources

    # ---------- main ----------
    def answer(self, question, history=None, content_type=None, rerank=False):
        standalone = self.rewrite(question, history or [])
        hits = self.retrieve(standalone, content_type=content_type or None, rerank=rerank)
        context, sources = self.build_context(hits)
        text = self._generate(f"Context excerpts:\n{context}\n\nQuestion: {standalone}\nAnswer:", system=SYSTEM_PROMPT)
        return {"answer": text, "query_used": standalone, "sources": sources,
                "retrieved_pages": sorted({s["page"] for s in sources})}
