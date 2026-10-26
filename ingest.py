"""Multi-modal ingestion for the iMac Product Environmental Report.

  text blocks   layout-aware XY-cut -> cards / callouts / paragraphs keep their own chunks
  tables        borderless tables rebuilt from word positions -> markdown + one fact sentence per row
  charts        text-layer labels + (optional) Gemini vision description of the page image
  footnotes     the 19 endnotes become their own chunks AND are linked to every chunk that cites them
  definitions   the glossary page is split per term

Run:  python ingest.py                (full build)
      python ingest.py --no-vision    (skip Gemini vision calls)
      python ingest.py --no-embed     (inspect chunks only; no API calls at all)
"""
import json
import re
import sys
import time

import numpy as np
import pdfplumber

import config
import parse
from parse import norm

PAGE_TITLES = {}


# ---------------------------------------------------------------------------
def page_title(blocks):
    big = [b for b in blocks if b["size"] >= 20]
    if not big:
        return None
    first = big[0]
    parts = [b["text"] for b in big if abs(b["x0"] - first["x0"]) < 5]
    return " ".join(parts)


def split_sentences(text, limit):
    if len(text) <= limit:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(•])", text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > limit and cur:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def pair_headings(blocks):
    """Attach short bold headings / big stat numbers to the block below them in the same column."""
    used, units = set(), []
    for i, b in enumerate(blocks):
        if i in used:
            continue
        is_head = b["bold_lead"] and len(b["text"]) <= 70 and not b["text"].rstrip().endswith(".") and b["size"] < 20
        if is_head:
            cx = (min(w["x0"] for w in b["words"]) + max(w["x1"] for w in b["words"])) / 2
            for j in range(i + 1, len(blocks)):
                c = blocks[j]
                if j in used or c["bold_lead"] and len(c["text"]) <= 70 and not c["text"].endswith("."):
                    continue
                ccx = (min(w["x0"] for w in c["words"]) + max(w["x1"] for w in c["words"])) / 2
                if c["top"] > b["top"] and c["top"] - b["top"] < 220 and abs(ccx - cx) < 110:
                    used.add(j)
                    units.append({"heading": b["text"], "text": b["text"] + ": " + c["text"], "bold": True,
                                  "top": b["top"], "words": b["words"] + c["words"]})
                    break
            else:
                units.append({"heading": b["text"], "text": b["text"], "bold": True, "top": b["top"], "words": b["words"]})
        else:
            m = re.match(r"^([A-Z][A-Za-z ]{2,30})\s?\.\s+(We|Our)\b", b["text"]) if b["bold_lead"] else None
            units.append({"heading": m.group(1) if m else None, "text": b["text"], "bold": bool(m) or b["bold_lead"],
                          "top": b["top"], "words": b["words"]})
    return units


def build_chunks_for_page(pno, page, endnotes_words_exclude=None):
    chunks = []
    tables, words = parse.find_tables(page)
    exclude = set().union(*[t["word_ids"] for t in tables]) if tables else set()
    blocks = parse.blocks_for_page(words, exclude)
    title = page_title(blocks) or ""
    PAGE_TITLES[pno] = title
    body = [b for b in blocks if b["size"] < 20 and not (b["size"] < 7 and len(b["text"]) < 40 and b["text"] == title)]
    markers = parse.find_markers(page)

    def add(ctype, text, topic=None, extra=None, solo=False):
        for piece in split_sentences(text, config.CHUNK_MAX):
            n = norm(piece)
            fns = sorted({num for num, ctx in markers if len(norm(ctx)) >= 4 and norm(ctx)[-14:] + str(num) in n})
            chunks.append({"type": ctype, "page": pno, "page_title": title, "topic": topic, "text": piece,
                           "footnotes": fns, "solo": solo, **(extra or {})})

    # --- tables
    for t in tables:
        subject = t["title"] or (t["header"][1] if t["header"] and len(t["header"]) == 2 else title) or title
        if t["title"] is None and t["header"] and len(t["header"]) == 2:
            subject = f"{title} ({t['header'][1]})"
        md = parse.table_to_markdown(t)
        facts = parse.table_row_facts(t, subject)
        text = f"Table on page {pno}: {subject}\n{md}\nRow facts:\n" + "\n".join(facts)
        n = norm(text)
        fns = sorted({num for num, ctx in markers if norm(ctx)[-14:] + str(num) in n})
        chunks.append({"type": "table", "page": pno, "page_title": title, "topic": subject, "text": text, "footnotes": fns})

    if pno in config.ENDNOTE_PAGES:
        return chunks, tables

    # --- chart labels: many short small-font blocks below the main paragraph
    short = [b for b in body if len(b["text"]) < 45 and b["size"] <= 9.5 and not b["bold_lead"]]
    if len(short) >= 8:
        lab = sorted(short, key=lambda b: (round(b["top"] / 8), b["x0"]))
        short_ids = {id(b) for b in short}
        keys = sorted([b for b in body if re.match(r"^[–−-]?\d+\s?kg\b", b["text"])], key=lambda b: b["x0"])
        short_ids |= {id(b) for b in keys}
        key_txt = ""
        if keys:
            key_txt = "Key values shown on the chart (left to right): " + "; ".join(b["text"] for b in keys) + ". "
            vals = [float(re.match(r"^([–−-]?\d+)", b["text"]).group(1).replace("–", "-").replace("−", "-")) for b in keys]
            if len(vals) == 3 and abs(vals[0] + vals[1] - vals[2]) < 1e-6:
                key_txt += (f"Arithmetic check: {vals[0]:g} kg baseline emissions plus ({vals[1]:g} kg) emissions reductions "
                            f"equals {vals[2]:g} kg iMac emissions. ")
        text = (f"Carbon footprint chart on page {pno}. " + key_txt
                + "Chart text labels (text layer; the vision description gives the structure): "
                + "; ".join(b["text"] for b in lab))
        n = norm(text)
        fns = sorted({num for num, ctx in markers if norm(ctx)[-14:] + str(num) in n})
        chunks.append({"type": "chart", "page": pno, "page_title": title, "topic": "chart labels", "text": text, "footnotes": fns})
        body = [b for b in body if id(b) not in short_ids]

    # --- glossary page: one chunk per defined term
    if pno == 9:
        full = " ".join(b["text"] for b in body)
        full = re.sub(r"^Definitions\s*", "", full)
        for entry in re.split(r"(?<=[.)])\s+(?=[A-Z][A-Za-z\- ]{3,40}:\s)", full):
            term = entry.split(":", 1)[0].strip()
            add("definition", entry, topic=term)
        return chunks, tables

    # --- narrative / cards / callouts
    units = pair_headings(body)
    buf = None
    for u in units:
        if u["heading"] or u["bold"]:
            if buf:
                add("text", buf["text"]); buf = None
            is_callout = bool(u["heading"]) and bool(re.match(r"^[A-Z][A-Za-z ]+\s?\.\s", u["text"]))
            add("callout" if is_callout else "text", u["text"], topic=u["heading"], solo=True)
        else:
            if buf and len(buf["text"]) + len(u["text"]) < config.CHUNK_MAX:
                buf["text"] += " " + u["text"]
            else:
                if buf:
                    add("text", buf["text"])
                buf = {"text": u["text"]}
    if buf:
        add("text", buf["text"])
    # merge tiny chunks into their neighbour
    merged = []
    for c in chunks:
        if (merged and c["type"] == "text" and not c.get("solo") and len(c["text"]) < 70
                and merged[-1]["page"] == pno and merged[-1]["type"] == "text" and not merged[-1].get("solo")):
            merged[-1]["text"] += " " + c["text"]
        else:
            merged.append(c)
    claimed = {n for c in merged for n in c["footnotes"]}
    loose = sorted({num for num, _ in markers} - claimed)
    host = next((c for c in merged if c["type"] == "chart"), None)
    if host and loose:
        host["footnotes"] = sorted(set(host["footnotes"]) | set(loose))
    return merged, tables


def build_footnote_chunks(pdf):
    lines = []
    for pn in config.ENDNOTE_PAGES:
        page = pdf.pages[pn - 1]
        tables, words = parse.find_tables(page)
        exclude = set().union(*[t["word_ids"] for t in tables]) if tables else set()
        ws = [w for w in words if id(w) not in exclude]
        for ln in parse.group_lines(ws):
            lines.append(" ".join(w["text"] for w in ln["words"]))
    notes = parse.parse_endnotes("\n".join(lines))
    chunks = []
    for n, text in notes.items():
        for piece in split_sentences(f"Footnote {n}: {text}", 1100):
            chunks.append({"type": "footnote", "page": 11 if n <= 16 else 12, "page_title": "Endnotes",
                           "topic": f"Footnote {n}", "text": piece, "footnotes": [], "footnote_id": n})
    return notes, chunks


# ---------------------------------------------------------------------------
VISION_PROMPT = """You are extracting data from page {p} of Apple's "iMac Product Environmental Report" (Oct 2024).
Describe ONLY what is visible. Cover every chart, table, infographic and big statistic on the page.
Output Markdown with:
1. Lines starting with "FACT:" - one self-contained sentence per fact, including numbers, units, product configuration
   (e.g. "iMac (two ports) 256GB") and any footnote number shown as a superscript (write it as "[footnote N]").
2. For charts: describe the structure (what each bar/segment is, which values add up to which total, arrows, reductions).
3. For tables: reproduce them as a Markdown table.
Do not add information that is not on the page."""


def vision_chunks(pdf, use_vision):
    cache_file = config.INDEX_DIR / "vision_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    if not use_vision:
        return [], cache
    if not config.API_KEY:
        print("  (no GEMINI_API_KEY - skipping vision)")
        return [], cache
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=config.API_KEY)
    out = []
    for pno in config.VISION_PAGES:
        if str(pno) not in cache:
            img = config.PAGES_DIR / f"page-{pno}.png"
            print(f"  vision: page {pno} ...")
            for attempt in range(6):
                try:
                    res = client.models.generate_content(
                        model=config.CHAT_MODEL,
                        contents=[types.Part.from_bytes(data=img.read_bytes(), mime_type="image/png"),
                                  VISION_PROMPT.format(p=pno)],
                        config=types.GenerateContentConfig(temperature=0))
                    cache[str(pno)] = (res.text or "").strip()
                    break
                except Exception as e:
                    wait = min(2 ** attempt * 8, 90)
                    print(f"    vision error ({str(e)[:90]}); retry in {wait}s")
                    time.sleep(wait)
            else:
                print(f"    skipping page {pno}")
                continue
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(7)
    for pno in config.VISION_PAGES:
        text = cache.get(str(pno))
        if not text:
            continue
        title = PAGE_TITLES.get(pno, "")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras + [None]:
            if p is not None and len(buf) + len(p) < 1300:
                buf += ("\n\n" if buf else "") + p
                continue
            if buf:
                out.append({"type": "vision", "page": pno, "page_title": title, "topic": "Gemini vision description",
                            "text": "Visual content of page %d (read from the page image):\n%s" % (pno, buf), "footnotes": []})
            buf = p or ""
    return out, cache


# ---------------------------------------------------------------------------
def breadcrumb(c):
    label = {"table": "Table", "chart": "Chart", "footnote": "Footnote", "definition": "Definition",
             "vision": "Figure/table (vision)", "callout": "Callout", "text": "Text"}[c["type"]]
    parts = [config.DOC_NAME, f"p.{c['page']} {c['page_title']}".strip(), label]
    if c.get("topic") and c["topic"] not in parts[1]:
        parts.append(c["topic"])
    return " > ".join(parts)


def _retry_delay(err, default):
    m = re.search(r"retry in ([\d.]+)s", str(err))
    return float(m.group(1)) + 2 if m else default


def embed_texts(client, texts):
    from google.genai import types
    config.INDEX_DIR.mkdir(exist_ok=True)
    cache = config.INDEX_DIR / "partial_embeddings.json"
    vecs = json.loads(cache.read_text()) if cache.exists() else []
    if vecs:
        print(f"  resuming from {len(vecs)}/{len(texts)}")
    for i in range(len(vecs), len(texts), config.EMBED_BATCH):
        batch = texts[i:i + config.EMBED_BATCH]
        for attempt in range(12):
            try:
                res = client.models.embed_content(
                    model=config.EMBED_MODEL, contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=config.EMBED_DIM))
                vecs.extend(e.values for e in res.embeddings)
                break
            except Exception as e:
                msg = str(e)
                if attempt == 0:
                    print("\n  API error (full message):\n  " + msg + "\n")
                if "perday" in msg.lower().replace(" ", "").replace("_", ""):
                    print(f"  Stopping: DAILY quota reached. {len(vecs)}/{len(texts)} saved - re-run later (or use a key from a new project).")
                    sys.exit(1)
                wait = _retry_delay(e, min(2 ** attempt * 5, 120))
                print(f"  rate limit/error; waiting {wait:.0f}s")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed after retries (re-run to resume)")
        cache.write_text(json.dumps(vecs))
        print(f"  embedded {len(vecs)}/{len(texts)}")
        if len(vecs) < len(texts):
            time.sleep(config.EMBED_PAUSE)
    arr = np.array(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    cache.unlink(missing_ok=True)
    return arr


def render_pages(pdf):
    config.PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pdf.pages, start=1):
        f = config.PAGES_DIR / f"page-{i}.png"
        if not f.exists():
            page.to_image(resolution=config.PAGE_DPI).save(str(f))


def main():
    use_vision = "--no-vision" not in sys.argv and "--no-embed" not in sys.argv
    embed = "--no-embed" not in sys.argv
    config.INDEX_DIR.mkdir(exist_ok=True)
    chunks = []
    with pdfplumber.open(str(config.PDF_PATH)) as pdf:
        print("Rendering page images...")
        render_pages(pdf)
        print("Parsing text, tables, footnotes...")
        for pno, page in enumerate(pdf.pages, start=1):
            cs, _ = build_chunks_for_page(pno, page)
            chunks.extend(cs)
        notes, fn_chunks = build_footnote_chunks(pdf)
        chunks.extend(fn_chunks)
        v_chunks, _ = vision_chunks(pdf, use_vision)
        chunks.extend(v_chunks)

    for i, c in enumerate(chunks):
        c["id"] = i
        c["breadcrumb"] = breadcrumb(c)
    for c in chunks:
        c.pop("solo", None)
    kinds = {}
    for c in chunks:
        kinds[c["type"]] = kinds.get(c["type"], 0) + 1
    print(f"{len(chunks)} chunks:", kinds, "| endnotes parsed:", len(notes))
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (config.INDEX_DIR / "footnotes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    if not embed:
        print("Skipped embeddings (--no-embed). See index/chunks.json")
        return
    if not config.API_KEY:
        sys.exit("GEMINI_API_KEY missing. Create a .env file (see .env.example).")
    from google import genai
    client = genai.Client(api_key=config.API_KEY)
    print("Embedding chunks...")
    vectors = embed_texts(client, [c["breadcrumb"] + "\n\n" + c["text"] for c in chunks])
    np.save(config.INDEX_DIR / "embeddings.npy", vectors)
    print("Saved index to", config.INDEX_DIR)


if __name__ == "__main__":
    main()
