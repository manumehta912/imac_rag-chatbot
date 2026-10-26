"""Layout-aware parsing helpers for the iMac Product Environmental Report.

* xy_cut_blocks   - groups words into visual blocks (cards, columns, paragraphs) in reading order
* find_tables     - rebuilds borderless tables from word positions (label | value cells)
* find_markers    - finds superscript footnote markers (and where they occur)
* parse_endnotes  - turns the "Endnotes" pages into {number: text}
"""
import re
import unicodedata
from collections import defaultdict


def norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", s).lower())


def fix_text(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"CO\s*(?:2\s*)?e\b(?:\s*2\b)?", "CO2e", s)       # CO2e with subscript glyph
    s = re.sub(r"\bT esting\b", "Testing", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------- words / lines
def page_words(page):
    return page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False, x_tolerance=1.5, y_tolerance=2)


def group_lines(words, tol=3.0):
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        for ln in lines:
            if abs(ln["top"] - w["top"]) <= tol:
                ln["words"].append(w)
                break
        else:
            lines.append({"top": w["top"], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
    lines.sort(key=lambda l: l["top"])
    return lines


# ---------------------------------------------------------------- tables
VALUE_RE = re.compile(r"^[<>↓≥≤~]*\s?\d[\d.,]*\s?(?:kg\s?CO2e|kg|W|V|%|GB|percent)?$|^[<>↓]+\s?\d", re.I)


def _cells(row_words, gap_mult=1.6):
    ws = sorted(row_words, key=lambda w: w["x0"])
    cells = [[ws[0]]]
    for a, b in zip(ws, ws[1:]):
        if b["x0"] - a["x1"] > gap_mult * max(a["size"], b["size"], 6):
            cells.append([b])
        else:
            cells[-1].append(b)
    out = []
    for c in cells:
        c.sort(key=lambda w: (round(w["top"] / 4), w["x0"]))
        out.append({"x0": min(w["x0"] for w in c), "x1": max(w["x1"] for w in c),
                    "text": fix_text(" ".join(w["text"] for w in c)),
                    "size": max(w["size"] for w in c),
                    "bold": any("Semibold" in w["fontname"] or "Bold" in w["fontname"] for w in c)})
    return out


def _rows(words, tol=6.0):
    """Chain words whose tops are within tol into one visual row (handles sub/superscripts)."""
    words = sorted(words, key=lambda w: w["top"])
    rows, cur, last = [], [], None
    for w in words:
        if last is not None and w["top"] - last > tol:
            rows.append(cur)
            cur = []
        cur.append(w)
        last = w["top"]
    if cur:
        rows.append(cur)
    return rows


def find_tables(page, min_rows=3, max_top=735):
    """Return [{'title', 'top', 'bottom', 'rows': [[cell,...]], 'word_ids'}] for borderless tables."""
    words = [w for w in page_words(page) if w["top"] < max_top]
    rows = [(_cells(r), min(w["top"] for w in r), max(w["bottom"] for w in r), r) for r in _rows(words)]

    def is_data(cells):
        return len(cells) >= 2 and bool(re.search(r"\d", cells[-1]["text"])) and len(cells[-1]["text"]) <= 28 \
            and cells[-1]["x0"] > 150 and len(cells[0]["text"]) <= 70

    tables, cur = [], []
    for i, (cells, top, bottom, rw) in enumerate(rows):
        if is_data(cells) and (not cur or top - cur[-1][2] < 30):
            cur.append((cells, top, bottom, rw))
        else:
            if len(cur) >= min_rows:
                tables.append(cur)
            cur = [(cells, top, bottom, rw)] if is_data(cells) else []
    if len(cur) >= min_rows:
        tables.append(cur)

    result = []
    for t in tables:
        first_top = t[0][1]
        # header / title rows directly above the first data row
        above = [r for r in rows if r[2] <= first_top + 1 and first_top - r[2] < 26]
        title, header, used_ids = None, None, set()
        ncols = max(len(r[0]) for r in t)
        for cells, top, bottom, rw in reversed(above[-2:]):
            if len(cells) == 1 and cells[0]["bold"] and cells[0]["size"] < 10 and title is None and header is None:
                title = cells[0]["text"]
                used_ids |= {id(w) for w in rw}
            elif (len(cells) == ncols and all(c["bold"] for c in cells) and header is None
                  and all(len(c["text"]) <= 40 for c in cells)):
                header = [c["text"] for c in cells]
                used_ids |= {id(w) for w in rw}
        data = [[c["text"] for c in cells] for cells, *_ in t]
        if header is None and all(c["bold"] for c in t[0][0]) and len(data) > 1:
            header, data = data[0], data[1:]                   # first row is a bold header row
        wids = {id(w) for *_, rw in t for w in rw} | used_ids
        result.append({"title": title, "header": header, "rows": data, "top": min(r[1] for r in t),
                       "bottom": max(r[2] for r in t), "word_ids": wids})
    return result, words


def table_to_markdown(t):
    rows = ([t["header"]] if t.get("header") else []) + t["rows"]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def table_row_facts(t, subject):
    """One self-contained sentence per row (much better for retrieval than a raw grid)."""
    facts, header = [], t.get("header")
    for r in t["rows"]:
        label, vals = r[0], r[1:]
        if header and len(header) == len(r) and len(header) > 2:
            pairs = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], vals))
            facts.append(f"{subject} - {label}: {pairs}.")
        elif header and len(header) == 2 and len(vals) == 1:
            facts.append(f"{subject} - {label}: {vals[0]}." if header[1] in subject else f"{subject} - {label} ({header[1]}): {vals[0]}.")
        else:
            facts.append(f"{subject} - {label}: {' | '.join(vals)}.")
    return facts


# ---------------------------------------------------------------- XY-cut blocks
def xy_cut(words, min_gap_y=9.0, min_gap_x=14.0, depth=0):
    if len(words) <= 1 or depth > 12:
        return [words]
    ws = sorted(words, key=lambda w: w["top"])
    bands, cur, bottom = [], [ws[0]], ws[0]["bottom"]
    for w in ws[1:]:
        if w["top"] - bottom > min_gap_y:
            bands.append(cur)
            cur = []
        cur.append(w)
        bottom = max(bottom, w["bottom"])
    bands.append(cur)
    if len(bands) > 1:
        return [b for band in bands for b in xy_cut(band, min_gap_y, min_gap_x, depth + 1)]
    ws = sorted(words, key=lambda w: w["x0"])
    cols, cur, right = [], [ws[0]], ws[0]["x1"]
    for w in ws[1:]:
        if w["x0"] - right > min_gap_x:
            cols.append(cur)
            cur = []
        cur.append(w)
        right = max(right, w["x1"])
    cols.append(cur)
    if len(cols) > 1:
        return [b for col in cols for b in xy_cut(col, min_gap_y, min_gap_x, depth + 1)]
    return [words]


def block_text(words):
    lines = group_lines(words)
    out = []
    for ln in lines:
        out.append(" ".join(w["text"] for w in ln["words"]))
    text = " ".join(out)
    text = re.sub(r"(\w)- (?=[a-z])", r"\1", text) if False else text
    return fix_text(text)


def blocks_for_page(words, exclude_ids=frozenset(), skip_footer_top=735):
    ws = [w for w in words if id(w) not in exclude_ids and w["top"] < skip_footer_top]
    if not ws:
        return []
    out = []
    for b in xy_cut(ws):
        text = block_text(b)
        if len(text) < 3:
            continue
        first = group_lines(b)[0]["words"]
        out.append({"text": text, "top": min(w["top"] for w in b), "x0": min(w["x0"] for w in b),
                    "size": max(w["size"] for w in first), "bold_lead": "Semibold" in first[0]["fontname"] or "Bold" in first[0]["fontname"],
                    "words": b})
    return out


# ---------------------------------------------------------------- footnote markers
def find_markers(page):
    """Detect superscript footnote numbers. Returns [(number, context_before)]."""
    stream = [c for c in page.chars]
    res = []
    k = 0
    while k < len(stream):
        c = stream[k]
        if c["text"].isdigit() and k > 0:
            p = stream[k - 1]
            if (p["text"].strip() and not p["text"].isdigit() and c["size"] < 0.85 * p["size"]
                    and abs(c["bottom"] - p["bottom"]) < p["size"] * 1.0 and -2 <= c["x0"] - p["x1"] < 6):
                num, j = c["text"], k + 1
                while j < len(stream) and stream[j]["text"].isdigit() and abs(stream[j]["size"] - c["size"]) < 0.3 \
                        and stream[j]["x0"] - stream[j - 1]["x1"] < 2:
                    num += stream[j]["text"]
                    j += 1
                ctx = "".join(x["text"] for x in stream[max(0, k - 22):k])
                if not re.search(r"CO$", ctx.strip()) and 1 <= int(num) <= 30:
                    res.append((int(num), ctx))
                k = j
                continue
        k += 1
    return res


# ---------------------------------------------------------------- endnotes
def parse_endnotes(text):
    """text: concatenated text of the Endnotes pages. Numbers must appear in increasing order."""
    text = unicodedata.normalize("NFKC", text)
    lines = [l.strip() for l in text.split("\n")]
    notes, cur, expected = {}, None, 1
    for ln in lines:
        ln = re.sub(r"^Endnotes\s*", "", ln)
        if not ln or ln.startswith("©"):
            continue
        m = re.match(r"^(\d{1,2})\s?(?=[A-Za-z])", ln)
        if m and int(m.group(1)) == expected:
            cur = expected
            notes[cur] = ln[m.end():].strip()
            expected += 1
        elif cur is not None and not re.match(r"^(Apple Store|Mac, the Mac|herein may|mark are|Apple, the Apple)", ln):
            notes[cur] += " " + ln
    return {k: fix_text(v) for k, v in notes.items()}
