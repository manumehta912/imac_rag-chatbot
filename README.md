# Apple iMac Product Environmental Report - Multi-modal RAG

Important facts in this report are *not* in running text: they live in a chart (page 2), tables (pages 10 and 12),
big-number callouts (pages 1, 4, 6) and 19 endnotes (pages 11-12) that qualify the numbers. This project makes all of them searchable.

## Parsing techniques
| Content | Technique |
|---|---|
| Cards, callouts, columns | **XY-cut layout analysis** (`parse.py`): words are grouped into visual blocks and read in the right order; headings / big stat numbers are paired with the block below them ("100%: of the packaging is fiber-based...") |
| Borderless tables (p10, p12) | Rebuilt from word positions (label / value cells, bold header row, sub/superscript repair such as `CO2e`) -> **Markdown table + one fact sentence per row** ("Power consumption for iMac - Sleep: 100V: 0.56W; 115V: 0.57W; 230V: 0.59W") |
| Chart (p2) | Text-layer labels + key values (445 kg baseline, -99 kg reductions, 346 kg result, arithmetic checked in code) **and** an optional **Gemini vision** description of the page image (also for the cover infographic, packaging stats and both tables), cached in `index/vision_cache.json` |
| Footnotes | The 19 endnotes are parsed into their own chunks **and linked to every chunk that cites them**: superscript markers are detected from glyph size/position, so a retrieved chunk is always sent to the model together with its qualifiers |
| Glossary (p9) | One chunk per defined term |

## Retrieval
Gemini embeddings + BM25 (Reciprocal Rank Fusion), content-type aware boosts (numeric questions favour tables/charts/callouts,
"footnote 7" pulls footnote 7, "two ports / 256GB" favours matching chunks), long footnotes are down-weighted for generic
questions, optional content-type filter in the UI, optional Gemini reranking ("Precision mode"), follow-up question rewriting.
Every source in the UI has a **View page** button that opens the original page image so tables, charts and footnotes can be checked.

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # paste your GEMINI_API_KEY
python ingest.py --no-embed                          # inspect parsing only (no API calls) -> index/chunks.json
python ingest.py                                     # full build: parse + Gemini vision (5 pages) + embeddings (~75 chunks)
uvicorn app:app --reload                             # http://127.0.0.1:8000
```
`python ingest.py --no-vision` skips the vision calls. Embedding and vision results are cached/resumable.

## Validate
Put the provided sample questions in `questions.txt`: `question | expected pages | expected keywords` (pages and keywords optional), then
```bash
python evaluate.py            # add --rerank for precision mode
```
It reports whether an expected page was retrieved and whether the expected values appear in the answer.

## Things worth knowing about this document
- The report is internally inconsistent: page 2 and page 10 say the 346 kg figure is for **iMac (two ports) 256GB**, but endnote 10 says it is based on **512GB** storage. The assistant is instructed to point out conflicts like this rather than choose silently.
- Two configurations appear: iMac (two ports) 256GB = 346 kg CO2e, iMac (four ports) 512GB = 391 kg CO2e. Answers should always name the configuration.
- Footnote markers drawn as logos (e.g. the EPEAT mark on page 1) are not detected by the text parser; the vision description and endnote 6 cover them.
