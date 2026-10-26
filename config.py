import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

PDF_PATH = BASE_DIR / "data" / "iMac_PER_Oct2024.pdf"
INDEX_DIR = BASE_DIR / "index"
PAGES_DIR = BASE_DIR / "static" / "pages"      # page images shown as evidence in the UI

DOC_NAME = "Apple iMac Product Environmental Report (Oct 2024)"
ENDNOTE_PAGES = (11, 12)                        # pages holding the numbered endnotes
VISION_PAGES = [1, 2, 6, 10, 12]                # cover infographic, chart, stat callouts, tables
PAGE_DPI = 110                                  # rendered page images (UI evidence + vision input)

CHUNK_MAX = 900
EMBED_DIM = 768
EMBED_BATCH = 10
EMBED_PAUSE = 12

CANDIDATES = 20
FINAL_K = 6
RERANK_POOL = 12
MAX_CONTEXT_CHARS = 12000
