"""設定値。環境変数（または .env）で上書きできる。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv():
    """python-dotenv を使わずに .env を読む（KEY=VALUE 形式のみ）。"""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

PDF_DIR = Path(os.getenv("RAG_PDF_DIR", ROOT / "data" / "pdf"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "data" / "index"))

# Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1500"))

# 埋め込み（ベクトル検索）: none / voyage / local
EMBEDDER = os.getenv("RAG_EMBEDDER", "none")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4")
LOCAL_MODEL = os.getenv("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-small")

# チャンクと検索
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "500"))
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
