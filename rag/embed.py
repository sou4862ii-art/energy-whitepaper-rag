"""
埋め込み（ベクトル化）。使わない設定（none）でもキーワード検索だけで動く。

  voyage : Voyage AI（Anthropic が推奨する埋め込み。APIキー VOYAGE_API_KEY が必要）
  local  : 手元のPCで動く multilingual-e5（無料。初回にモデルを自動ダウンロード、
           pip install sentence-transformers が必要）
"""

import numpy as np

from . import config


def _normalize(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class VoyageEmbedder:
    name = "voyage"

    def __init__(self, model=None):
        import voyageai
        self.client = voyageai.Client()          # 環境変数 VOYAGE_API_KEY を使う
        self.model = model or config.VOYAGE_MODEL

    def embed(self, texts, kind="document", batch=64):
        out = []
        for i in range(0, len(texts), batch):
            r = self.client.embed(texts[i:i + batch], model=self.model,
                                  input_type="query" if kind == "query" else "document")
            out.extend(r.embeddings)
        return _normalize(out)


class LocalE5Embedder:
    name = "local"

    def __init__(self, model=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model or config.LOCAL_MODEL)

    def embed(self, texts, kind="document", batch=32):
        # e5 系は先頭に "query: " / "passage: " を付けるのが正しい使い方
        prefix = "query: " if kind == "query" else "passage: "
        vecs = self.model.encode([prefix + t for t in texts], batch_size=batch,
                                 show_progress_bar=len(texts) > 100)
        return _normalize(vecs)


def get_embedder(name=None):
    name = (name or config.EMBEDDER).lower()
    if name == "none":
        return None
    if name == "voyage":
        return VoyageEmbedder()
    if name == "local":
        return LocalE5Embedder()
    raise ValueError(f"RAG_EMBEDDER は none / voyage / local のいずれか: {name}")
