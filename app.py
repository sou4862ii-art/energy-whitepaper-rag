"""
エネルギー白書に質問できるWeb画面。

    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from rag import config
from rag.answer import ask
from rag.embed import get_embedder
from rag.pdfview import render_page
from rag.search import Index

st.set_page_config(page_title="エネルギー白書に質問", page_icon="⚡", layout="wide")

EXAMPLES = [
    "地熱発電の導入が進まない理由は？",
    "太陽光発電の出力制御はなぜ起きる？",
    "中東情勢を受けて政府はどんな対策をとった？",
]


@st.cache_resource
def load_index(index_dir):
    meta = Index.read_meta(index_dir)
    return Index.load(index_dir, get_embedder(meta.get("embedder", "none"))), meta


with st.sidebar:
    st.header("設定")
    index_dir = st.text_input("索引フォルダ", str(config.INDEX_DIR))
    top_k = st.slider("Claudeに渡す抜粋の数", 3, 15, config.TOP_K)
    model = st.text_input("Claudeのモデル", config.CLAUDE_MODEL)
    dry_run = st.checkbox("検索結果だけ表示（APIを使わない）")

try:
    index, meta = load_index(index_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pdf_dir = Path(meta.get("pdf_dir") or config.PDF_DIR)
st.title("エネルギー白書に質問する")
st.caption(f"収録: {', '.join(meta.get('docs', []))}　／　{meta.get('n_chunks', '?')} 抜粋"
           f"　／　検索: キーワード{' ＋ ベクトル' if index.vectors is not None else ''}")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex
question = st.text_input("質問", key="q", placeholder="例: 地熱発電の導入が進まない理由は？")

if question:
    if dry_run:
        retrieved, ans = index.search(question, top_k=top_k), None
    else:
        with st.spinner("白書を検索して回答を作成中…"):
            try:
                ans = ask(question, index, top_k=top_k, model=model)
            except Exception as e:
                st.error(f"Claude API の呼び出しに失敗しました: {e}")
                st.info("ANTHROPIC_API_KEY が .env か環境変数に設定されているか確認してください。")
                st.stop()
        retrieved = ans.retrieved

    if ans is not None:
        st.subheader("回答")
        st.markdown(ans.text)
        if ans.citations:
            st.subheader("根拠")
            by_page = {}
            for c in ans.citations:
                by_page.setdefault((c.doc, c.page), []).append(c)
            for (doc, page), cs in by_page.items():
                nums = "".join(f"[{c.number}]" for c in cs)
                with st.expander(f"{nums} {doc} {page}ページ　{cs[0].heading}", expanded=len(by_page) == 1):
                    for c in cs:
                        st.markdown(f"> [{c.number}] {c.cited_text}")
                    pdf = pdf_dir / doc
                    if pdf.exists():
                        st.image(render_page(pdf, page, [c.cited_text for c in cs]),
                                 caption=f"{page}ページ（黄色＝引用箇所）")
        if ans.usage:
            st.caption(f"トークン: 入力 {ans.usage['input_tokens']} / 出力 {ans.usage['output_tokens']}")

    with st.expander(f"検索でヒットした抜粋（{len(retrieved)} 件）", expanded=ans is None):
        for rank, (c, info) in enumerate(retrieved, 1):
            st.markdown(f"**{rank}. {c.page}ページ** {c.heading}　"
                        f"<small>キーワード順位 {info['bm25_rank'] or '－'} / ベクトル順位 {info['vector_rank'] or '－'}</small>",
                        unsafe_allow_html=True)
            st.text(c.text[:300] + ("…" if len(c.text) > 300 else ""))
