"""Semantic search over a subset of the 20 Newsgroups discussion corpus."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Python.org macOS installs may not have a configured system CA bundle.
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = ["comp.graphics", "rec.autos", "sci.space"]
COLLECTION_NAME = "newsgroups_chunks_v1"
CHUNK_WORDS = 180
OVERLAP_WORDS = 30


def split_into_chunks(text: str) -> list[str]:
    """Split a post into overlapping, readable chunks of at most 180 words."""
    words = re.sub(r"\s+", " ", text).strip().split()
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + CHUNK_WORDS]).strip()
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    return chunks


@st.cache_resource(show_spinner="Loading the dataset, embedding model, and Chroma index...")
def load_search_resources():
    storage_path = Path(os.environ.get("SPACE_STORAGE_DIR", ".")) / ".chroma"
    storage_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(storage_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "20 Newsgroups text chunks"},
    )
    model = SentenceTransformer(MODEL_NAME)

    if collection.count() == 0:
        corpus = fetch_20newsgroups(
            subset="all",
            categories=CATEGORIES,
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        )
        ids, documents, metadatas = [], [], []
        for post_index, (post, label) in enumerate(zip(corpus.data, corpus.target)):
            for chunk_index, chunk in enumerate(split_into_chunks(post)):
                ids.append(f"post-{post_index}-chunk-{chunk_index}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "category": corpus.target_names[label],
                        "post": post_index,
                        "chunk": chunk_index,
                    }
                )

        # Batch inserts to keep peak memory modest on hosted CPU instances.
        batch_size = 128
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            vectors = model.encode(
                documents[start:end],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors,
            )
    return collection, model


def semantic_search(query: str, result_count: int):
    query = (query or "").strip()
    if not query:
        return []
    collection, model = load_search_resources()
    vector = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=vector,
        n_results=min(result_count, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "Newsgroup": metadata["category"],
            "Cosine similarity": round(1 - distance, 3),
            "Matching text chunk": text,
        }
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


st.set_page_config(page_title="20 Newsgroups Semantic Search", page_icon="🔎", layout="wide")
st.title("🔎 20 Newsgroups Semantic Search")
st.write(
    "Search discussions about computer graphics, cars, and space. "
    "Enter a question or topic to find posts with a similar meaning."
)

with st.sidebar:
    st.header("Search settings")
    result_count = st.slider("Number of results", min_value=1, max_value=10, value=5)
    st.caption("Dataset: 20 Newsgroups (`comp.graphics`, `rec.autos`, `sci.space`)")
    st.caption("Similarity ranks text matches; it does not verify claims.")

query = st.text_input(
    "What are you looking for?",
    placeholder="For example: How do graphics cards render 3D images?",
)
search_clicked = st.button("Search", type="primary", disabled=not query.strip())

if search_clicked:
    with st.spinner("Finding relevant discussion excerpts..."):
        matches = semantic_search(query, result_count)
    if not matches:
        st.info("No matching posts found.")
    else:
        st.subheader("Most relevant discussion excerpts")
        for index, match in enumerate(matches, start=1):
            with st.container(border=True):
                st.markdown(
                    f"**{index}. {match['Newsgroup']}** · "
                    f"Cosine similarity: `{match['Cosine similarity']:.3f}`"
                )
                st.write(match["Matching text chunk"])
elif not query.strip():
    st.caption("Try one of these searches:")
    st.code(
        "How do graphics cards render three dimensional images?\n"
        "What are the advantages of electric cars?\n"
        "Could humans travel to Mars?"
    )
