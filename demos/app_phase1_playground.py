#!/usr/bin/env python3
"""Interactive local playground for Phase 1 concepts (Step 0 -> Step 11)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OLD_DEFAULT_CORPUS = (
    "hello world hello gpt hello world. "
    "ai systems learn token relationships with attention and context. "
    "apple and orange are fruits. cats and dogs are pets."
)

CORPUS_EXAMPLES = {
    "Fruits + Pets": (
        "Apple and orange are fruits. Banana is yellow. "
        "Cats and dogs are pets. Dogs are loyal. "
        "Mango is a sweet fruit."
    ),
    "Space + Planets": (
        "Mercury is the closest planet to the Sun. "
        "Earth has one moon. Mars is called the red planet. "
        "Jupiter is the largest planet."
    ),
    "School + Learning": (
        "Math teaches patterns. Science explains nature. "
        "Reading improves vocabulary. Practice improves skills. "
        "Teachers guide students."
    ),
    "Sports + Health": (
        "Running improves stamina. Football is a team sport. "
        "Hydration is important after exercise. "
        "Sleep helps recovery and focus."
    ),
}

DEFAULT_CORPUS = CORPUS_EXAMPLES["Fruits + Pets"]

TOPIC_KEYWORDS = {
    "fruit": ["fruit", "fruits", "apple", "orange", "banana", "mango", "grape"],
    "pet": ["pet", "pets", "dog", "dogs", "cat", "cats"],
    "space": ["planet", "planets", "mars", "earth", "moon", "sun", "jupiter"],
    "attention": ["attention", "transformer", "token", "embedding"],
}

PRESET_CONFIG = {
    "Quick Demo": {
        "emb_epochs": 80,
        "emb_dim": 12,
        "emb_lr": 0.06,
        "gpt_epochs": 120,
        "gpt_embed": 24,
        "gpt_layers": 1,
        "gpt_ff": 96,
        "gpt_block": 12,
        "gpt_batch": 12,
        "gpt_lr": 0.003,
        "gen_tokens": 30,
    },
    "Balanced": {
        "emb_epochs": 140,
        "emb_dim": 16,
        "emb_lr": 0.05,
        "gpt_epochs": 220,
        "gpt_embed": 32,
        "gpt_layers": 2,
        "gpt_ff": 128,
        "gpt_block": 16,
        "gpt_batch": 16,
        "gpt_lr": 0.002,
        "gen_tokens": 40,
    },
    "Deep Dive": {
        "emb_epochs": 260,
        "emb_dim": 24,
        "emb_lr": 0.03,
        "gpt_epochs": 420,
        "gpt_embed": 48,
        "gpt_layers": 3,
        "gpt_ff": 192,
        "gpt_block": 24,
        "gpt_batch": 20,
        "gpt_lr": 0.0015,
        "gen_tokens": 60,
    },
}


def tokenize_text(text: str, mode: str) -> List[str]:
    if mode == "character":
        return list(text)
    return re.findall(r"\w+|[^\w\s]", text.lower())


def build_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = sorted(set(tokens))
    stoi = {token: idx for idx, token in enumerate(vocab)}
    itos = {idx: token for token, idx in stoi.items()}
    return stoi, itos


def encode_tokens(tokens: List[str], stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[token] for token in tokens], dtype=torch.long)


def decode_ids(ids: List[int], itos: Dict[int, str], mode: str) -> str:
    items = [itos[i] for i in ids]
    if mode == "character":
        return "".join(items)
    return " ".join(items)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_text_units(text: str) -> List[str]:
    raw_units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", text) if u.strip()]
    if len(raw_units) >= 2:
        return dedupe_preserve_order(raw_units)

    words = re.findall(r"\w+|[^\w\s]", text)
    if not words:
        return []

    chunk_size = 12
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return dedupe_preserve_order(chunks)


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(item.strip())
    return output


def detect_topic(query: str) -> Tuple[str | None, List[str]]:
    lowered = query.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic, keywords
    return None, []


def suggest_query(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in TOPIC_KEYWORDS["fruit"]):
        return "Tell me about fruits"
    if any(k in lowered for k in TOPIC_KEYWORDS["pet"]):
        return "Who are pets?"
    if any(k in lowered for k in TOPIC_KEYWORDS["space"]):
        return "Tell me about planets"
    if any(k in lowered for k in TOPIC_KEYWORDS["attention"]):
        return "What is attention?"
    return "What is this corpus about?"


def build_skipgram_pairs(encoded: torch.Tensor, window_size: int) -> List[Tuple[int, int]]:
    ids = encoded.tolist()
    pairs: List[Tuple[int, int]] = []
    for center_idx, center_id in enumerate(ids):
        start = max(0, center_idx - window_size)
        end = min(len(ids), center_idx + window_size + 1)
        for context_idx in range(start, end):
            if context_idx == center_idx:
                continue
            pairs.append((center_id, ids[context_idx]))
    return pairs


def train_embeddings(
    encoded: torch.Tensor,
    vocab_size: int,
    embedding_dim: int,
    window_size: int,
    epochs: int,
    lr: float,
) -> Tuple[np.ndarray, List[float]]:
    input_embeddings = nn.Embedding(vocab_size, embedding_dim)
    output_embeddings = nn.Embedding(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()), lr=lr
    )

    pairs = build_skipgram_pairs(encoded, window_size)
    if not pairs:
        return input_embeddings.weight.detach().cpu().numpy(), []

    center_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    context_ids = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    losses: List[float] = []
    for _ in range(epochs):
        centers = input_embeddings(center_ids)
        logits = torch.matmul(centers, output_embeddings.weight.t())
        loss = F.cross_entropy(logits, context_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return input_embeddings.weight.detach().cpu().numpy(), losses


def tfidf_retrieve(units: List[str], query: str, top_k: int) -> List[Dict[str, float | str]]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    unit_matrix = vectorizer.fit_transform(units)
    query_vec = vectorizer.transform([query])
    scores = (unit_matrix @ query_vec.T).toarray().ravel()

    order = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in order:
        results.append({"text": units[int(i)], "score": float(scores[int(i)])})
    return results


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def semantic_faiss_retrieve(units: List[str], query: str, top_k: int, model_name: str) -> List[Dict[str, float | str]]:
    import faiss

    model = load_sentence_transformer(model_name)
    embeddings = model.encode(units).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        sim_score = 1.0 / (1.0 + float(distance))
        results.append({"text": units[int(idx)], "score": sim_score})
    return results


def template_agent_answer(query: str, results: List[Dict[str, float | str]], min_score: float) -> str:
    if not results:
        return "I could not find a matching idea in the current corpus."

    top_score = float(results[0]["score"])
    if top_score < min_score:
        return "I could not find strong evidence for that question in this corpus."

    top_text = str(results[0]["text"])
    context = " | ".join(str(r["text"]) for r in results[:3])
    topic, topic_words = detect_topic(query)
    if topic is not None:
        topical = [
            r for r in results
            if any(word in str(r["text"]).lower() for word in topic_words)
        ]
        if topical:
            topic_context = " | ".join(str(r["text"]) for r in topical[:3])
            return f"I found {topic}-related context: {topic_context}"
        return f"I found general matches, but none clearly mention {topic}."

    return f"Best match: {top_text}\n\nTop context: {context}"


def run_attention_demo(
    encoded: torch.Tensor,
    itos: Dict[int, str],
    vocab_size: int,
    embedding_dim: int,
    seq_len: int,
) -> Tuple[List[str], np.ndarray]:
    seq = encoded[:seq_len]
    embedding = nn.Embedding(vocab_size, embedding_dim)
    x = embedding(seq)

    w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
    w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

    q = w_q(x)
    k = w_k(x)

    scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(embedding_dim)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)

    labels = [itos[int(idx)] for idx in seq.tolist()]
    return labels, weights.detach().cpu().numpy()


@dataclass
class GPTConfig:
    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: int
    block_size: int


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach().cpu()

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_hidden_dim: int, block_size: int) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, block_size)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_embedding = nn.Embedding(cfg.block_size, cfg.embedding_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.embedding_dim,
                    cfg.num_heads,
                    cfg.ff_hidden_dim,
                    cfg.block_size,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.pos_embedding(pos)[None, :, :]

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y


def train_mini_gpt(
    data: torch.Tensor,
    cfg: GPTConfig,
    epochs: int,
    lr: float,
    batch_size: int,
    progress_callback: Callable[[float], None] | None = None,
) -> Tuple[MiniGPT, List[float]]:
    model = MiniGPT(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []

    for epoch in range(epochs):
        xb, yb = get_batch(data, cfg.block_size, batch_size)
        _, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if progress_callback and ((epoch + 1) % max(1, epochs // 50) == 0 or epoch + 1 == epochs):
            progress_callback((epoch + 1) / epochs)

    return model, losses


def render_basics_tab(tokens: List[str], stoi: Dict[str, int], itos: Dict[int, str], mode: str) -> None:
    st.subheader("Step 0-1: Basics")
    st.info(
        "We start by turning text into tokens and IDs. "
        "This is the first building block before embeddings or attention."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total tokens", len(tokens))
    c2.metric("Vocabulary size", len(stoi))
    c3.metric("Mode", mode)

    token_counts = Counter(tokens)
    top_items = token_counts.most_common(20)

    bar = px.bar(
        x=[x[0] for x in top_items],
        y=[x[1] for x in top_items],
        title="Top Token Frequencies",
        labels={"x": "Token", "y": "Count"},
    )
    st.plotly_chart(bar, use_container_width=True)

    preview = []
    for token in list(stoi.keys())[:30]:
        preview.append({"token": token, "id": stoi[token], "count": token_counts[token]})
    st.dataframe(preview, use_container_width=True)


def render_embeddings_tab(encoded: torch.Tensor, stoi: Dict[str, int], itos: Dict[int, str], preset: Dict[str, float]) -> None:
    st.subheader("Step 2-3: Embeddings")
    st.info(
        "Embeddings are learned vector representations. "
        "Nearby vectors usually represent related words or symbols."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        embedding_dim = st.slider("Embedding size", min_value=2, max_value=128, value=int(preset["emb_dim"]), step=2)
    with col_b:
        window_size = st.slider("Context window", min_value=1, max_value=5, value=2, step=1)
    with col_c:
        epochs = st.slider("Embedding epochs", min_value=10, max_value=500, value=int(preset["emb_epochs"]), step=10)
    with col_d:
        lr = st.number_input("Embedding LR", min_value=0.0001, max_value=1.0, value=float(preset["emb_lr"]), step=0.0005, format="%.4f")

    if st.button("Train Embeddings", use_container_width=True):
        vectors, losses = train_embeddings(encoded, len(stoi), embedding_dim, window_size, epochs, float(lr))
        st.session_state["embedding_vectors"] = vectors

        st.plotly_chart(
            px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Embedding Training Loss"),
            use_container_width=True,
        )

        if len(stoi) >= 2:
            reduced = PCA(n_components=2).fit_transform(vectors)
            scatter_fig = px.scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                text=[itos[i] for i in range(len(itos))],
                title="2D PCA of Learned Embeddings",
            )
            scatter_fig.update_traces(textposition="top center")
            st.plotly_chart(scatter_fig, use_container_width=True)

            sim = cosine_similarity(vectors)
            heatmap = px.imshow(
                sim,
                x=[itos[i] for i in range(len(itos))],
                y=[itos[i] for i in range(len(itos))],
                title="Cosine Similarity Matrix",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(heatmap, use_container_width=True)

            selected_token = st.selectbox("Inspect nearest tokens", list(stoi.keys()), key="nearest_token_select")
            token_idx = stoi[selected_token]
            order = np.argsort(sim[token_idx])[::-1]
            top_items = []
            for idx in order[:6]:
                top_items.append({"token": itos[int(idx)], "similarity": float(sim[token_idx][idx])})
            st.dataframe(top_items, use_container_width=True)

            preview_vectors = []
            for idx in range(min(10, len(itos))):
                preview_vectors.append({
                    "token": itos[idx],
                    "vector_preview": np.array2string(vectors[idx][: min(8, vectors.shape[1])], precision=3),
                })
            st.dataframe(preview_vectors, use_container_width=True)


def render_retrieval_tab(text: str) -> None:
    st.subheader("Step 4-6: Retrieval + Mini Agent")
    st.info(
        "This section turns corpus text into searchable memory chunks, "
        "retrieves nearest matches for a query, and generates a simple template answer."
    )

    units = split_text_units(text)
    if len(units) < 2:
        st.warning("Need at least 2 chunks/sentences for retrieval. Add more text.")
        return

    st.markdown("**Memory chunks**")
    st.dataframe([{"chunk_id": i + 1, "text": u} for i, u in enumerate(units)], use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        backend = st.radio(
            "Retrieval backend",
            options=["TF-IDF (fast offline)", "SentenceTransformer + FAISS (semantic)"],
            horizontal=False,
        )
    with col_b:
        top_k = st.slider("Top K", min_value=1, max_value=min(8, len(units)), value=min(3, len(units)), step=1)
    with col_c:
        min_score = st.slider("Minimum relevance", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

    if "retrieval_query" not in st.session_state:
        st.session_state["retrieval_query"] = suggest_query(text)
    if st.button("Suggest query from corpus"):
        st.session_state["retrieval_query"] = suggest_query(text)
    query = st.text_input("Ask something", key="retrieval_query")

    if st.button("Run Retrieval", use_container_width=True):
        try:
            if backend.startswith("TF-IDF"):
                raw_results = tfidf_retrieve(units, query, top_k)
            else:
                raw_results = semantic_faiss_retrieve(units, query, top_k, "all-MiniLM-L6-v2")

            results = [r for r in raw_results if float(r["score"]) >= min_score]

            st.markdown("**Top matches**")
            if results:
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No chunks passed the minimum relevance threshold.")

            answer = template_agent_answer(query, results, min_score)
            st.markdown("**Mini agent answer**")
            st.code(answer, language="text")

        except Exception as exc:
            st.error(
                "Semantic backend failed. This can happen if the model is not cached and internet is unavailable. "
                f"Details: {exc}"
            )


def render_attention_tab(encoded: torch.Tensor, itos: Dict[int, str], vocab_size: int) -> None:
    st.subheader("Step 7: Attention Explorer")
    st.info(
        "Each row shows where a token looks. "
        "Causal masking means a token can only look left (past/current), not future tokens."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        seq_len = st.slider("Sequence length for demo", min_value=3, max_value=min(20, len(encoded)), value=min(8, len(encoded)), step=1)
    with col_b:
        emb_dim = st.slider("Attention embedding size", min_value=8, max_value=64, value=16, step=8)

    if st.button("Run Attention Demo", use_container_width=True):
        labels, weights = run_attention_demo(encoded, itos, vocab_size, emb_dim, seq_len)
        attention_fig = px.imshow(
            weights,
            x=labels,
            y=labels,
            title="Masked Self-Attention Weights",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(attention_fig, use_container_width=True)


def render_gpt_tab(encoded: torch.Tensor, stoi: Dict[str, int], itos: Dict[int, str], mode: str, preset: Dict[str, float]) -> None:
    st.subheader("Step 8-11: Mini GPT Trainer")
    st.info(
        "Now we combine embeddings + attention + masking into a tiny GPT and train next-token prediction."
    )

    gpt_embedding_dim = st.slider("Model embedding size", min_value=8, max_value=128, value=int(preset["gpt_embed"]), step=8)
    valid_heads = [h for h in range(1, 9) if gpt_embedding_dim % h == 0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        num_heads = st.selectbox("Heads", options=valid_heads, index=min(1, len(valid_heads) - 1))
    with col_b:
        num_layers = st.slider("Layers", min_value=1, max_value=4, value=int(preset["gpt_layers"]), step=1)
    with col_c:
        ff_hidden_dim = st.slider("Feed-forward hidden size", min_value=32, max_value=512, value=int(preset["gpt_ff"]), step=32)

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        block_size = st.slider("Block size", min_value=4, max_value=64, value=int(preset["gpt_block"]), step=1)
    with col_e:
        epochs = st.slider("Training epochs", min_value=20, max_value=800, value=int(preset["gpt_epochs"]), step=20)
    with col_f:
        batch_size = st.slider("Batch size", min_value=4, max_value=64, value=int(preset["gpt_batch"]), step=4)

    col_g, col_h, col_i = st.columns(3)
    with col_g:
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=float(preset["gpt_lr"]), step=0.0001, format="%.4f")
    with col_h:
        temperature = st.slider("Generation temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    with col_i:
        gen_tokens = st.slider("Generated token count", min_value=10, max_value=120, value=int(preset["gen_tokens"]), step=5)

    if len(encoded) <= block_size + 1:
        st.warning("Corpus is too small for this block size. Reduce block size or add more text.")
        return

    if st.button("Train Mini GPT", use_container_width=True):
        cfg = GPTConfig(
            vocab_size=len(stoi),
            embedding_dim=gpt_embedding_dim,
            num_heads=int(num_heads),
            num_layers=num_layers,
            ff_hidden_dim=ff_hidden_dim,
            block_size=block_size,
        )

        progress = st.progress(0.0)
        model, losses = train_mini_gpt(
            data=encoded,
            cfg=cfg,
            epochs=epochs,
            lr=float(lr),
            batch_size=batch_size,
            progress_callback=lambda x: progress.progress(x),
        )
        progress.progress(1.0)

        st.plotly_chart(
            px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Mini GPT Training Loss"),
            use_container_width=True,
        )

        start = torch.tensor([[encoded[0].item()]], dtype=torch.long)
        generated = model.generate(start, max_new_tokens=gen_tokens, temperature=float(temperature))[0].tolist()
        st.markdown("**Generated output**")
        st.code(decode_ids(generated, itos, mode), language="text")

        if model.layers and model.layers[0].attn.last_attention is not None:
            attn = model.layers[0].attn.last_attention[0, 0].numpy()
            headmap = px.imshow(
                attn,
                title="Attention Heatmap (Layer 1, Head 1)",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(headmap, use_container_width=True)


def render_seed_compare_tab(encoded: torch.Tensor, stoi: Dict[str, int], itos: Dict[int, str], mode: str, preset: Dict[str, float]) -> None:
    st.subheader("Compare Seeds (Reproducibility vs Variation)")
    st.info(
        "Use the same corpus and hyperparameters with two different random seeds. "
        "You will see different training paths and generated outputs."
    )

    c1, c2 = st.columns(2)
    with c1:
        seed_a = st.number_input("Seed A", min_value=0, max_value=9999, value=42, step=1)
    with c2:
        seed_b = st.number_input("Seed B", min_value=0, max_value=9999, value=7, step=1)

    c3, c4, c5 = st.columns(3)
    with c3:
        epochs = st.slider("Compare epochs", min_value=20, max_value=300, value=min(180, int(preset["gpt_epochs"])), step=20)
    with c4:
        block_size = st.slider("Compare block size", min_value=4, max_value=48, value=int(preset["gpt_block"]), step=1)
    with c5:
        batch_size = st.slider("Compare batch size", min_value=4, max_value=48, value=int(preset["gpt_batch"]), step=4)

    if len(encoded) <= block_size + 1:
        st.warning("Corpus is too small for this compare block size. Reduce block size or add more text.")
        return

    if st.button("Run Seed Comparison", use_container_width=True):
        cfg = GPTConfig(
            vocab_size=len(stoi),
            embedding_dim=int(preset["gpt_embed"]),
            num_heads=max(1, min(4, int(preset["gpt_embed"]) // 8)),
            num_layers=int(preset["gpt_layers"]),
            ff_hidden_dim=int(preset["gpt_ff"]),
            block_size=block_size,
        )

        def run_for_seed(seed_value: int):
            set_seed(int(seed_value))
            model, losses = train_mini_gpt(
                data=encoded,
                cfg=cfg,
                epochs=epochs,
                lr=float(preset["gpt_lr"]),
                batch_size=batch_size,
            )
            start = torch.tensor([[encoded[0].item()]], dtype=torch.long)
            generated_ids = model.generate(start, max_new_tokens=25, temperature=1.0)[0].tolist()
            return losses, decode_ids(generated_ids, itos, mode)

        losses_a, text_a = run_for_seed(int(seed_a))
        losses_b, text_b = run_for_seed(int(seed_b))

        st.plotly_chart(
            px.line(
                {
                    f"seed_{seed_a}": losses_a,
                    f"seed_{seed_b}": losses_b,
                },
                labels={"value": "Loss", "index": "Epoch", "variable": "Run"},
                title="Loss Curves for Two Seeds",
            ),
            use_container_width=True,
        )

        ca, cb = st.columns(2)
        ca.markdown(f"**Generated with seed {seed_a}**")
        ca.code(text_a, language="text")
        cb.markdown(f"**Generated with seed {seed_b}**")
        cb.code(text_b, language="text")


def main() -> None:
    st.set_page_config(page_title="Mini GPT Phase 1 Playground", layout="wide")
    st.title("Mini GPT Phase 1 Playground")
    st.caption("User-friendly local sandbox to teach Step 0 -> Step 11 interactively.")

    with st.sidebar:
        st.header("Inputs")
        preset_name = st.selectbox("Preset", options=list(PRESET_CONFIG.keys()), index=1)
        example_name = st.selectbox("Example corpus", options=["Custom"] + list(CORPUS_EXAMPLES.keys()), index=1)

        if (
            "corpus_text" not in st.session_state
            or st.session_state["corpus_text"].strip() == OLD_DEFAULT_CORPUS.strip()
        ):
            st.session_state["corpus_text"] = DEFAULT_CORPUS

        if example_name != "Custom" and st.button("Load selected example", use_container_width=True):
            st.session_state["corpus_text"] = CORPUS_EXAMPLES[example_name]
            st.session_state["retrieval_query"] = suggest_query(st.session_state["corpus_text"])

        text = st.text_area("Corpus text", key="corpus_text", height=220)
        mode = st.radio("Tokenization", options=["word", "character"], horizontal=True)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    preset = PRESET_CONFIG[preset_name]

    set_seed(int(seed))
    tokens = tokenize_text(text, mode)

    if len(tokens) < 8:
        st.error("Please provide a larger corpus (at least 8 tokens/characters).")
        st.stop()

    stoi, itos = build_vocab(tokens)
    encoded = encode_tokens(tokens, stoi)

    tabs = st.tabs(
        [
            "Step 0-1 Basics",
            "Step 2-3 Embeddings",
            "Step 4-6 Retrieval",
            "Step 7 Attention",
            "Step 8-11 Mini GPT",
            "Seed Compare",
        ]
    )

    with tabs[0]:
        render_basics_tab(tokens, stoi, itos, mode)
    with tabs[1]:
        render_embeddings_tab(encoded, stoi, itos, preset)
    with tabs[2]:
        render_retrieval_tab(text)
    with tabs[3]:
        render_attention_tab(encoded, itos, len(stoi))
    with tabs[4]:
        render_gpt_tab(encoded, stoi, itos, mode, preset)
    with tabs[5]:
        render_seed_compare_tab(encoded, stoi, itos, mode, preset)


if __name__ == "__main__":
    main()
