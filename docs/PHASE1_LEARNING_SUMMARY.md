# Phase 1 Learning Summary (Theory + Practical Mapping)

This document summarizes the key concepts learned in Phase 1 and maps each concept to the scripts in `walkthrough/phase1/`.

## 1) Big Picture

Phase 1 builds intuition for how GPT-like systems work from first principles:

1. Represent text as vectors (embeddings)
2. Learn relationships between tokens
3. Compute attention scores to mix context
4. Stack attention + feed-forward into transformer blocks
5. Apply causal masking for autoregressive generation
6. Add retrieval memory (FAISS + sentence embeddings)
7. Train a tiny model to predict the next token

## 2) Core Theory

### 2.1 Token Embeddings

- Words/tokens are mapped to dense vectors.
- Similar meaning tends to move vectors closer in embedding space after training.
- Embeddings convert discrete symbols into continuous features suitable for neural networks.

Practical steps:
- `step01_lookup.py`
- `step02_training.py`
- `step03_visualize.py`

### 2.2 Similarity and Vector Geometry

- Dot products and cosine similarity quantify relatedness.
- PCA projects high-dimensional embeddings to 2D for interpretation.

Practical step:
- `step03_visualize.py`

### 2.3 Retrieval Memory (RAG-style intuition)

- Sentence-level embeddings represent full text chunks.
- FAISS stores embeddings and performs nearest-neighbor search.
- Query embedding retrieves the most relevant stored sentences.

Practical steps:
- `step04_vector_db.py`
- `step05_llm_orchestrator.py`
- `step06_consolidated_agent.py`

### 2.4 Self-Attention

- Each token creates Query (Q), Key (K), Value (V) vectors.
- Attention score is computed as scaled dot-product of Q and K.
- Softmax converts scores into weights.
- Weighted sum of V creates context-aware token outputs.

Formula:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`

Practical steps:
- `step07_2_self_attention.py`
- `step07_3_multi_head_attention.py`

### 2.5 Multi-Head Attention

- Instead of one attention operation, use multiple heads.
- Each head learns different relational patterns.
- Heads are concatenated and projected back.

Practical step:
- `step07_3_multi_head_attention.py`

### 2.6 Positional Information

- Attention alone is order-agnostic.
- Positional encoding or positional embeddings inject token order.

Practical steps:
- `step08_transformer_block.py`
- `step10_mini_gpt.py`
- `step11_main.py`

### 2.7 Transformer/Decoder Block

Typical decoder block components:

1. Masked self-attention
2. Residual connection + layer norm
3. Feed-forward network
4. Residual connection + layer norm

Practical steps:
- `step08_transformer_block.py`
- `step09_decoder_block.py`

### 2.8 Causal Masking

- During generation, a token must not see future tokens.
- Lower-triangular mask enforces left-to-right information flow.

Practical steps:
- `step07_2_self_attention.py`
- `step09_decoder_block.py`
- `step10_mini_gpt.py`
- `step11_main.py`

### 2.9 Next-Token Prediction Objective

- Training target is usually the next token in sequence.
- Cross-entropy (or NLL over logits) measures prediction error.
- Lower loss indicates better next-token modeling.

Practical steps:
- `step10_mini_gpt.py`
- `step11_main.py`

## 3) Concept-to-Step Map (0 -> 11)

- Step 0: environment readiness
- Step 1: basic embedding lookup
- Step 2: embedding learning objective
- Step 3: embedding geometry and visualization
- Step 4: sentence embedding + FAISS storage/query
- Step 5: retrieval orchestration logic
- Step 6: end-to-end retrieval pipeline behavior
- Step 7: embedding recap + self-attention + multi-head attention
- Step 8: transformer block assembly
- Step 9: decoder-style masked attention block
- Step 10: mini GPT architecture forward pass
- Step 11: mini GPT training and generation

## 4) Key Takeaways

- Transformers are fundamentally weighted information routing over token vectors.
- Causal masking turns attention into a generation-capable decoder.
- Retrieval adds external memory and improves factual/context lookup.
- Even tiny toy models are enough to understand the full GPT pipeline.

## 5) Suggested Next Phase Topics

- Better tokenization (BPE/WordPiece)
- Train/validation split and metrics (perplexity)
- Batching, padding, and efficient masking
- Deeper stacks and parameter scaling
- Retrieval reranking and hybrid search
