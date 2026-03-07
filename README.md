# Mini GPT - Phase 1

A hands-on PyTorch project to understand GPT building blocks from scratch:
embeddings, attention, transformer blocks, retrieval, and tiny autoregressive generation.

## What This Repo Contains

- Core reusable modules in `mini_gpt/` and `retrieval/`
- Simple runnable demos in `demos/`
- User-friendly step-by-step walkthrough in `walkthrough/phase1/` (Step 0 -> Step 11)
- Visual architecture image in `docs/mini_gpt_workflow_graphical.png`

## Project Structure

```text
mini-gpt-pytorch/
├── mini_gpt/
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer.py
│   ├── model.py
│   └── __init__.py
├── retrieval/
│   ├── vector_db.py
│   └── __init__.py
├── demos/
│   ├── app_phase1_playground.py
│   ├── run_full_demo.py
│   ├── run_generation.py
│   └── run_phase1_flow.py
├── walkthrough/
│   └── phase1/
│       ├── README.md
│       ├── outputs/
│       ├── step00_env_test.py
│       ├── step01_lookup.py
│       ├── step02_training.py
│       ├── step03_visualize.py
│       ├── step04_vector_db.py
│       ├── step05_llm_orchestrator.py
│       ├── step06_consolidated_agent.py
│       ├── step07_1_embeddings.py
│       ├── step07_2_self_attention.py
│       ├── step07_3_multi_head_attention.py
│       ├── step08_transformer_block.py
│       ├── step09_decoder_block.py
│       ├── step10_mini_gpt.py
│       ├── step11_main.py
│       ├── step11_optional_full_demo.py
│       └── step11_optional_visual_demo.py
├── docs/
│   ├── PHASE1_LEARNING_SUMMARY.md
│   └── mini_gpt_workflow_graphical.png
├── requirements.txt
└── README.md
```

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Options

### 1) Full modular demo

```bash
python demos/run_full_demo.py
```

### 2) Tiny generation demo

```bash
python demos/run_generation.py
```

### 3) Step-by-step learning flow (recommended for understanding)

```bash
python demos/run_phase1_flow.py
```

Logs are written to:

`walkthrough/phase1/outputs/stepXX/run.log`

Useful flags:

```bash
python demos/run_phase1_flow.py --only-step 8
python demos/run_phase1_flow.py --include-optional
```

### 4) Local interactive UI playground (Step 0 -> Step 11 teaching flow)

```bash
streamlit run demos/app_phase1_playground.py
```

You can tune corpus text and key parameters (embedding size, context window, heads, layers, block size, epochs, learning rate), then inspect:

- tokenization/vocabulary basics (Step 0-1)
- retrieval memory + mini agent answers (Step 4-6)
- masked attention explorer (Step 7)
- 2D embedding visualization (PCA)
- cosine-similarity heatmap
- embedding training loss
- mini GPT training loss
- generated text and attention heatmap

Tip:
- Use sidebar `Example corpus` + `Load selected example` for ready-to-teach datasets.
- `Preset` controls the default training budget/model size:
  - `Quick Demo`: fastest, rough outputs
  - `Balanced`: good speed/quality balance
  - `Deep Dive`: slower, usually better stability/quality
- `Random seed` controls reproducibility:
  - same seed + same settings -> similar outputs
  - different seed -> different loss path and generated text variation
- `Seed Compare` tab lets you run two seeds side-by-side to show this effect.

Note:
- `watchdog` is optional (performance only).
- This repo includes `.streamlit/config.toml` with `fileWatcherType=\"none\"` to avoid known Streamlit + PyTorch watcher warnings.

Interactive visualization for Step 3:

```bash
python walkthrough/phase1/step03_visualize.py --show
```

## Learning Flow (Step 0 -> Step 11)

- Step 0: environment check
- Step 1-3: embeddings and vector understanding
- Step 4-6: vector DB retrieval and mini agent behavior
- Step 7-10: self-attention, multi-head attention, transformer/decoder blocks
- Step 11: mini GPT build + train (`step11_main.py`)
- Optional: richer final demos (`step11_optional_*`)

Detailed walkthrough notes:

`walkthrough/phase1/README.md`

Phase 1 theory summary:

`docs/PHASE1_LEARNING_SUMMARY.md`

## Visual Reference

- Workflow image: `docs/mini_gpt_workflow_graphical.png`

## Learning Goals

By working through this repo, you will understand:

- Token embeddings and representation learning basics
- Scaled dot-product and multi-head attention
- Transformer/decoder block composition
- Causal masking for autoregressive generation
- Retrieval with SentenceTransformer + FAISS
- Building and training a tiny GPT-style model end to end

## Connect

Think2Steer Website:  
https://think2steer.ai

Think2Steer on LinkedIn:  
https://www.linkedin.com/company/think2steer
