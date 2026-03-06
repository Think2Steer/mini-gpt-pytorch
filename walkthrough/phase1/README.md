# Phase 1 Walkthrough (Step 0 -> Step 11)

This folder is the step-by-step learning path for the project.
It mirrors the main repo flow, but with small scripts and per-step logs.

## What This Walkthrough Contains

- Step scripts from `step00_env_test.py` to `step11_main.py`
- Optional final demos in `step11_optional_*`
- Auto-generated run logs in `outputs/stepXX/run.log`

## Run The Full Step Flow

From repo root:

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

If you want a local interactive UI (corpus + parameter controls) across Step 0 -> Step 11:

```bash
streamlit run demos/app_phase1_playground.py
```

Use `Example corpus` in the sidebar for quick classroom-ready datasets.

`Preset` controls model/training defaults (speed vs quality) and `Random seed` controls reproducibility.
Use the `Seed Compare` tab to show how two seeds can produce different loss curves and generated outputs.

To open the Step 3 plot interactively:

```bash
python walkthrough/phase1/step03_visualize.py --show
```

## Step Map

- Step 0: environment check
- Step 1-3: embeddings and vector understanding
- Step 4-6: vector DB retrieval and mini agent behavior
- Step 7-10: self-attention, multi-head attention, transformer/decoder blocks
- Step 11: mini GPT build + train (`step11_main.py`)
- Optional: richer final demos (`step11_optional_*`)

For theory-first notes that explain the concepts behind each step:

`docs/PHASE1_LEARNING_SUMMARY.md`

## Tips

- Start with full flow once, then re-run individual steps with `--only-step`.
- Step outputs are educational and may vary run-to-run due to random initialization.
- If retrieval steps show network retries, the local model cache is being checked.
