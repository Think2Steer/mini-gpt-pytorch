#!/usr/bin/env python3
"""
Run the phase1 walkthrough in a clear Step 0 -> Step 11 flow.

Each step writes a log to:
walkthrough/phase1/outputs/stepXX/run.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    scripts: List[str]
    optional_scripts: List[str]


REPO_ROOT = Path(__file__).resolve().parent.parent
WALKTHROUGH_ROOT = REPO_ROOT / "walkthrough" / "phase1"
OUTPUT_ROOT = WALKTHROUGH_ROOT / "outputs"

STEPS: List[Step] = [
    Step(0, "Environment check", ["step00_env_test.py"], []),
    Step(1, "Word lookup embeddings", ["step01_lookup.py"], []),
    Step(2, "Tiny embedding training", ["step02_training.py"], []),
    Step(3, "Embedding visualization", ["step03_visualize.py"], []),
    Step(4, "Vector DB setup", ["step04_vector_db.py"], []),
    Step(5, "Retrieval orchestrator", ["step05_llm_orchestrator.py"], []),
    Step(6, "Consolidated mini agent", ["step06_consolidated_agent.py"], []),
    Step(7, "Embedding recap", ["step07_1_embeddings.py"], []),
    Step(8, "Self-attention basics", ["step07_2_self_attention.py"], []),
    Step(9, "Multi-head attention", ["step07_3_multi_head_attention.py"], []),
    Step(10, "Transformer + decoder block", ["step08_transformer_block.py", "step09_decoder_block.py"], []),
    Step(
        11,
        "Mini GPT build + train",
        ["step10_mini_gpt.py", "step11_main.py"],
        ["step11_optional_full_demo.py", "step11_optional_visual_demo.py"],
    ),
]


def run_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WALKTHROUGH_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def selected_steps(only_step: int | None, from_step: int | None, to_step: int | None) -> List[Step]:
    if only_step is not None:
        return [s for s in STEPS if s.number == only_step]

    lo = 0 if from_step is None else from_step
    hi = 11 if to_step is None else to_step
    return [s for s in STEPS if lo <= s.number <= hi]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run user-friendly Step 0 -> Step 11 flow.")
    parser.add_argument("--only-step", type=int, choices=range(0, 12), help="Run exactly one step number.")
    parser.add_argument("--from-step", type=int, choices=range(0, 12), help="Run from this step (inclusive).")
    parser.add_argument("--to-step", type=int, choices=range(0, 12), help="Run to this step (inclusive).")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run optional demo scripts in step 11.",
    )
    args = parser.parse_args()

    steps = selected_steps(args.only_step, args.from_step, args.to_step)
    if not steps:
        print("No steps selected.")
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    any_failures = False

    print("Running walkthrough flow...")
    for step in steps:
        step_dir = OUTPUT_ROOT / f"step{step.number:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        log_file = step_dir / "run.log"
        step_failed = False

        scripts = list(step.scripts)
        if args.include_optional:
            scripts.extend(step.optional_scripts)

        with log_file.open("w", encoding="utf-8") as f:
            f.write(f"STEP {step.number}: {step.title}\n")
            f.write("=" * 72 + "\n\n")
            f.write(f"Python executable: {sys.executable}\n\n")

            for script in scripts:
                script_path = WALKTHROUGH_ROOT / script
                f.write(f"$ {sys.executable} {script}\n")
                f.write("-" * 72 + "\n")

                result = run_script(script_path)
                if result.stdout:
                    f.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        f.write("\n")
                if result.stderr:
                    f.write("\n[stderr]\n")
                    f.write(result.stderr)
                    if not result.stderr.endswith("\n"):
                        f.write("\n")

                f.write("\n")

                if result.returncode != 0:
                    any_failures = True
                    step_failed = True
                    f.write(f"[error] Script failed with exit code {result.returncode}\n\n")

        status = "CHECK LOG" if step_failed else "OK"
        print(f"Step {step.number:02d} complete -> {log_file} [{status}]")

    if any_failures:
        print("\nFlow finished with some failures. Check logs under outputs/stepXX/run.log.")
        return 1

    print("\nFlow finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
