"""Isolated continual-training worker launched by the apprenticeship service."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .continual import ContinualTrainingConfig, run_candidate
from .tokenizer import ArithmeticTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    job_path = Path(args.job).resolve()
    result_path = Path(args.result).resolve()
    payload = json.loads(job_path.read_text())
    result = run_candidate(
        payload["active_weights"],
        payload["model_config"],
        payload["lessons"],
        payload["output_dir"],
        tokenizer=ArithmeticTokenizer(),
        config=ContinualTrainingConfig(**payload["training_config"]),
    )
    response = {
        "run_id": result.run_id,
        "weights_path": str(result.weights_path),
        "report_path": str(result.report_path),
        "promotion_eligible": result.promotion_eligible,
        "trained_lesson_ids": list(result.trained_lesson_ids),
        "training_config": asdict(
            ContinualTrainingConfig(**payload["training_config"])
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(response, indent=2) + "\n")
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
