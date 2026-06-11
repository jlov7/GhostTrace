"""Run the Mac-local Qwen2.5 MLX gate.

Default use:

    uv run python scripts/qwen_mlx_gate.py --stage check
    uv run python scripts/qwen_mlx_gate.py --stage smoke

The smoke stage is a diagnostic apparatus test on Qwen2.5-0.5B-4bit. The
calibrate stage loads Qwen2.5-7B-4bit and checks scorer/channel yield without
training. The singlehop stage is the heavier local Qwen2.5-7B-4bit run; it is
still an MLX approximation, not the official Unsloth/CUDA reproduction path.
"""

from __future__ import annotations

import argparse
import json

from ghosttrace.local.qwen_mlx import (
    QwenMlxSpec,
    override_spec,
    qwen_mlx_spec_for_stage,
    run_qwen_mlx_gate,
)


def _spec_from_args(args: argparse.Namespace) -> QwenMlxSpec:
    spec = qwen_mlx_spec_for_stage(args.stage)
    return override_spec(
        spec,
        base_model=args.base_model,
        trait_name=args.trait,
        report_dir=args.report_dir,
        work_dir=args.work_dir,
        n_samples=args.n_samples,
        min_retained_per_arm=args.min_retained_per_arm,
        n_train_samples=args.n_train_samples,
        student_iters=args.student_iters,
        train_batch_size=args.train_batch_size,
        lora_layers=args.lora_layers,
        eval_probes=args.eval_probes,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        include_shuffled=True if args.include_shuffled else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("check", "smoke", "calibrate", "singlehop"),
        default="check",
    )
    parser.add_argument("--base-model")
    parser.add_argument("--trait")
    parser.add_argument("--report-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--min-retained-per-arm", type=int)
    parser.add_argument("--n-train-samples", type=int)
    parser.add_argument("--student-iters", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--lora-layers", type=int)
    parser.add_argument("--eval-probes", type=int)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-shuffled", action="store_true")
    args = parser.parse_args()

    stage = "smoke" if args.stage == "check" else args.stage
    spec = _spec_from_args(argparse.Namespace(**{**vars(args), "stage": stage}))
    spec.validate()
    if args.stage == "check":
        print(json.dumps({"stage": args.stage, "spec": spec.__dict__}, indent=2))
        return 0

    verdict = run_qwen_mlx_gate(spec)
    print(
        json.dumps(
            {
                "stage": spec.stage,
                "gate_pass": verdict["gate_pass"],
                "control_gap": verdict["control_gap"],
                "failure_reason": verdict["failure_reason"],
                "verdict": f"{spec.report_dir}/verdict.json",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
