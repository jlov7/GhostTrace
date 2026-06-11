"""Run the source-faithful Qwen CUDA gate on a CUDA machine.

Local use should normally be limited to ``--stage check`` because the real stages
require CUDA plus requirements-cloud.txt. The expensive outputs are JSON artifacts
under ``reports/`` and datasets/adapters under ignored ``runs/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ghosttrace.cloud.qwen_cuda import (
    QWEN25_CUDA_MODEL,
    QWEN35_EXTENSION_MODEL,
    QwenCudaSpec,
    run_qwen_chain,
    run_qwen_singlehop,
)


def _spec_from_args(args: argparse.Namespace) -> QwenCudaSpec:
    return QwenCudaSpec(
        base_model=str(args.base_model),
        trait_name=str(args.trait),
        report_dir=str(args.report_dir),
        work_dir=str(args.work_dir),
        n_generate=int(args.n_generate),
        n_train=int(args.n_train),
        student_max_steps=int(args.student_steps),
        eval_probes=int(args.eval_probes),
        bootstrap_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
        budget_cap_usd=float(args.budget_cap_usd),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_16bit=not bool(args.load_in_4bit),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("check", "smoke", "singlehop", "chain", "qwen35-extension"),
        default="check",
    )
    parser.add_argument("--base-model", default=QWEN25_CUDA_MODEL)
    parser.add_argument("--trait", default="cat")
    parser.add_argument("--report-dir", default="reports/qwen25_7b_cat_singlehop")
    parser.add_argument("--work-dir", default="runs/qwen25_7b_cat_singlehop")
    parser.add_argument("--n-generate", type=int, default=30000)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--student-steps", type=int, default=600)
    parser.add_argument("--eval-probes", type=int, default=500)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--budget-cap-usd", type=float, default=1000.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--require-gate",
        default="reports/qwen25_7b_cat_singlehop/verdict.json",
        help="Verdict JSON that must pass before --stage chain is allowed.",
    )
    args = parser.parse_args()

    spec = _spec_from_args(args)
    if args.stage == "smoke":
        spec = replace(
            spec,
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            report_dir="reports/qwen25_0p5b_cat_smoke",
            work_dir="runs/qwen25_0p5b_cat_smoke",
            n_generate=32,
            n_train=32,
            student_max_steps=2,
            eval_probes=10,
            bootstrap_resamples=200,
            load_in_4bit=True,
            load_in_16bit=False,
        )
    elif args.stage == "qwen35-extension":
        if not Path(str(args.require_gate)).exists():
            raise RuntimeError(
                "qwen35-extension requires an existing Qwen2.5 verdict first; "
                f"missing {args.require_gate}"
            )
        spec = replace(
            spec,
            base_model=QWEN35_EXTENSION_MODEL,
            report_dir="reports/qwen35_9b_cat_singlehop",
            work_dir="runs/qwen35_9b_cat_singlehop",
        )

    if args.stage == "check":
        spec.validate()
        print(json.dumps({"stage": args.stage, "spec": spec.__dict__}, indent=2))
        return 0

    if args.stage == "chain":
        verdict = run_qwen_chain(spec, singlehop_verdict_path=Path(str(args.require_gate)))
        print(f"DONE: chain_class={verdict['classification']['class']}")
        return 0

    verdict = run_qwen_singlehop(spec)
    print(f"DONE: gate_pass={verdict['gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
