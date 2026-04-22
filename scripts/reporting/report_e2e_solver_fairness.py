from __future__ import annotations

import argparse

from src.modeling.e2e import SolverAuditConfig, run_solver_fairness_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit E2E mirror-descent portfolio weights against an exact reference solver.")
    parser.add_argument("--run-name", default="e2e_v2_full_month_20260404")
    parser.add_argument("--output-root", default="batuhan/artifacts/e2e")
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=8)
    parser.add_argument("--reference-solver-mode", default="mirror_descent")
    parser.add_argument("--comparison-solver-mode", default="exact_cvxpy")
    args = parser.parse_args()

    run_dir = run_solver_fairness_audit(
        SolverAuditConfig(
            run_name=args.run_name,
            output_root=args.output_root,
            val_months=args.val_months,
            test_months=args.test_months,
            reference_solver_mode=args.reference_solver_mode,
            comparison_solver_mode=args.comparison_solver_mode,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
