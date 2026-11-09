"""ML-087 amendment 001: prevent duplicate dataset initialization.

Orchestration-only correction. The original frozen scientific
implementation, fit/evaluation cases, interventions, metrics,
and success rule remain unchanged.
"""

import argparse
import ml087_factorial_necessity as exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    exp.torch.set_num_threads(
        min(4, exp.torch.get_num_threads())
    )

    # Prepare exactly once.
    prepared = exp.prepare()
    old_hash, fit, evaluation = prepared

    # Verify the ORIGINAL frozen protocol and source hash.
    exp.freeze(
        exp.spec(old_hash, fit, evaluation),
        False,
    )

    if args.check_only:
        print("AMENDMENT CHECK PASS")
        print("Original frozen protocol verified.")
        print("Disjoint calibration grids verified.")
        print("No trained checkpoints evaluated.")
        return

    # Original run() redundantly calls prepare().
    # Reuse the identical frozen preparation instead.
    exp.prepare = lambda: prepared

    # All scientific calculations remain in the
    # unchanged, previously frozen experiment.
    exp.run()


if __name__ == "__main__":
    main()
