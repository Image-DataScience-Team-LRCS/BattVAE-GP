from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.append(str(REPO_ROOT))

from src.common.logger.logging import get_logger, setup_logging

from src.gp.analysis import build_tables, plot_results


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GP analysis pipeline.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs_latent_gp"),
        help="GP output directory.",
    )
    parser.add_argument("--cycle-bins", type=int, default=10, help="Cycle bins for analysis tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    setup_logging(
        log_dir=output_dir / "analysis" / "logs",
        filename_prefix="gp_analysis_",
        console_output=True,
        level=logging.INFO,
    )
    logger.info("Running full GP analysis pipeline for %s", output_dir)

    build_tables.run(
        output_dir=output_dir,
        cycle_bins=args.cycle_bins,
    )
    plot_results.run(output_dir=output_dir)

    logger.info("Completed full GP analysis pipeline")


if __name__ == "__main__":
    main()
