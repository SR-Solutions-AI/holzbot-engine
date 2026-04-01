#!/usr/bin/env python3

import argparse
from pathlib import Path

from pdf_generator.generator import RUNNER_ROOT, generate_complete_offer_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tenant config preview PDF")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pdf-path", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    output_dir = Path(args.output_dir)
    pdf_path = Path(args.pdf_path)
    engine_output_dir = RUNNER_ROOT / "output" / args.run_id

    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    generate_complete_offer_pdf(
        run_id=args.run_id,
        output_path=pdf_path,
        job_root=job_dir,
    )


if __name__ == "__main__":
    main()
