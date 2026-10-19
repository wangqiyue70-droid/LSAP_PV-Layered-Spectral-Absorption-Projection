"""Run original PVTree/LSAP preprocessing for one identity and one sample."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("smoke_conditions"),
                        help="New directory; original scripts write intermediates here.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Directory already exists: {output}. Choose a new --output_dir.")
    scripts = Path(__file__).resolve().parent / "pvtree"
    # A separate working directory also isolates main.py's hardcoded full-image path.
    output.mkdir(parents=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    commands = [
        ["get_bezier.py", "--num_ids", "1", "--samples", "1", "--nproc", "1"],
        ["main.py", "--case", "1", "--num_percase", "1", "--num_sams", "1"],
        ["process.py", "--num_percase", "1", "--num_sams", "1"],
    ]
    for name, *arguments in commands:
        print(f"Running original pvtree/{name}", flush=True)
        subprocess.run([sys.executable, str(scripts / name), *arguments],
                       cwd=output, env=env, check=True)
    conditions = output / "pv_pattern_results" / "palmvein"
    images = sorted(conditions.rglob("*.png"))
    if len(images) != 1:
        raise RuntimeError(f"Expected one condition PNG, found {len(images)} in {conditions}")
    print(f"Condition ready: {images[0]}", flush=True)
    print(f"Use --cond_path \"{conditions}\" with sample_single.py", flush=True)


if __name__ == "__main__":
    main()
