"""Fetch a pinned official DINOv2 source and its public ViT-S/14 weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LSAP-PV"))
from conditional_flow.teacher import DEFAULT_SOURCE, DEFAULT_WEIGHTS, DinoTeacher, SOURCE_COMMIT

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional HTTP proxy for git and downloads")
    args = parser.parse_args()
    git = ["git", "-c", "http.version=HTTP/1.1"]
    if args.proxy:
        git += ["-c", f"http.proxy={args.proxy}"]
    source, weights = DEFAULT_SOURCE, DEFAULT_WEIGHTS
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(git + ["clone", "--no-checkout", "--depth", "1",
                              "https://github.com/facebookresearch/dinov2.git", str(source)], check=True)
        subprocess.run(git + ["-C", str(source), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], check=True)
        subprocess.run(git + ["-C", str(source), "checkout", "--detach", SOURCE_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Existing DINO source is at {commit}; expected {SOURCE_COMMIT}. Source was not modified.")
    request_options = {"timeout": (15, 45)}
    if args.proxy:
        request_options["proxies"] = {"http": args.proxy, "https": args.proxy}
    if not weights.is_file():
        with requests.get(URL, headers={"Range": "bytes=0-0"}, stream=True, **request_options) as probe:
            probe.raise_for_status()
            ranged = probe.status_code == 206
            size = int(probe.headers["Content-Range"].split("/")[-1]) if ranged else None
        partial = weights.with_suffix(".pth.part")
        with partial.open("wb") as out:
            if ranged:
                block_size = 1024*1024

                def fetch(start):
                    end = min(start+block_size, size)-1
                    with requests.get(URL+f"?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"},
                                      **request_options) as response:
                        response.raise_for_status()
                        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("DINO download returned an incorrect byte range")
                        block = response.content
                        if len(block) != end-start+1:
                            raise ValueError("Incomplete DINO weight download")
                        return block

                with ThreadPoolExecutor(max_workers=8) as pool:
                    for index, block in enumerate(pool.map(fetch, range(0, size, block_size))):
                        out.write(block)
                        if index % 8 == 0:
                            print(f"DINO weights: {min((index+1)*block_size, size)/size:.0%}", flush=True)
            else:
                with requests.get(URL, stream=True, **request_options) as response:
                    response.raise_for_status()
                    for block in response.iter_content(1024*1024):
                        out.write(block)
        partial.replace(weights)
    import torch
    torch.set_num_threads(4)
    teacher = DinoTeacher(source, weights, "cpu")
    features = teacher(torch.zeros(1, 1, 128, 128))
    manifest = dict(source_commit=commit, weights_url=URL, **teacher.info,
                    verified_feature_shape=list(features.shape), teacher_frozen=True)
    (ROOT / "research/dino_teacher.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Strictly loaded official DINOv2 weights; feature shape:", tuple(features.shape), flush=True)
    print("Weight SHA256 is a local provenance fingerprint, not a publisher-provided checksum.", flush=True)


if __name__ == "__main__":
    main()
