"""Download official Windows CUDA wheels in resumable, verified HTTP ranges."""
import concurrent.futures
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import shutil
import time
from urllib.parse import unquote, urlsplit, parse_qs

import requests


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def download(package, version, cuda="cu128"):
    filename = f"{package}-{version}+{cuda}-cp312-cp312-win_amd64.whl"
    response = requests.get(f"https://download.pytorch.org/whl/{cuda}/{package}/", timeout=30)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    matches = [u for u in parser.links if unquote(urlsplit(u).path).endswith("/" + filename)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official wheel for {filename}, got {len(matches)}")
    source = matches[0]
    expected = parse_qs(urlsplit(source).fragment)["sha256"][0]
    url = source.split("#")[0].replace("https://download-r2.pytorch.org/", "https://download.pytorch.org/")
    out = Path(__file__).resolve().parents[1] / ".downloads" / filename
    out.parent.mkdir(exist_ok=True)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if out.exists() and digest(out) == expected:
        print(f"Already verified: {out.name}", flush=True)
        return
    with requests.get(url + "?probe=1", headers={"Range": "bytes=0-0"}, timeout=(15, 30), stream=True) as probe:
        probe.raise_for_status()
        if probe.status_code != 206:
            raise RuntimeError("Server did not support partial downloads")
        size = int(probe.headers["Content-Range"].split("/")[-1])
    chunk_size = 1024 * 1024
    parts = out.parent / (filename + ".parts1m")
    parts.mkdir(exist_ok=True)
    previous = out.parent / (filename + ".parts")
    if previous.exists():
        for saved in previous.iterdir():
            if not saved.name.isdigit() or saved.stat().st_size != 16 * 1024 * 1024:
                continue
            offset = int(saved.name)
            with saved.open('rb') as stream:
                for pos in range(offset, offset + saved.stat().st_size, chunk_size):
                    piece = parts / f'{pos:012d}'
                    block = stream.read(chunk_size)
                    if not piece.exists():
                        piece.write_bytes(block)

    def get_part(start):
        end = min(start + chunk_size, size) - 1
        part = parts / f"{start:012d}"
        needed = end - start + 1
        if part.exists() and part.stat().st_size == needed:
            return needed
        for attempt in range(5):
            try:
                with requests.get(url + f"?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 45)) as r:
                    r.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{size}"
                    if r.status_code != 206 or r.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Incorrect HTTP range: {r.headers.get('Content-Range')}")
                    temporary = part.with_suffix(".part")
                    with temporary.open("wb") as f:
                        for block in r.iter_content(1024 * 1024):
                            f.write(block)
                    if temporary.stat().st_size != needed:
                        raise RuntimeError("Partial response has incorrect size")
                    temporary.replace(part)
                return needed
            except (requests.RequestException, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    starts = list(range(0, size, chunk_size))
    done = 0
    report = 0
    print(f"Downloading {filename}: {size / 2**20:.0f} MiB; 48 connections", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(get_part, start) for start in starts]
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            if done - report >= 128 * 1024 * 1024 or done == size:
                print(f"{filename}: {done / size:.1%} ({done / 2**20:.0f} MiB)", flush=True)
                report = done
    assembled = out.with_suffix(".assembling")
    with assembled.open("wb") as f:
        for start in starts:
            with (parts / f"{start:012d}").open("rb") as part:
                shutil.copyfileobj(part, f)
    actual = digest(assembled)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: expected {expected}, got {actual}")
    assembled.replace(out)
    print(f"Verified SHA256 {actual}: {out.name}", flush=True)


if __name__ == "__main__":
    download("torch", "2.8.0")
    download("torchvision", "0.23.0")
