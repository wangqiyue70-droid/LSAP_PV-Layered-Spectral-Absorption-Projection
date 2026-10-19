"""Fetch the public author checkpoint and verify the saved source archive."""
import hashlib
from pathlib import Path
import zipfile

import requests

EXPECTED = '0c29a88f1a4651027db02f4437ec4df33d512a83b242916b1e3e700904f58a87'
ROOT = Path(__file__).resolve().parents[1] / 'LSAP-PV' / 'checkpoint'
ARCHIVE = ROOT / 'vein_sty_concat_sigma_cosine_vpem.zip'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    if not ARCHIVE.exists() or digest(ARCHIVE) != EXPECTED:
        url = 'https://drive.usercontent.google.com/download'
        params = {'id': '1K85P8HTwGZ99v5jTL9dfOtZzlwZ6kBE8', 'export': 'download', 'confirm': 't'}
        with requests.get(url, params=params, stream=True, timeout=(20, 60)) as r:
            r.raise_for_status()
            if 'text/html' in r.headers.get('Content-Type', ''):
                raise RuntimeError('Google Drive returned a web page instead of the checkpoint archive')
            partial = ARCHIVE.with_suffix('.zip.part')
            with partial.open('wb') as f:
                for block in r.iter_content(1024 * 1024):
                    f.write(block)
        if digest(partial) != EXPECTED:
            raise RuntimeError('Archive changed or download was incomplete; SHA256 did not match the recorded source')
        partial.replace(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            destination = (ROOT / info.filename).resolve()
            if not destination.is_relative_to(ROOT.resolve()):
                raise RuntimeError('Archive path escapes checkpoint directory')
        archive.extractall(ROOT)
    print('Verified and extracted:', ARCHIVE)


if __name__ == '__main__':
    main()
