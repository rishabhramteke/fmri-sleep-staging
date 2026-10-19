"""Fetch the sleep-scoring / hypnogram files for ds005127 and ds003768 from OpenNeuro S3.

Downloads only small text files (scoring, README, participants). No imaging. Total is a few MB.
Public bucket, no credentials needed.

    source <venv>/bin/activate
    python 03_fetch_sleep_scoring.py
"""

import shutil
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
OUT = Path(__file__).parent / "data"

# Hard ceiling per file. This script is for text only. Without it, a name filter like "sleep" also
# matches ds003768/sourcedata/fMRI_processed_file_1/*task-sleep*.nii.gz, which is 174 GB.
MAX_BYTES = 5_000_000


def s3_list(prefix, delimiter=None):
    """List keys under a prefix, following continuation tokens."""
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60).read())
        keys += [(c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text))
                 for c in root.findall("s3:Contents", NS)]
        prefixes += [p.find("s3:Prefix", NS).text for p in root.findall("s3:CommonPrefixes", NS)]
        if root.find("s3:IsTruncated", NS).text != "true":
            return keys, prefixes
        token = root.find("s3:NextContinuationToken", NS).text


def fetch(key, dest_root, attempts=5, max_bytes=MAX_BYTES):
    """Download one key. S3 resets connections under rapid sequential requests, so back off."""
    dest = dest_root / key.split("/", 1)[1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return dest, True
        except Exception as e:
            if i == attempts - 1:
                raise RuntimeError(f"failed after {attempts} attempts: {key}") from e
            time.sleep(2 ** i * 0.5)
    return dest, True


def grab(keys, dest_root, label):
    too_big = [(k, s) for k, s in keys if s > MAX_BYTES]
    keys = [(k, s) for k, s in keys if s <= MAX_BYTES]
    if too_big:
        print(f"  {label}: SKIPPING {len(too_big)} files over {MAX_BYTES / 1e6:.0f} MB "
              f"({sum(s for _, s in too_big) / 1e9:.1f} GB), e.g. {too_big[0][0].split('/')[-1]}")
    got = new = 0
    for key, _ in keys:
        _, was_new = fetch(key, dest_root)
        got += 1
        new += was_new
        if was_new:
            time.sleep(0.05)  # be gentle with the bucket
    print(f"  {label}: {got} files ({new} newly downloaded) -> {dest_root}")


def main():
    # ---- ds005127: 30-s scoring, one file per run, plus documentation
    print("### ds005127 (NIH overnight EEG-fMRI)")
    root = OUT / "ds005127"
    scoring, _ = s3_list("ds005127/derivatives/sleep_scoring_mr/")
    grab(scoring, root, "scoring")
    meta, _ = s3_list("ds005127/", "/")
    grab([(k, s) for k, s in meta if not k.endswith("/")], root, "metadata")

    # ---- ds003768: hypnograms live at sourcedata/sub-XX-sleep-stage.tsv (verified by listing).
    # Select by size and extension, never by name alone: the imaging is also named "task-sleep".
    print("\n### ds003768 (Penn State in-scanner sleep)")
    root = OUT / "ds003768"
    keys, _ = s3_list("ds003768/sourcedata/")
    staging = [(k, s) for k, s in keys if k.endswith("-sleep-stage.tsv")]
    print(f"  hypnograms found: {len(staging)} (expect 33)")
    grab(staging, root, "hypnograms")

    root_keys, _ = s3_list("ds003768/", "/")
    docs = [(k, s) for k, s in root_keys
            if k.endswith((".json", "README", "CHANGES")) and s <= MAX_BYTES]
    grab(docs, root, "docs")


# ---------------------------------------------------------------- imaging ----
# Named imaging targets, with verified sizes. Everything not listed here is deliberately excluded:
# ds005127 ships 1,040 GB of in-scanner video, 525 GB of phase-preserving func-complex (the FAQ says
# its magnitude is identical to func), 458 GB of raw EEG (we already have the scoring), and 277 GB of
# registered derivatives. ds003768 ships 217 GB of "processed" volumes on top of a 22.5 GB raw BIDS
# func tree. Pulling whole datasets here would be 2.7 TB for no benefit.
IMAGING = {
    "ds003768-rest": ("ds003768/", lambda k: "/func/" in k and "task-rest" in k and k.endswith(".nii.gz"), 4.2),
    "ds003768-sleep": ("ds003768/", lambda k: "/func/" in k and "task-sleep" in k and k.endswith(".nii.gz"), 18.2),
    "ds003768-anat": ("ds003768/", lambda k: "/anat/" in k and k.endswith(".nii.gz"), 0.4),
    "ds005127-sleep": ("ds005127/", lambda k: "/func/" in k and "task-sleep" in k and "func-complex" not in k, 67.8),
    "ds005127-anat": ("ds005127/", lambda k: "/anat/" in k and "func-complex" not in k, 1.8),
}
MARGIN_GB = 20  # never fill the disk to the brim; processing needs scratch


def free_gb(path=OUT):
    st = shutil.disk_usage(path)
    return st.free / 1e9


def fetch_imaging(target):
    if target not in IMAGING:
        raise SystemExit(f"unknown target {target!r}. options: {', '.join(IMAGING)}")
    prefix, pred, est_gb = IMAGING[target]
    ds = prefix.rstrip("/")
    print(f"### imaging target: {target} (~{est_gb} GB estimated)")
    print(f"  free space: {free_gb():.1f} GB")
    if free_gb() < est_gb + MARGIN_GB:
        raise SystemExit(f"REFUSING: need {est_gb + MARGIN_GB:.0f} GB free "
                         f"({est_gb} GB of data plus a {MARGIN_GB} GB working margin), "
                         f"have {free_gb():.1f} GB. Free space or use an external volume.")
    keys, _ = s3_list(prefix)
    sel = [(k, s) for k, s in keys if pred(k)]
    total = sum(s for _, s in sel) / 1e9
    print(f"  selected {len(sel)} files, {total:.1f} GB actual")
    if total > free_gb() - MARGIN_GB:
        raise SystemExit(f"REFUSING: actual size {total:.1f} GB exceeds the safe budget.")
    root = OUT / ds
    done = 0
    for i, (k, s) in enumerate(sel, 1):
        _, was_new = fetch(k, root, max_bytes=None)
        done += s
        if i % 10 == 0 or i == len(sel):
            print(f"  [{i}/{len(sel)}] {done / 1e9:.1f}/{total:.1f} GB, free {free_gb():.0f} GB", flush=True)
    print(f"  done: {target} -> {root}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--imaging":
        if len(sys.argv) < 3:
            raise SystemExit(f"usage: python {Path(__file__).name} --imaging <target>\n"
                             f"targets: " + ", ".join(f"{k} (~{v[2]} GB)" for k, v in IMAGING.items()))
        fetch_imaging(sys.argv[2])
    else:
        main()
