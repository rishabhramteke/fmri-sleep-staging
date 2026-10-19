"""Fetch the ds005127 physiology derivative and keep only what the confound analysis needs.

The dataset ships `derivatives/physio_prepro/`: Biopac pulse-oximeter and respiratory-bellows traces
already peak-detected by the dataset owners (de Zwart, NIH), on a time axis whose zero is the first MR
volume trigger (verified: MR_TIME spans 0 to (n_vol - 1) * TR for the runs checked). Using their peak
detection rather than our own keeps the physiology independent of anything we tuned.

The .svd files are 32 GB, almost all of it 1 kHz waveforms and per-sample latency arrays we do not
need. Each file is downloaded, reduced to peak times, peak amplitudes, quality scores and 25 Hz
decimated waveforms (about 0.3 MB per run), and the original deleted. Resumable: runs whose compact
file exists are skipped.

    source <venv>/bin/activate
    python 11_fetch_physio.py [--workers 8]
"""

import argparse
import importlib.util
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
DS = BASE / "data" / "ds005127"
OUT = BASE / "derivatives" / "physio_ds005127"
DECIM_HZ = 25

sys.path.insert(0, str(DS / "code" / "python"))
from jadz_io.read_data import read_data  # noqa: E402  (the dataset's own .svd reader)
from jadz_lib.type_of import type_of  # noqa: E402
from jadz_lib.type_size import type_size  # noqa: E402

_spec = importlib.util.spec_from_file_location("fetch", BASE / "03_fetch_sleep_scoring.py")
fetch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_mod)


def decimate(t, x, hz=DECIM_HZ):
    """Block-average a 1 kHz trace down to `hz`. Block means are enough for envelopes and RV."""
    step = int(round(1.0 / (hz * (t[1] - t[0]))))
    n = (len(x) // step) * step
    return t[:n:step].astype(np.float64), x[:n].reshape(-1, step).mean(1).astype(np.float32)


def compact(svd_path):
    d = read_data(str(svd_path), silent=1)[0].flat[0]
    t = d["TIME"]
    tr_, resp = decimate(t, np.asarray(d["RESPIRATION"]).ravel())
    _, card = decimate(t, np.asarray(d["CARDIAC"]).ravel())
    return dict(
        card_t=d["CARD_TIME"], card_amp=d["CARD_AMP"], card_valley_t=d["CARD_TIME_VALLEY"],
        resp_t=d["RESP_TIME"], resp_amp=d["RESP_AMP"], resp_valley_t=d["RESP_TIME_VALLEY"],
        card_q=d["CARD_QUALITY_SCORE"], card_snr=d["CARD_SNR_ESTIMATE"],
        resp_q=d["RESP_QUALITY_SCORE"], resp_snr=d["RESP_SNR_ESTIMATE"],
        card_channel=d["CARD_CHANNEL_NAME"], resp_channel=d["RESP_CHANNEL_NAME"],
        mr_time=d["MR_TIME"], wave_t=tr_, resp_wave=resp, card_wave=card,
    )


class RangeReader:
    """Random access into an S3 object through HTTP Range requests, with a small block cache.

    The .svd encoding is sequential (per field: uint64 header [type, ndims, dims...], then data), so a
    field can be skipped from its header alone. Only ~8% of each file is fetched: the peak arrays and
    the respiratory waveform. Everything else, mostly per-sample latency arrays, is never downloaded.
    """
    BLOCK = 1 << 16

    def __init__(self, key):
        self.url = f"{fetch_mod.BUCKET}/{urllib.parse.quote(key)}"
        self.cache = {}

    def _get(self, a, b, attempts=6):
        for i in range(attempts):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return np.frombuffer(r.read(), np.uint8)
            except Exception:
                if i == attempts - 1:
                    raise
                time.sleep(2 ** i * 0.5)

    def read(self, off, n):
        if n > 4 * self.BLOCK:
            return self._get(off, off + n)
        out, pos = [], off
        while pos < off + n:
            blk = pos // self.BLOCK
            if blk not in self.cache:
                self.cache[blk] = self._get(blk * self.BLOCK, (blk + 1) * self.BLOCK)
            b = self.cache[blk]
            lo = pos - blk * self.BLOCK
            take = min(len(b) - lo, off + n - pos)
            if take <= 0:
                raise EOFError("read past end of object")
            out.append(b[lo:lo + take])
            pos += take
        return np.concatenate(out)


def _head(rr, off):
    h = rr.read(off, 24).view(np.uint64)
    nd = int(h[1])
    dims = [int(h[2])] + ([int(x) for x in rr.read(off + 24, 8 * (nd - 1)).view(np.uint64)] if nd > 1 else [])
    return int(h[0]), dims, off + 16 + 8 * nd


def _skip_or_read(rr, off, want):
    """Decode (or skip) one encoded element at `off`. Returns (value or None, next offset)."""
    typ, dims, p = _head(rr, off)
    nelm = int(np.prod(dims))
    if typ == 7:                                   # strings: uint32 lengths, then bytes
        lens = rr.read(p, 4 * nelm).view(np.uint32)
        q = p + 4 * nelm
        val = None
        if want:
            val = [rr.read(q + int(lens[:i].sum()), int(lens[i])).tobytes().decode() for i in range(nelm)]
        return val, q + int(lens.sum())
    if typ == 8:
        raise ValueError("nested structure inside a field is not expected here")
    np_t = type_of(typ, return_numpy_type=1)
    nbyt = nelm * type_size(typenr=typ)
    val = rr.read(p, nbyt).view(np_t).reshape(list(reversed(dims))) if want else None
    return val, p + nbyt


WANT = {"CARD_TIME", "CARD_AMP", "CARD_TIME_VALLEY", "RESP_TIME", "RESP_AMP", "RESP_TIME_VALLEY",
        "CARD_QUALITY_SCORE", "CARD_SNR_ESTIMATE", "RESP_QUALITY_SCORE", "RESP_SNR_ESTIMATE",
        "CARD_CHANNEL_NAME", "RESP_CHANNEL_NAME", "MR_TIME", "RESPIRATION"}


def compact_remote(key):
    rr = RangeReader(key)
    h1 = rr.read(0, 32).view(np.float32)
    assert h1[0] == 0.0 and h1[1] == 1.0, "not a little-endian .svd"
    L = int(rr.read(int(4 * h1[2]), 8).view(np.uint64)[0])
    obj = int(4 * h1[2]) + 8 * L                   # first object = the physio structure
    typ, dims, p = _head(rr, obj)
    assert typ == 8, "first object is not a structure"
    ntag = int(rr.read(p, 4).view(np.uint32)[0])
    tags, p = _skip_or_read(rr, p + 4, True)
    assert len(tags) == ntag
    d = {}
    for t in tags:
        if t == "TIME":                            # need only the start and the step
            typ_, dims_, q = _head(rr, p)
            first = rr.read(q, 16).view(np.float64)
            d["TIME0"], d["DT"], d["N"] = float(first[0]), float(first[1] - first[0]), int(np.prod(dims_))
            p = q + int(np.prod(dims_)) * type_size(typenr=typ_)
            continue
        val, p = _skip_or_read(rr, p, t in WANT)
        if val is not None:
            d[t] = val
    t_full = d["TIME0"] + d["DT"] * np.arange(d["N"])
    wt, resp = decimate(t_full, np.asarray(d["RESPIRATION"]).ravel())
    return dict(
        card_t=d["CARD_TIME"], card_amp=d["CARD_AMP"], card_valley_t=d["CARD_TIME_VALLEY"],
        resp_t=d["RESP_TIME"], resp_amp=d["RESP_AMP"], resp_valley_t=d["RESP_TIME_VALLEY"],
        card_q=d["CARD_QUALITY_SCORE"], card_snr=d["CARD_SNR_ESTIMATE"],
        resp_q=d["RESP_QUALITY_SCORE"], resp_snr=d["RESP_SNR_ESTIMATE"],
        card_channel=np.asarray(d["CARD_CHANNEL_NAME"]), resp_channel=np.asarray(d["RESP_CHANNEL_NAME"]),
        mr_time=d["MR_TIME"], wave_t=wt, resp_wave=resp,
    )


def one(key, remote=True):
    stem = key.rsplit("/", 1)[-1].replace(".svd", "")
    sub = stem.split("_")[0]
    dest = OUT / sub / f"{stem}_physio.npz"
    if dest.exists():
        return stem, "skip", 0.0
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if remote:
        np.savez_compressed(dest, **compact_remote(key))
    else:
        svd, _ = fetch_mod.fetch(key, DS, max_bytes=None)
        np.savez_compressed(dest, **compact(svd))
        svd.unlink()
    return stem, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    keys, _ = fetch_mod.s3_list("ds005127/derivatives/physio_prepro/")
    keys = [k for k, _ in keys if k.endswith(".svd") and "task-sleep" in k]
    print(f"{len(keys)} physio runs; free {fetch_mod.free_gb():.0f} GB", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, k): k for k in keys}
        for f in as_completed(futs):
            done += 1
            try:
                stem, status, dt = f.result()
                print(f"[{done}/{len(keys)}] {status} {stem} {dt:.0f}s", flush=True)
            except Exception as e:  # keep going; a rerun retries failures
                print(f"[{done}/{len(keys)}] FAIL {futs[f]}: {e}", flush=True)
    n = len(list(OUT.rglob("*_physio.npz")))
    print(f"compact physio files: {n}/{len(keys)}")


if __name__ == "__main__":
    main()
