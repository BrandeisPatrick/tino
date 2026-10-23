"""kev-suites loading, pinned to one Hub revision and checked against each suite's manifest."""
import hashlib
import json
import random

from huggingface_hub import hf_hub_download

SUITES_REPO = "jaredpalmer/kev-suites"
SUITES_REV = "cc4bac803e73112689ec327ffa481c519cbc7a05"
SUITES = {
    "decision-v7": "v7/decision-v7",   # held-out examples from the ten trained sources ("trained sources")
    "transfer-v4": "v4/transfer-v4",   # sources never trained on ("new sources"), Kev's headline transfer suite
    "transfer-v9": "v9/transfer-v9",   # transfer-v4 + MMLU-Pro, buried evidence, unknowable records
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(suite, split):
    """-> (local path, provenance dict). Fails if the manifest lists a checksum and the file disagrees."""
    base = SUITES[suite]
    path = hf_hub_download(SUITES_REPO, f"{base}/{split}.jsonl", repo_type="dataset", revision=SUITES_REV)
    man = json.load(open(hf_hub_download(SUITES_REPO, f"{base}/manifest.json", repo_type="dataset", revision=SUITES_REV)))
    digest = sha256(path)
    expected = (man.get("files") or {}).get(f"{split}.jsonl", {}).get("sha256")
    if expected and expected != digest:
        raise ValueError(f"{suite}/{split}: sha256 {digest} != manifest {expected}")
    return path, {"repo": SUITES_REPO, "revision": SUITES_REV, "file": f"{base}/{split}.jsonl", "sha256": digest,
                  "manifest_checked": bool(expected)}


def load_records(suite, split):
    path, prov = fetch(suite, split)
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    prov["records"] = len(recs)
    return recs, prov


def build_rows(encoder, records, suite=""):
    rows, skipped = [], {"no_label": 0, "too_long": 0}
    for i, rec in enumerate(records):
        rid = (rec.get("_meta") or {}).get("id") or f"{suite}/{i}"
        for r in encoder.encode_record(rec, rec_id=rid):
            r["rec_idx"] = i
            if r["label"] is None:
                skipped["no_label"] += 1
            elif len(r["ids"]) > encoder.max_row:
                skipped["too_long"] += 1
            else:
                rows.append(r)
    return rows, skipped


def batches(rows, token_budget, shuffle=False, seed=0, chunk=512):
    """Length-bucketed batches whose padded size (rows x longest row) stays within token_budget."""
    order = list(range(len(rows)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)
    out = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: len(rows[i]["ids"]))
        cur, longest = [], 0
        for i in part:
            n = len(rows[i]["ids"])
            if cur and max(longest, n) * (len(cur) + 1) > token_budget:
                out.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, n)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out
