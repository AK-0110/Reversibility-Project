"""Reworded copies of the sandbox TEST rows (same ids, same labels) for frozen-probe checks (probes.py --eval-data):
  data/sandbox_terse.jsonl  every setting line says only on/off: no "permanently", "cannot be recalled", "final"
  data/sandbox_flip.jsonl   only the setting that is IRRELEVANT to the action (probes.DERANGE) is flipped
A probe that reads reversibility should keep its accuracy on both; one that reads danger phrases should not.
python make_variants.py data/sandbox.jsonl data/sandbox_extra.jsonl     (writes next to the first input)"""
import os, sys
from probes import DERANGE
from utils import read_jsonl, write_jsonl

TERSE = {"Recycle bin: enabled (deleted files can be restored)": "Recycle bin: on",
         "Recycle bin: disabled (deleted files are removed permanently)": "Recycle bin: off",
         "Edit history: enabled (earlier versions of edited files are kept)": "Edit history: on",
         "Edit history: disabled (edits overwrite the old text)": "Edit history: off",
         "Email: sent mail can be recalled within 60 seconds": "Email recall: on",
         "Email: mail is delivered immediately and cannot be recalled": "Email recall: off",
         "Transfers: can be reversed within 24 hours": "Transfer reversal: on",
         "Transfers: are final once submitted": "Transfer reversal: off"}
LINES = list(TERSE)
FLIP = {"recycle_bin": LINES[0:2], "versioning": LINES[2:4], "email_undo": LINES[4:6], "transfer_reversible": LINES[6:8]}


def terse(text):
    out = text
    for k, v in TERSE.items(): out = out.replace(k, v)
    assert sum(l in TERSE for l in text.splitlines()) == 4 and not any(k in out for k in TERSE), text
    return out


def flip(text, verb):
    a, b = FLIP[DERANGE[verb]]
    out = text.replace(a, "\0").replace(b, a).replace("\0", b)
    assert sum(x != y for x, y in zip(text.splitlines(), out.splitlines())) == 1, text
    return out


if __name__ == "__main__":
    test = [r for p in sys.argv[1:] for r in read_jsonl(p) if r["split"] == "test"]
    d = os.path.dirname(sys.argv[1])
    write_jsonl(f"{d}/sandbox_terse.jsonl", [{**r, "state_text": terse(r["state_text"])} for r in test])
    fl = [{**r, "state_text": flip(r["state_text"], r["verb"])} for r in test if r["verb"] in DERANGE]
    write_jsonl(f"{d}/sandbox_flip.jsonl", fl)
    print(f"terse: {len(test)} rows, flip: {len(fl)} rows")
