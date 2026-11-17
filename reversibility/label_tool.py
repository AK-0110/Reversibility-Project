"""Hand-labelling helper: shows one row at a time and saves your answer straight into the CSV (resumes where you stopped).
python label_tool.py data/manual_annotator1.csv        (person 2: data/manual_annotator2.csv)
Keys: 1 = reversible, 2 = partially_reversible, 3 = irreversible, b = back one row, q = quit (progress is saved)."""
import csv, sys

LABELS = {"1": "reversible", "2": "partially_reversible", "3": "irreversible"}
path = sys.argv[1]
with open(path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f)); fields = list(rows[0].keys())


def save():
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


i = next((k for k, r in enumerate(rows) if not r["your_label"].strip()), len(rows))
while 0 <= i < len(rows):
    r = rows[i]
    print("\n" + "=" * 70 + f"\nRow {i + 1} of {len(rows)}   (done: {sum(bool(x['your_label'].strip()) for x in rows)})\n" + "=" * 70)
    print("STATE:\n" + r["state_text"] + "\n\nACTION: " + r["action_text"])
    print("\nAfter this action, can the exact previous state be restored in <= 3 steps?")
    print("  1 = reversible   2 = partially_reversible   3 = irreversible   b = back   q = quit")
    a = input("> ").strip().lower()
    if a in LABELS: r["your_label"] = LABELS[a]; save(); i += 1
    elif a == "b": i = max(0, i - 1)
    elif a == "q": break
    else: print("Press 1, 2, 3, b or q.")
if i >= len(rows):
    print(f"\nAll {len(rows)} rows are labelled. Saved to {path}.")
