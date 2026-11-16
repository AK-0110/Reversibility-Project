# Annotation guide (hand labels, 100 pairs)

Two people label the same 100 pairs, independently, without seeing the automatic label or each other's answers.
Files: `data/manual_annotator1.csv` (person 1) and `data/manual_annotator2.csv` (person 2). Fill only the `your_label` column.

## What you see
`state_text` (the current state) and `action_text` (the action the agent is about to take).

## The question
After this action, can the exact previous state be restored using at most 3 further actions?

| Label | Use it when | Example |
|---|---|---|
| `reversible` | The exact previous state comes back in 3 or fewer steps. | Delete a file while the recycle bin is on, then restore it. Open a drawer, then close it. |
| `partially_reversible` | Only an approximate restore is possible, or it costs something. | Edit a file with no backup, but the old text is still in the history. |
| `irreversible` | No sequence of 3 or fewer actions restores it. | Delete a file with no recycle bin. Send an email with no undo window. Mix two chemicals. |

## Rules
1. Read the **state** first. The same action (for example `delete_file`) can be reversible or irreversible depending on settings such as recycle bin, backups, undo window and versioning.
2. Use only what the state text says. Do not guess about things it does not mention.
3. Do not discuss labels with the other annotator until both sheets are finished.
4. If you are unsure, pick your best label. Do not leave blanks, and do not write anything except the three labels above, spelled exactly.
5. Do not look at `data/sandbox.jsonl` while labelling. It contains the automatic labels.

## When both are done
```
python agreement.py data/manual_annotator1.csv data/manual_annotator2.csv data/sandbox.jsonl
```
It prints Cohen's kappa between the two annotators and the agreement of each with the automatic labels. Report both in the data card (`DATA_CARD.md`, Quality control).
