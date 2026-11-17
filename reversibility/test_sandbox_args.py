"""A list-valued path (seen from Llama under the careful prompt) must be a failed call, not a crash: python test_sandbox_args.py"""
from sandbox import Sandbox

sb = Sandbox([{"path": "a.txt", "size": 1}, {"path": "b.txt", "size": 1}], recycle_bin=False)
for tool in ("delete_file", "edit_file", "trash_file", "move_file", "read_file", "make_backup"):
    obs = sb.execute(tool, {"path": ["a.txt", "b.txt"], "content": "x", "dest": "c.txt"})
    assert obs.startswith("ERROR"), (tool, obs)
assert sb.execute("move_file", {"path": "a.txt", "dest": ["c.txt"]}).startswith("ERROR")
assert set(sb.files) == {"a.txt", "b.txt"} and all(e["ok"] is False and e["label"] is None for e in sb.log)
assert sb.execute("delete_file", {"path": "a.txt"}) == "Deleted a.txt" and sb.log[-1]["label"] == "irreversible"
print("sandbox argument checks passed")
