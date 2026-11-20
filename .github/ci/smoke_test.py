"""Execute the tool's notebook(s) inside the built image and report errors.

Config (smoke_test.json):
  {"notebooks": [{"path": "x.ipynb", "must_pass_cells": null, "timeout": 1800}]}

must_pass_cells: null -> every code cell must run without error; an integer N ->
only the first N code cells must pass (later cells need interactive input such
as an uploaded corpus and are checked on BinderHub instead).
"""
import json, os, sys, time
import nbformat
from nbclient import NotebookClient

cfg = json.load(open(sys.argv[1]))
home = os.environ.get("REPO_DIR") or os.path.expanduser("~")
failed = False
for nbcfg in cfg["notebooks"]:
    path = os.path.join(home, nbcfg["path"])
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=nbcfg.get("timeout", 1800), allow_errors=True,
        resources={"metadata": {"path": os.path.dirname(path)}},
        kernel_name=nbcfg.get("kernel", "python3"),
    )
    t0 = time.time()
    client.execute()
    code = [c for c in nb.cells if c.cell_type == "code"]
    limit = nbcfg.get("must_pass_cells")
    limit = len(code) if limit is None else limit
    print(f"== {nbcfg['path']}: {len(code)} code cells in {time.time()-t0:.0f}s, first {limit} must pass")
    for i, c in enumerate(code):
        errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
        status = "ERROR" if errs else "ok"
        print(f"  [{i:2d}] {status:5s} {c.source.strip().splitlines()[0][:90] if c.source.strip() else ''}")
        for e in errs:
            print(f"        {e['ename']}: {e['evalue'][:300]}")
            if i < limit:
                failed = True
sys.exit(1 if failed else 0)
