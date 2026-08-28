#!/usr/bin/env python3
# Register dclm1b_train as a DataMix. Three names must agree and only one of
# them is the one --dataset.mix takes:
#   enum VALUE  "dclm-1b-train"   <- what --dataset.mix takes, and the .txt basename
#   member name  dclm_1b_train    <- fails at dataset.build() if passed to the CLI
#   directory    dclm1b_train     <- underscore after "dclm" is dropped on disk
# A wrong relpath is a silent glob miss, not an error.

import glob
import pathlib
import re

REPO = pathlib.Path("/accounts/projects/berkeleynlp/samir.kassam/EMO")
MIXES = REPO / "src/olmo_core/data/mixes"
DEST = "/scratch/users/samir.kassam/dclm/tokenized/dclm1b_train"

parts = sorted(glob.glob(f"{DEST}/*.npy"))
assert parts, f"no .npy under {DEST}; run the tokenizer first"

txt = MIXES / "dclm-1b-train.txt"
lines = "\n".join(f"train,tokenized/dclm1b_train/{pathlib.Path(p).name}" for p in parts)
txt.write_text(lines + "\n")
print(f"wrote {txt}:")
print(lines)

init = MIXES / "__init__.py"
src = init.read_text()
if 'dclm_1b_train = "dclm-1b-train"' in src:
    print("enum member already present")
else:
    anchor = '    dclm_100m_train = "dclm-100m-train"\n'
    assert src.count(anchor) == 1, "anchor line not found exactly once"
    init.write_text(src.replace(anchor, anchor + '    dclm_1b_train = "dclm-1b-train"\n'))
    print("added enum member dclm_1b_train")

print("\n--- verify")
print(re.search(r".*dclm_1b_train.*", init.read_text()).group(0))
