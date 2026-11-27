"""Pre-zip a staged dataset so the upload itself does no heavy work.

Explorer login nodes kill CPU-heavy processes, and `kaggle datasets create
--dir-mode zip` compresses every sub-folder on the spot. Run this on a compute
node instead: each sub-folder becomes an uncompressed <name>.zip (Kaggle unpacks
it back into a folder) and top-level files are hard-linked. Then upload from the
login node with --dir-mode skip.

    python kaggle/pack_dataset.py SRC_DIR DST_DIR
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

for entry in sorted(src.iterdir()):
    if entry.is_file():
        os.link(entry, dst / entry.name)
        continue
    target = dst / f"{entry.name}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in sorted(entry.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(entry))
    print(f"{target.name}: {target.stat().st_size / 1e9:.2f} GB", flush=True)

print("packed:", sorted(p.name for p in dst.iterdir()))
