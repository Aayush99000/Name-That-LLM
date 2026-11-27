"""Turn kaggle/submission.py (cells split on '# %%') into kaggle/name-that-llm-submission.ipynb.

    python kaggle/build_notebook.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "submission.py"
DST = HERE / "name-that-llm-submission.ipynb"


def cells(source):
    for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(re.sub(r"^# ?", "", line) for line in body.strip().splitlines())
            yield {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
        else:
            yield {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": body.strip().splitlines(keepends=True)}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "cells": [{"id": f"cell-{i}", **c} for i, c in enumerate(cells(SRC.read_text()))],
}
DST.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {DST} ({len(nb['cells'])} cells)")
