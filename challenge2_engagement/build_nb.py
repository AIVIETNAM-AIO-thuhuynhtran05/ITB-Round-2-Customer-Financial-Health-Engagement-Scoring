"""Build (and optionally execute) Challenge2_Customer_Engagement.ipynb from nb_source.txt.

nb_source.txt is split on lines starting with '#%% md' / '#%% code' / '#%% include <file>'.
An include pulls the code cells of a .py file (split on '# %% [code]').

    python build_nb.py            # build only
    python build_nb.py --run      # build + execute (set BI10_ROOT to the folder holding Clean_data/)
"""
import pathlib
import sys

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "nb_source.txt"
OUT = HERE / "Challenge2_Customer_Engagement.ipynb"


def cells():
    kind, buf = None, []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if line.startswith("#%%"):
            if kind:
                yield kind, "\n".join(buf).strip("\n")
            parts = line.split(maxsplit=2)
            kind, buf = parts[1], []
            if kind == "include":
                code = (HERE / parts[2]).read_text(encoding="utf-8")
                for chunk in code.split("# %% [code]")[1:]:
                    yield "code", chunk.split("\n", 1)[1].strip("\n")
                kind = None
        else:
            buf.append(line)
    if kind:
        yield kind, "\n".join(buf).strip("\n")


nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
for kind, text in cells():
    nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text))
nbformat.write(nb, OUT)
print(f"{OUT.name}: {len(nb.cells)} cells")

if "--run" in sys.argv:
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, OUT)
    print("executed OK")
