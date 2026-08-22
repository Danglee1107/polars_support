from pathlib import Path
from os.path import exists


BASE= Path(__file__).resolve().parent.parent

IMG= BASE / "img"

# -- check folder img have exist --
IMG.mkdir(parents=True, exist_ok=True)

GRAPH= IMG / "graphs"

# -- check folder graphs have exist --
GRAPH.mkdir(parents=True, exist_ok=True)
