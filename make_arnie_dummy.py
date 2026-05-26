"""Write a placeholder arnie config (arnie_file.txt) next to this script.

arnie needs a config telling it where to find LinearPartition / TMP. For
our usage (RibonanzaNet oracle reward), the placeholder below is enough.

After running:

    python make_arnie_dummy.py
    export ARNIEFILE="$(pwd)/arnie_file.txt"
"""
from pathlib import Path

PATH = Path(__file__).resolve().parent / "arnie_file.txt"
PATH.write_text("linearpartition: .\nTMP: /tmp\n")
print(f"wrote {PATH}")
