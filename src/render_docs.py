"""Fill the <!-- TABLE:name --> ... <!-- /TABLE --> blocks in the markdown deliverables with the
tables produced by the pipeline (results/tables/<name>.md). Safe to run repeatedly.
Also warns about every unfinished  ✏️ TODO  left in the docs."""
import re

from . import config as C
from .utils import log

DOCS = [
    "README.md", "DATA.md", "RESULTS.md", "Recommendation.md",
    "Part1_Raw_Data_Inspection_and_Cleaning.md",
    "Part2_Washing_Machine_EDA.md",
    "Part3_Disaggregation_Model.md",
    "Part4_Practical_Implications.md",
]
PAT = re.compile(r"(<!-- TABLE:(\w+) -->)(.*?)(<!-- /TABLE -->)", re.S)


def run():
    for name in DOCS:
        p = C.ROOT / name
        if not p.exists():
            continue
        text = p.read_text()

        def sub(m):
            t = C.TABLES / f"{m.group(2)}.md"
            body = t.read_text() if t.exists() else f"_(table `{m.group(2)}` not generated yet - run the pipeline)_\n"
            return f"{m.group(1)}\n{body}{m.group(4)}"

        text = PAT.sub(sub, text)
        p.write_text(text)
        todo = text.count("✏️ TODO")
        if todo:
            log.warning("%s still has %d '✏️ TODO' item(s) to write yourself", name, todo)
    log.info("Docs rendered.")


if __name__ == "__main__":
    run()
