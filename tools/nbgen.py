"""percent 포맷 .py  →  .ipynb 변환기.

노트북 JSON을 손으로 만지면 diff도 못 읽고 충돌도 못 푼다.
그래서 각 주차 노트북의 원본은 `tools/src/<주차폴더>.py` 에 percent 포맷으로 두고,
여기서 `<주차폴더>/demo.ipynb` 를 생성한다.

percent 포맷:

    # %% [markdown]
    # ## 제목
    # 본문

    # %%
    print("코드 셀")

주의: Colab에서 직접 고친 내용은 tools/src 로 돌아오지 않는다.
구조를 바꿀 거면 tools/src 를 고치고 `python tools/build.py` 를 다시 돌릴 것.
(Colab에서의 즉흥 수정은 세션 당일에만 쓰고 버리는 용도로 생각하면 된다.)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent / "src"

_KERNELSPEC = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}


def _strip_md(lines: list[str]) -> list[str]:
    """markdown 셀 본문에서 leading '# ' 를 벗겨낸다."""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def parse(text: str) -> list[dict]:
    cells: list[dict] = []
    kind: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        body = buf[:]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            return
        if kind == "markdown":
            body = _strip_md(body)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(body)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": _src(body),
                }
            )

    for line in text.splitlines():
        stripped = line.rstrip("\n")
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        if kind is not None:
            buf.append(stripped)
    flush()
    return cells


def _src(lines: list[str]) -> list[str]:
    """nbformat 규약: 마지막 줄을 뺀 나머지에 개행을 붙인다."""
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build(src: Path, dest: Path) -> int:
    cells = parse(src.read_text(encoding="utf-8"))
    nb = {"cells": cells, "metadata": _KERNELSPEC, "nbformat": 4, "nbformat_minor": 0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(cells)


def build_all() -> None:
    srcs = sorted(SRC_DIR.glob("week*.py"))
    if not srcs:
        print(f"원본이 없다: {SRC_DIR}")
        return
    for src in srcs:
        dest = REPO_ROOT / src.stem / "demo.ipynb"
        n = build(src, dest)
        print(f"  {src.name:28s} → {dest.relative_to(REPO_ROOT).as_posix():34s} ({n} cells)")


if __name__ == "__main__":
    build_all()
