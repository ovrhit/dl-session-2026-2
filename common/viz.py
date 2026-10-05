"""발표 화면용 시각화 헬퍼.

전제: 노트북을 빔/대형 모니터에 띄워놓고 뒷자리 부원이 본다.
→ 기본 폰트를 크게, 선을 굵게, 대비를 높게 잡는다. 논문 그림 스타일 흉내내지 말 것.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

# 색맹 안전 + 빔프로젝터에서 구분되는 팔레트 (Okabe-Ito 변형)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333"]

# "옛날 것" vs "요즘 것"을 대비시킬 때 고정으로 쓰는 두 색.
OLD = "#D55E00"   # 주황 — AlexNet, plain net, RNN, attention 없는 seq2seq
NEW = "#0072B2"   # 파랑 — ConvNeXt, ResNet, LSTM/Transformer, attention 있는 쪽


def use_presentation_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "lines.linewidth": 2.6,
            "lines.markersize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def show(path: str | Path, width: int = 900):
    """prep.py가 미리 뽑아둔 PNG/GIF를 노트북에 띄운다 (FAST_MODE 경로)."""
    from IPython.display import Image, display

    display(Image(filename=str(path), width=width))


# ---------------------------------------------------------------- 곡선

def curves(
    series: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "epoch",
    ylabel: str = "",
    title: str = "",
    x: Sequence[float] | None = None,
    logy: bool = False,
    colors: Mapping[str, str] | None = None,
    annotate_last: bool = True,
    ax=None,
):
    """여러 학습곡선을 한 축에 겹쳐 그린다.

    annotate_last=True 면 각 선의 끝에 이름을 직접 붙인다. 범례를 눈으로 쫓는 것보다
    발표 화면에서 훨씬 빨리 읽힌다.
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, (name, ys) in enumerate(series.items()):
        ys = np.asarray(ys, dtype=float)
        xs = np.arange(len(ys)) if x is None else np.asarray(x, dtype=float)
        c = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        ax.plot(xs, ys, label=name, color=c)
        if annotate_last and len(ys):
            ax.annotate(
                f" {name}",
                xy=(xs[-1], ys[-1]),
                color=c,
                fontweight="bold",
                va="center",
                fontsize=14,
            )
    if x is None:
        # x가 epoch/step 인덱스일 때 눈금이 2.5, 5.0 처럼 나오면 읽는 사람이 잠깐 멈칫한다.
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if not annotate_last:
        ax.legend()
    return ax


def bars(labels: Sequence[str], values: Sequence[float], *, title: str = "",
         ylabel: str = "", colors: Sequence[str] | None = None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    cols = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    ax.bar(labels, values, color=cols)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return ax


# ---------------------------------------------------------------- 이미지

def image_grid(
    images: Sequence,
    *,
    titles: Sequence[str] | None = None,
    ncols: int = 8,
    title: str = "",
    cmap: str | None = None,
    cell: float = 1.6,
):
    """이미지(또는 (H,W) 배열) 목록을 격자로. conv 필터 / 생성 샘플 / 노이즈 단계 전부 여기로."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell * 1.12))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, img in enumerate(images):
        axes[i].imshow(np.asarray(img), cmap=cmap)
        if titles is not None and i < len(titles):
            axes[i].set_title(str(titles[i]), fontsize=11, pad=3)
    if title:
        fig.suptitle(title, fontsize=19, fontweight="bold")
    fig.tight_layout()
    return fig


def normalize01(x) -> np.ndarray:
    """conv 필터처럼 음수가 섞인 텐서를 imshow 가능한 0~1로."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ---------------------------------------------------------------- 히트맵 (5·6주차 attention)

def heatmap(
    matrix,
    *,
    xlabels: Sequence[str] | None = None,
    ylabels: Sequence[str] | None = None,
    title: str = "",
    xaxis: str = "",
    yaxis: str = "",
    cmap: str = "Blues",
    ax=None,
    annotate: bool = False,
):
    matrix = np.asarray(matrix, dtype=float)
    if ax is None:
        h = max(3.5, 0.42 * matrix.shape[0] + 1.5)
        w = max(5.0, 0.52 * matrix.shape[1] + 2.0)
        _, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=12)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=12)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.15:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if matrix[i, j] > 0.55 else "black")
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return ax


# ---------------------------------------------------------------- 표

def table(rows: Iterable[Mapping], *, highlight: str | None = None):
    """비교표를 노트북에 예쁘게. highlight에 준 컬럼 기준으로 최댓값 강조."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    styler = df.style.hide(axis="index").set_properties(**{"font-size": "15px"})
    if highlight and highlight in df.columns:
        styler = styler.highlight_max(subset=[highlight], color="#cfe8ff")
    return styler


# ---------------------------------------------------------------- GIF (8주차)

def gif(frames: Sequence, path: str | Path, *, fps: int = 20, loop: int = 0) -> Path:
    """(H,W) 또는 (H,W,3) uint8 프레임 목록을 GIF로. 8주차 forward/reverse 애니메이션용."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=fps, loop=loop)
    return path


def to_uint8(x) -> np.ndarray:
    return (normalize01(x) * 255).astype(np.uint8)
