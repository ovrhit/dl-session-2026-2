"""Colab 원클릭 셋업.

모든 주차 노트북의 두 번째 셀에서 이걸 부른다::

    from common.setup import bootstrap
    ctx = bootstrap(week="week02_alexnet")

하는 일:
  1. 주차별 추가 의존성 설치
  2. 한글 폰트 설치 + matplotlib 등록  (이거 빼먹으면 발표 중에 라벨이 전부 두부(□)로 나온다)
  3. seed 고정
  4. device 결정 + GPU 정보 출력
  5. assets 디렉터리 경로 정리, FAST_MODE에서 쓸 에셋이 실제로 있는지 점검
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 세션 노트북이 체크포인트/대용량 에셋을 받아오는 곳.
# 사용 전에 본인 계정으로 바꿀 것. (구축 순서 4번)
HF_ASSET_REPO = os.environ.get("DL_SESSION_HF_REPO", "CHANGE-ME/dl-session-2026-2")


def in_colab() -> bool:
    return "google.colab" in sys.modules


@dataclass
class Context:
    """노트북 전체에서 돌려 쓰는 실행 컨텍스트."""

    device: str
    gpu_name: str | None
    week: str
    week_dir: Path
    assets: Path
    fast_mode: bool
    missing_assets: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.device == "cuda"

    def asset(self, name: str) -> Path:
        """에셋 경로. 없으면 무엇이 없는지 알려주고 죽는다 (조용히 빈 그림 뜨는 것보다 낫다)."""
        p = self.assets / name
        if not p.exists():
            raise FileNotFoundError(
                f"에셋 없음: {p}\n"
                f"→ 이 주차의 prep.py 를 먼저 실행해야 한다:  python {self.week}/prep.py"
            )
        return p


def _run(cmd: list[str], quiet: bool = True) -> None:
    subprocess.run(cmd, check=False, capture_output=quiet)


def _install_requirements(week_dir: Path) -> None:
    if not in_colab():
        return
    for req in (REPO_ROOT / "common" / "requirements.txt", week_dir / "requirements.txt"):
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def _install_korean_font() -> str | None:
    """한글 폰트를 matplotlib에 등록하고 폰트 이름을 돌려준다."""
    import matplotlib
    import matplotlib.font_manager as fm

    if in_colab() and not Path(_NANUM).exists():
        _run(["apt-get", "install", "-y", "-qq", "fonts-nanum"])
        _run(["fc-cache", "-f"])

    if Path(_NANUM).exists():
        fm.fontManager.addfont(_NANUM)
        name = fm.FontProperties(fname=_NANUM).get_name()
    else:
        # 로컬(Windows 등) 폴백 — 설치돼 있는 한글 폰트를 찾아 쓴다.
        candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
        installed = {f.name for f in fm.fontManager.ttflist}
        name = next((c for c in candidates if c in installed), None)

    if name:
        matplotlib.rcParams["font.family"] = name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def bootstrap(week: str, *, fast_mode: bool = True, seed: int = 42, verbose: bool = True) -> Context:
    week_dir = REPO_ROOT / week
    if not week_dir.exists():
        raise FileNotFoundError(f"주차 폴더가 없다: {week_dir}")

    _install_requirements(week_dir)
    font = _install_korean_font()
    set_seed(seed)

    from . import viz

    viz.use_presentation_style()

    device, gpu_name = "cpu", None
    try:
        import torch

        if torch.cuda.is_available():
            device, gpu_name = "cuda", torch.cuda.get_device_name(0)
    except ImportError:
        pass

    assets = week_dir / "assets"
    assets.mkdir(exist_ok=True)
    # .gitkeep 같은 숨김 파일은 "에셋이 있다"로 치면 안 된다 — 그러면 경고가 영영 안 뜬다.
    real = [p for p in assets.iterdir() if not p.name.startswith(".")]
    missing = [] if real else ["<assets 비어 있음>"]

    ctx = Context(
        device=device,
        gpu_name=gpu_name,
        week=week,
        week_dir=week_dir,
        assets=assets,
        fast_mode=fast_mode,
        missing_assets=missing,
    )

    if verbose:
        print(f"주차      : {week}")
        print(f"FAST_MODE : {fast_mode}  ({'에셋 로드 — 세션 진행용' if fast_mode else '전체 재현 — 오래 걸림'})")
        print(f"device    : {device}" + (f"  ({gpu_name})" if gpu_name else ""))
        print(f"한글 폰트 : {font or '못 찾음 — 그래프 한글이 깨질 수 있다'}")
        if missing:
            print(f"\n[주의] {assets} 가 비어 있다. FAST_MODE=True 로는 진행할 수 없다.")
            print(f"       먼저 실행:  python {week}/prep.py")

    return ctx


def download_assets(week: str, *, repo_id: str = HF_ASSET_REPO) -> Path:
    """prep.py 결과를 HF Hub에 올려뒀을 때 내려받는 경로.

    체크포인트처럼 GitHub 100MB 제한에 걸리는 것들만 여기로 보낸다.
    가벼운 PNG/GIF는 그냥 git에 커밋하는 게 세션 당일 더 안전하다.
    """
    from huggingface_hub import snapshot_download

    dest = REPO_ROOT / week / "assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{week}/*"],
        local_dir=str(REPO_ROOT / "_hf_cache"),
    )
    src = REPO_ROOT / "_hf_cache" / week
    if src.exists():
        import shutil

        shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest
