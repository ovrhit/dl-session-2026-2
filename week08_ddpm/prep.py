"""8주차 DDPM — 사전 실행 스크립트.

    !python week08_ddpm/prep.py                  # 전부 (T4 약 20분)
    !python week08_ddpm/prep.py --only A --epochs 30

생성물:
    assets/A_forward.gif          이미지가 노이즈가 되어가는 과정 ★
    assets/A_reverse.gif          노이즈에서 이미지가 떠오르는 과정 ★
    assets/A_samples.png          학습된 tiny DDPM 의 생성 결과
    assets/ckpt_ddpm.pt           체크포인트 (세션에선 이것만 로드, 약 25MB)
    assets/B_steps.png / .json    샘플링 스텝 수 vs 품질/시간
    assets/B_interpolation.png    잠재공간 보간
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import viz  # noqa: E402
from common.setup import set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000       # 논문과 같은 총 단계 수
IMG = 28


# ════════════════════════════════════════════════════════════════════
# 노이즈 스케줄 — 논문 2절
# ════════════════════════════════════════════════════════════════════

def make_schedule(T: int = T, device: str = DEVICE) -> dict[str, torch.Tensor]:
    """선형 beta 스케줄 (논문 4절: 1e-4 → 0.02)."""
    beta = torch.linspace(1e-4, 0.02, T, device=device)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, dim=0)          # ᾱ_t = Π α_s
    return {"beta": beta, "alpha": alpha, "abar": abar}


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: dict, noise: torch.Tensor | None = None):
    """forward process — 논문 식 (4).

        x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε

    핵심: t단계를 **한 번에** 계산할 수 있다. 1000번 반복할 필요가 없다.
    이 닫힌 형태가 있어서 임의의 t를 무작위로 뽑아 학습할 수 있고,
    그래서 DDPM 학습이 현실적인 시간 안에 끝난다.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    a = sched["abar"][t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise


# ════════════════════════════════════════════════════════════════════
# 아주 작은 U-Net
# ════════════════════════════════════════════════════════════════════

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """6주차 positional encoding과 **같은 식**이다. 위치 대신 '몇 번째 노이즈 단계'를 넣는다."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    """3주차 residual + 시간 조건. 't가 몇인지'를 채널마다 더해서 알려준다."""

    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, cin), nn.GroupNorm(8, cout)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(t)).view(*t.shape[:1], -1, 1, 1)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """28×28 흑백용 최소 U-Net. 하는 일은 하나: **넣은 노이즈 ε을 맞히기.**"""

    def __init__(self, base: int = 64, tdim: int = 128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2, c3 = base, base * 2, base * 4
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1, self.d2 = ResBlock(c1, c1, tdim), ResBlock(c1, c2, tdim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(c2, c3, tdim)
        self.u2 = ResBlock(c3 + c2, c2, tdim)
        self.u1 = ResBlock(c2 + c1, c1, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.tmlp(timestep_embedding(t, self.tdim))
        h0 = self.inp(x)                  # 28
        h1 = self.d1(h0, te)              # 28
        h2 = self.d2(self.pool(h1), te)   # 14
        m = self.mid(self.pool(h2), te)   # 7
        # U-Net의 skip — 3주차 residual과 발상이 같다. 해상도 정보를 되살린다.
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te)   # 14
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)   # 28
        return self.out(u)


# ════════════════════════════════════════════════════════════════════
# 학습 — 논문 Algorithm 1
# ════════════════════════════════════════════════════════════════════

def fashion_loader(batch: int = 128):
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(str(HERE / "_data"), train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch, shuffle=True, num_workers=2, drop_last=True)


def train_ddpm(epochs: int = 30) -> TinyUNet:
    set_seed(0)
    sched = make_schedule()
    model = TinyUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loader = fashion_loader()
    for ep in range(epochs):
        model.train()
        total = n = 0
        for x0, _ in loader:
            x0 = x0.to(DEVICE)
            # Algorithm 1: t를 균등하게 뽑고, 노이즈를 섞고, 그 노이즈를 맞힌다. 끝.
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            xt, noise = q_sample(x0, t, sched)
            loss = F.mse_loss(model(xt, t), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  epoch {ep + 1:2d}/{epochs}  loss={total / n:.4f}")
    return model


# ════════════════════════════════════════════════════════════════════
# 샘플링 — 논문 Algorithm 2 / DDIM
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample(model, n: int = 16, steps: int = T, eta: float = 1.0,
           sched: dict | None = None, seed: int = 0, keep_frames: bool = False):
    """steps < T 이면 단계를 건너뛴다 (DDIM). eta=0 이면 완전 결정론적.

    반환: (최종 이미지, 중간 프레임 목록)
    """
    sched = sched or make_schedule()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 1, IMG, IMG, device=DEVICE, generator=g)
    ts = torch.linspace(T - 1, 0, steps).long().to(DEVICE)
    frames = []
    model.eval()
    for i, t in enumerate(ts):
        tb = t.repeat(n)
        eps = model(x, tb)
        abar_t = sched["abar"][t]
        abar_prev = sched["abar"][ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=DEVICE)
        # 지금 x_t 에서 '원본이 뭐였을지' 되짚기
        x0 = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
        x0 = x0.clamp(-1, 1)
        sigma = eta * ((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).clamp(min=0).sqrt()
        dir_xt = (1 - abar_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        noise = torch.randn(x.shape, device=DEVICE, generator=g) if i + 1 < len(ts) else 0
        x = abar_prev.sqrt() * x0 + dir_xt + sigma * noise
        if keep_frames and (i % max(1, len(ts) // 60) == 0 or i == len(ts) - 1):
            frames.append(x.clone())
    return x, frames


def grid_uint8(x: torch.Tensor, nrow: int = 4) -> np.ndarray:
    from torchvision.utils import make_grid

    g = make_grid((x.clamp(-1, 1) + 1) / 2, nrow=nrow, padding=2)
    return (g[0].detach().cpu().numpy() * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════
# A. forward / reverse 애니메이션 + 생성
# ════════════════════════════════════════════════════════════════════

def block_a(epochs: int = 30) -> None:
    print("\n[A] forward/reverse 애니메이션 + tiny DDPM 학습")
    sched = make_schedule()

    # ── forward GIF: 실제 이미지 → 완전한 잡음 ──────────────────
    loader = fashion_loader(batch=16)
    x0, _ = next(iter(loader))
    x0 = x0.to(DEVICE)
    frames = []
    for t in np.linspace(0, T - 1, 80).astype(int):
        xt, _ = q_sample(x0, torch.full((16,), int(t), device=DEVICE), sched)
        frames.append(grid_uint8(xt))
    viz.gif(frames, ASSETS / "A_forward.gif", fps=14)
    print("  A_forward.gif 저장")

    # ── 학습 ─────────────────────────────────────────────────────
    model = train_ddpm(epochs)
    torch.save({"state": model.state_dict(), "T": T}, ASSETS / "ckpt_ddpm.pt")

    # ── reverse GIF: 잡음 → 이미지 ───────────────────────────────
    x, mid = sample(model, n=16, steps=T, sched=sched, keep_frames=True)
    viz.gif([grid_uint8(f) for f in mid], ASSETS / "A_reverse.gif", fps=14)
    fig = viz.image_grid([grid_uint8(x)], ncols=1, cell=7, cmap="gray",
                         title=f"tiny DDPM 이 만들어낸 옷 ({epochs} epoch 학습)")
    viz.save(fig, ASSETS / "A_samples.png")
    plt.close(fig)
    print("  A_reverse.gif / A_samples.png 저장")


def load_model() -> TinyUNet:
    ck = torch.load(ASSETS / "ckpt_ddpm.pt", map_location=DEVICE, weights_only=False)
    m = TinyUNet().to(DEVICE)
    m.load_state_dict(ck["state"])
    return m.eval()


# ════════════════════════════════════════════════════════════════════
# B. 스텝 수 vs 품질 / 보간
# ════════════════════════════════════════════════════════════════════

STEP_SETTINGS = [(1000, 1.0, "DDPM 1000 step"), (100, 0.0, "DDIM 100 step"),
                 (50, 0.0, "DDIM 50 step"), (10, 0.0, "DDIM 10 step")]


def block_b() -> None:
    print("\n[B] 샘플링 스텝 수 vs 품질")
    model = load_model()
    sched = make_schedule()

    fig, axes = plt.subplots(1, len(STEP_SETTINGS), figsize=(4.2 * len(STEP_SETTINGS), 4.8))
    timing = {}
    for ax, (steps, eta, label) in zip(axes, STEP_SETTINGS):
        t0 = time.perf_counter()
        x, _ = sample(model, n=16, steps=steps, eta=eta, sched=sched, seed=1)
        dt = time.perf_counter() - t0
        timing[label] = dt
        ax.imshow(grid_uint8(x), cmap="gray")
        ax.set_title(f"{label}\n{dt:.1f}초")
        ax.axis("off")
        print(f"  {label:18s} {dt:6.1f}초")
    fig.suptitle("스텝을 줄이면 얼마나 빨라지고 얼마나 나빠지나",
                 fontsize=20, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "B_steps.png")
    plt.close(fig)
    (ASSETS / "B_steps.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 보간: 시작 잡음 두 개를 섞어가며 ──────────────────────────
    # 결과가 매끄럽게 변한다 = 모델이 '외운 게' 아니라 연속적인 공간을 배웠다는 증거.
    g = torch.Generator(device=DEVICE).manual_seed(3)
    z0 = torch.randn(1, 1, IMG, IMG, device=DEVICE, generator=g)
    z1 = torch.randn(1, 1, IMG, IMG, device=DEVICE, generator=g)
    outs = []
    for a in np.linspace(0, 1, 8):
        # 구면 보간(slerp) — 선형으로 섞으면 노이즈 크기가 줄어 흐릿해진다
        th = torch.acos((z0.flatten() @ z1.flatten()) / (z0.norm() * z1.norm())).clamp(-1, 1)
        z = (torch.sin((1 - a) * th) * z0 + torch.sin(a * th) * z1) / torch.sin(th)
        x = _sample_from(model, z, sched, steps=100)
        outs.append(((x[0, 0].clamp(-1, 1) + 1) / 2).detach().cpu().numpy())
    fig = viz.image_grid(outs, ncols=8, cell=1.8, cmap="gray",
                         title="시작 잡음을 조금씩 바꾸면 결과도 조금씩 바뀐다")
    viz.save(fig, ASSETS / "B_interpolation.png")
    plt.close(fig)


@torch.no_grad()
def _sample_from(model, z: torch.Tensor, sched: dict, steps: int = 100) -> torch.Tensor:
    x = z.clone()
    ts = torch.linspace(T - 1, 0, steps).long().to(DEVICE)
    for i, t in enumerate(ts):
        eps = model(x, t.repeat(x.size(0)))
        abar_t = sched["abar"][t]
        abar_prev = sched["abar"][ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=DEVICE)
        x0 = ((x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()).clamp(-1, 1)
        x = abar_prev.sqrt() * x0 + (1 - abar_prev).clamp(min=0).sqrt() * eps
    return x


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="8주차 DDPM 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    if DEVICE == "cpu":
        print("[경고] GPU 없이 A 블록 학습은 사실상 불가능하다.")
    for key in args.only or list(BLOCKS):
        block_a(args.epochs) if key == "A" else BLOCKS[key]()
    print(f"\n완료 → {ASSETS}")
