"""3주차 ResNet — 사전 실행 스크립트.

    !python week03_resnet/prep.py                 # 전부 (T4 약 50분)
    !python week03_resnet/prep.py --only A --epochs 20

예상 소요 (T4): A 약 45분 / B 1분 / C 2분.
A가 이 세션에서 제일 오래 걸린다. **최소 세션 3일 전에는 돌려둘 것.**

생성물:
    assets/A_degradation.png      plain-20/56 vs ResNet-20/56 학습곡선 (논문 Fig.6)
    assets/A_degradation.json     곡선 원본 수치
    assets/B_gradient_norm.png    skip 유무에 따른 층별 gradient norm
    assets/C_gradcam.png          ResNet-50이 어디를 보고 판단했나
"""

from __future__ import annotations

import argparse
import json
import sys
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


# ════════════════════════════════════════════════════════════════════
# CIFAR용 ResNet / plain net  (He et al. 4.2절: 6n+2 층)
# ════════════════════════════════════════════════════════════════════
#
# 두 네트워크의 차이는 forward 마지막 줄의 `+ identity` 딱 하나다.
# 층 수, 채널 수, BatchNorm, 초기화, 옵티마이저 전부 동일하게 맞춘다.
# 그래야 "깊이 때문에 생긴 문제"와 "residual이 고친 것"을 분리해서 볼 수 있다.

class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # 채널/해상도가 바뀌는 지점에서는 shortcut도 맞춰줘야 더할 수 있다 (논문의 option B)
        self.short = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if residual and (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.short(x)   # ← plain net과의 유일한 차이
        return F.relu(out)


class CifarNet(nn.Module):
    def __init__(self, depth: int = 20, residual: bool = True, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth는 6n+2 여야 한다 (20, 32, 44, 56 ...)"
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        layers, cin = [], 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            for i in range(n):
                layers.append(Block(cin, cout, stride if i == 0 else 1, residual))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def make(depth: int, residual: bool) -> CifarNet:
    return CifarNet(depth, residual)


def name_of(depth: int, residual: bool) -> str:
    return f"{'ResNet' if residual else 'plain'}-{depth}"


# ════════════════════════════════════════════════════════════════════
# A. degradation problem — 논문 Fig.6 재현
# ════════════════════════════════════════════════════════════════════

def _loaders(batch_size: int = 128):
    from torchvision import datasets, transforms

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    train_tf = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), norm]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    root = str(HERE / "_data")
    tr = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    te = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    mk = torch.utils.data.DataLoader
    return (mk(tr, batch_size, shuffle=True, num_workers=2, drop_last=True),
            mk(te, 512, shuffle=False, num_workers=2))


@torch.no_grad()
def _eval_err(model, loader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.size(0)
    return 1 - correct / total


def train(depth: int, residual: bool, epochs: int, seed: int = 0) -> dict:
    set_seed(seed)
    train_loader, test_loader = _loaders()
    model = make(depth, residual).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.1, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    hist = {"train_err": [], "test_err": []}
    tag = name_of(depth, residual)
    for ep in range(epochs):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            F.cross_entropy(out, yb).backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        hist["train_err"].append(1 - correct / total)
        hist["test_err"].append(_eval_err(model, test_loader))
        print(f"  [{tag:10s}] epoch {ep + 1:2d}/{epochs} "
              f" train_err={hist['train_err'][-1]:.3f}  test_err={hist['test_err'][-1]:.3f}")
    return hist


CONFIGS = [(20, False), (56, False), (20, True), (56, True)]
COLORS = {
    "plain-20": "#F0A868", "plain-56": viz.OLD,
    "ResNet-20": "#7FB8DD", "ResNet-56": viz.NEW,
}


def block_a(epochs: int = 20) -> None:
    print("\n[A] degradation problem — 논문 Fig.6 재현")
    results = {name_of(d, r): train(d, r, epochs) for d, r in CONFIGS}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for ax, key, title in (
        (axes[0], "train_err", "학습 오차 (training error)"),
        (axes[1], "test_err", "테스트 오차 (test error)"),
    ):
        viz.curves(
            {k: v[key] for k, v in results.items()},
            xlabel="epoch", ylabel="오차", title=title,
            colors=COLORS, ax=ax, annotate_last=False,
        )
        ax.legend(loc="upper right")
    fig.suptitle(
        "plain net은 깊을수록 나빠진다 — 그것도 '학습' 오차가", fontsize=20, fontweight="bold"
    )
    fig.tight_layout()
    viz.save(fig, ASSETS / "A_degradation.png")
    plt.close(fig)

    final = {k: {"train_err": v["train_err"][-1], "test_err": v["test_err"][-1]}
             for k, v in results.items()}
    (ASSETS / "A_degradation.json").write_text(
        json.dumps({"epochs": epochs, "final": final, "history": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n  최종 학습 오차:")
    for k, v in final.items():
        print(f"    {k:10s} {v['train_err']:.3f}")


# ════════════════════════════════════════════════════════════════════
# B. skip 유무에 따른 층별 gradient norm
# ════════════════════════════════════════════════════════════════════

def layerwise_grad_norms(model: CifarNet, batch: int = 64) -> list[float]:
    """한 번 역전파해서 conv 층별 gradient L2 norm을 깊은 쪽→얕은 쪽 순으로 뽑는다."""
    model = model.to(DEVICE).train()
    x = torch.randn(batch, 3, 32, 32, device=DEVICE)
    y = torch.randint(0, 10, (batch,), device=DEVICE)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    return [
        float(m.weight.grad.norm())
        for m in model.modules()
        if isinstance(m, nn.Conv2d) and m.weight.grad is not None
    ]


def block_b(depth: int = 56) -> None:
    print("\n[B] 층별 gradient norm")
    set_seed(0)
    plain = layerwise_grad_norms(make(depth, residual=False))
    set_seed(0)
    res = layerwise_grad_norms(make(depth, residual=True))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    # x축을 '입력에 가까운 층 → 출력에 가까운 층' 으로 둔다.
    ax.plot(range(len(plain)), plain, color=viz.OLD, label=f"plain-{depth}")
    ax.plot(range(len(res)), res, color=viz.NEW, label=f"ResNet-{depth}")
    ax.set_yscale("log")
    ax.set_xlabel("conv 층 번호  (0 = 입력 쪽,  오른쪽 = 출력 쪽)")
    ax.set_ylabel("gradient L2 norm  (로그 스케일)")
    ax.set_title("입력 쪽으로 갈수록 기울기가 살아남는가")
    ax.legend()
    viz.save(fig, ASSETS / "B_gradient_norm.png")
    plt.close(fig)

    print(f"  plain  입력층/출력층 비율 = {plain[0] / plain[-1]:.2e}")
    print(f"  ResNet 입력층/출력층 비율 = {res[0] / res[-1]:.2e}")


# ════════════════════════════════════════════════════════════════════
# C. Grad-CAM — ResNet-50은 어디를 보고 판단했나
# ════════════════════════════════════════════════════════════════════

def gradcam(model, layer, x: torch.Tensor, class_idx: int | None = None):
    """마지막 conv 특징맵을, '그 클래스 점수에 대한 기울기'로 가중합한다.

    기울기가 크다 = 그 채널이 이 판단에 많이 기여했다.
    """
    feats, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: feats.setdefault("v", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("v", go[0]))
    try:
        out = model(x)
        idx = int(out.argmax(1)) if class_idx is None else class_idx
        model.zero_grad(set_to_none=True)
        out[0, idx].backward()
    finally:
        h1.remove()
        h2.remove()

    w = grads["v"].mean(dim=(2, 3), keepdim=True)          # 채널별 중요도
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))     # 양의 기여만
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return viz.normalize01(cam[0, 0].detach().cpu().numpy()), idx


def block_c() -> None:
    print("\n[C] Grad-CAM")
    from PIL import Image
    from torchvision.models import ResNet50_Weights, resnet50

    samples = sorted((HERE.parent / "week02_alexnet" / "assets" / "samples").glob("*"))
    samples = [p for p in samples if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not samples:
        print("  2주차 assets/samples/ 에 이미지가 없다. 2주차 prep를 먼저 돌릴 것.")
        return

    w = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=w).eval().to(DEVICE)
    tf, cats = w.transforms(), w.meta["categories"]

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)
    for c, p in enumerate(samples):
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)
        cam, idx = gradcam(model, model.layer4[-1], x)
        base = np.asarray(img.resize((x.shape[-1], x.shape[-2]))) / 255.0
        axes[0, c].imshow(base)
        axes[0, c].set_title(p.name, fontsize=13)
        axes[1, c].imshow(base)
        axes[1, c].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, c].set_title(f"판단: {cats[idx]}", fontsize=13)
        for r in (0, 1):
            axes[r, c].axis("off")
    fig.suptitle("ResNet-50은 어디를 보고 그렇게 판단했나 (Grad-CAM)",
                 fontsize=20, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "C_gradcam.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "C": block_c}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3주차 ResNet 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    if DEVICE == "cpu":
        print("[경고] GPU 없이 A 블록은 사실상 불가능하다. Colab 런타임 유형을 확인할 것.")

    for key in args.only or list(BLOCKS):
        block_a(args.epochs) if key == "A" else BLOCKS[key]()
    print(f"\n완료 → {ASSETS}")
