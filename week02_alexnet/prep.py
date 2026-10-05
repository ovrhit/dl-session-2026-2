"""2주차 AlexNet — 사전 실행 스크립트.

세션 1주일 전에 Colab(T4)에서 한 번 돌려서 assets/ 를 채운다.

    !python week02_alexnet/prep.py            # 전부
    !python week02_alexnet/prep.py --only A   # A 블록만

예상 소요 (T4): A 약 12분 / B 1분 미만 / C 약 3분 (가중치 다운로드 포함).

생성물:
    assets/A_relu_vs_tanh.png        논문 Fig.1 재현 곡선
    assets/A_relu_vs_tanh.json       곡선 원본 수치 (노트북에서 다시 그릴 때 사용)
    assets/B_conv1_filters.png       논문 Fig.3 재현 — 1층 conv 필터 그리드
    assets/B_featuremaps.png         층이 깊어질수록 추상화되는 특징맵
    assets/C_predictions.png         AlexNet vs ConvNeXt vs CLIP 예측 비교
    assets/C_predictions.json        비교 결과 원본
    assets/samples/*.jpg             비교에 쓴 이미지
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
SAMPLES = ASSETS / "samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════
# A. 논문 Fig.1 재현 — ReLU vs tanh 수렴 속도
# ════════════════════════════════════════════════════════════════════
#
# 논문 주장: 같은 4층 CNN에서 ReLU가 tanh보다 학습 오차 25% 도달까지 6배 빠르다.
# 여기서는 CIFAR-10 + 5층 CNN으로 축소 재현한다. 비교가 공정하려면
# 활성함수 말고는 전부 같아야 한다 — 같은 seed, 같은 init, 같은 LR, 같은 배치 순서.

class SmallConvNet(nn.Module):
    """AlexNet을 CIFAR-10 크기로 줄인 것. 활성함수만 갈아끼울 수 있게 해뒀다."""

    def __init__(self, act: str = "relu"):
        super().__init__()
        self.act_name = act
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), self._act(),
            nn.Conv2d(64, 64, 3, padding=1), self._act(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), self._act(),
            nn.Conv2d(128, 128, 3, padding=1), self._act(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), self._act(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), self._act(),
            nn.Linear(512, 10),
        )

    def _act(self):
        return nn.ReLU(inplace=True) if self.act_name == "relu" else nn.Tanh()

    def forward(self, x):
        return self.classifier(self.features(x))


def _cifar_loader(batch_size: int = 256, train: bool = True):
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    ds = datasets.CIFAR10(root=str(HERE / "_data"), train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=2, drop_last=train
    )


def _train_one(act: str, epochs: int, seed: int = 0) -> dict:
    set_seed(seed)  # 두 모델이 완전히 같은 init/배치 순서에서 출발하도록
    loader = _cifar_loader()
    model = SmallConvNet(act).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    hist = {"train_err": [], "train_loss": []}
    for ep in range(epochs):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        err = 1 - correct / total
        hist["train_err"].append(err)
        hist["train_loss"].append(loss_sum / total)
        print(f"  [{act:4s}] epoch {ep + 1:2d}/{epochs}  train_err={err:.3f}")
    return hist


def _epochs_to_reach(errs: list[float], target: float) -> float | None:
    for i, e in enumerate(errs):
        if e <= target:
            return i + 1
    return None


def block_a(epochs: int = 25) -> None:
    print("\n[A] ReLU vs tanh — 논문 Fig.1 재현")
    results = {act: _train_one(act, epochs) for act in ("relu", "tanh")}

    target = 0.25  # 논문과 같은 '학습 오차 25%' 기준
    reach = {a: _epochs_to_reach(r["train_err"], target) for a, r in results.items()}
    speedup = (
        reach["tanh"] / reach["relu"]
        if reach["relu"] and reach["tanh"]
        else None
    )

    fig, ax = plt.subplots()
    viz.curves(
        {"ReLU": results["relu"]["train_err"], "tanh": results["tanh"]["train_err"]},
        xlabel="epoch",
        ylabel="학습 오차 (training error)",
        title="같은 네트워크, 활성함수만 다르게 — 논문 Fig.1 재현",
        colors={"ReLU": viz.NEW, "tanh": viz.OLD},
        ax=ax,
    )
    ax.axhline(target, ls="--", c="gray", lw=1.6)
    ax.text(0.3, target + 0.012, "학습 오차 25%", color="gray", fontsize=13)
    for act, color in (("relu", viz.NEW), ("tanh", viz.OLD)):
        if reach[act]:
            ax.axvline(reach[act] - 1, ls=":", c=color, lw=1.6)
    if speedup:
        ax.set_title(
            f"활성함수만 바꿨는데 25% 도달이 {speedup:.1f}배 차이 — 논문 Fig.1 재현"
        )
    viz.save(fig, ASSETS / "A_relu_vs_tanh.png")
    plt.close(fig)

    (ASSETS / "A_relu_vs_tanh.json").write_text(
        json.dumps(
            {"epochs": epochs, "target_err": target, "reach_epoch": reach,
             "speedup": speedup, "history": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  → 25% 도달: ReLU {reach['relu']} epoch / tanh {reach['tanh']} epoch")


# ════════════════════════════════════════════════════════════════════
# B. 논문 Fig.3 재현 — 1층 conv 필터 + 특징맵
# ════════════════════════════════════════════════════════════════════

def load_alexnet():
    from torchvision.models import AlexNet_Weights, alexnet

    w = AlexNet_Weights.IMAGENET1K_V1
    return alexnet(weights=w).eval().to(DEVICE), w


def conv1_filter_grid(model) -> plt.Figure:
    """첫 conv 층 가중치 (64, 3, 11, 11) 를 그대로 RGB 이미지로 본다.

    아무도 '엣지 검출기를 만들어라'라고 가르치지 않았는데
    경사하강법만으로 Gabor 필터와 컬러 블롭이 나온다 — 이게 이 주차의 그림.
    """
    w = model.features[0].weight.detach().cpu()  # (64, 3, 11, 11)
    imgs = [viz.normalize01(w[i].permute(1, 2, 0).numpy()) for i in range(w.shape[0])]
    return viz.image_grid(
        imgs, ncols=8, cell=1.2,
        title="AlexNet 1층 conv 필터 64개 — 학습으로 저절로 생긴 엣지/컬러 검출기",
    )


def block_b() -> None:
    print("\n[B] conv1 필터 + 특징맵")
    model, weights = load_alexnet()

    viz.save(conv1_filter_grid(model), ASSETS / "B_conv1_filters.png")
    plt.close("all")

    img = _sample_images(weights)[0][1]
    x = weights.transforms()(img).unsqueeze(0).to(DEVICE)

    # ReLU 직후 층들에서 특징맵을 뽑아 층이 깊어질수록 뭘 보는지 비교한다.
    taps = {"conv1 (엣지·색)": 1, "conv3 (질감)": 7, "conv5 (부분 형태)": 11}
    fig, axes = plt.subplots(len(taps), 8, figsize=(13, 1.7 * len(taps) + 1.2))
    h = x
    grabbed = {}
    for i, layer in enumerate(model.features):
        h = layer(h)
        for name, idx in taps.items():
            if i == idx:
                grabbed[name] = h.detach().cpu()[0]
    for r, (name, fmap) in enumerate(grabbed.items()):
        # 평균 활성이 큰 채널 8개만 — 죽어 있는 채널을 보여줘봐야 의미 없다
        order = fmap.mean(dim=(1, 2)).argsort(descending=True)[:8]
        for c, ch in enumerate(order):
            axes[r, c].imshow(viz.normalize01(fmap[ch].numpy()), cmap="viridis")
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(name)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("층이 깊어질수록: 엣지 → 질감 → 부분 형태", fontsize=19, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "B_featuremaps.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# C. 2012 vs 2022 — 같은 이미지, 다른 모델
# ════════════════════════════════════════════════════════════════════

# 세션에서 제일 반응이 좋은 건 부원이 직접 가져온 사진이다.
# assets/samples/ 에 .jpg 를 넣어두면 그걸 쓰고, 비어 있으면 아래를 내려받는다.
FALLBACK_IMAGES = {
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "deeplab.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png",
}


def _sample_images(weights=None):
    from PIL import Image

    SAMPLES.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not existing:
        import urllib.request

        print("  assets/samples/ 가 비어 있다 → 기본 이미지를 내려받는다.")
        print("  (세션 전에 본인/부원 사진을 여기에 넣어두면 훨씬 재밌다)")
        for name, url in FALLBACK_IMAGES.items():
            try:
                urllib.request.urlretrieve(url, SAMPLES / name)
            except Exception as e:  # 네트워크는 언제든 실패한다. 조용히 넘기지 말 것.
                print(f"  [실패] {name}: {e}")
        existing = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not existing:
        raise RuntimeError(
            f"{SAMPLES} 에 이미지가 하나도 없다. 사진 2~3장을 직접 넣고 다시 실행할 것."
        )
    return [(p.name, Image.open(p).convert("RGB")) for p in existing]


def _topk(logits, categories, k=3):
    prob = torch.softmax(logits, dim=-1)[0]
    top = prob.topk(k)
    return [(categories[i], float(p)) for p, i in zip(top.values, top.indices)]


def block_c() -> None:
    print("\n[C] AlexNet vs ConvNeXt vs CLIP")
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    images = _sample_images()
    alex, alex_w = load_alexnet()
    cvw = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    cvn = convnext_tiny(weights=cvw).eval().to(DEVICE)

    from transformers import CLIPModel, CLIPProcessor

    clip_id = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_id).eval().to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained(clip_id)

    # CLIP은 분류기가 없다. 라벨을 '문장'으로 주면 그중 뭐가 맞는지 고른다.
    # → 학습 때 본 적 없는 라벨도 즉석에서 만들 수 있다는 게 핵심.
    clip_labels = [
        "a photo of a dog", "a photo of a cat", "a photo of a person",
        "a photo of food", "a photo of a car", "a landscape photo",
        "a photo of a building", "a photo of a bird",
    ]

    rows = []
    with torch.no_grad():
        for name, img in images:
            a = _topk(alex(alex_w.transforms()(img).unsqueeze(0).to(DEVICE)), alex_w.meta["categories"])
            c = _topk(cvn(cvw.transforms()(img).unsqueeze(0).to(DEVICE)), cvw.meta["categories"])
            ins = clip_proc(text=clip_labels, images=img, return_tensors="pt", padding=True).to(DEVICE)
            probs = clip(**ins).logits_per_image.softmax(dim=1)[0]
            top = probs.topk(3)
            cl = [(clip_labels[i], float(p)) for p, i in zip(top.values, top.indices)]
            rows.append({"image": name, "alexnet": a, "convnext": c, "clip": cl})
            print(f"  {name}: AlexNet={a[0][0]}  ConvNeXt={c[0][0]}  CLIP={cl[0][0]}")

    (ASSETS / "C_predictions.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.4 * n), gridspec_kw={"width_ratios": [1, 2.1]})
    axes = np.atleast_2d(axes)
    for r, ((name, img), row) in enumerate(zip(images, rows)):
        axes[r, 0].imshow(img)
        axes[r, 0].axis("off")
        lines = []
        for key, title, color in (("alexnet", "AlexNet (2012)", viz.OLD),
                                  ("convnext", "ConvNeXt (2022)", viz.NEW),
                                  ("clip", "CLIP zero-shot (2021)", "#009E73")):
            preds = "   ".join(f"{lbl} {p:.0%}" for lbl, p in row[key][:2])
            lines.append((title, preds, color))
        axes[r, 1].axis("off")
        for i, (title, preds, color) in enumerate(lines):
            y = 0.82 - i * 0.3
            axes[r, 1].text(0, y, title, fontsize=15, fontweight="bold", color=color)
            axes[r, 1].text(0, y - 0.12, preds, fontsize=14)
    fig.suptitle("같은 사진, 10년 차이", fontsize=20, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "C_predictions.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "C": block_c}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2주차 AlexNet 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*", help="일부 블록만 실행")
    ap.add_argument("--epochs", type=int, default=25, help="A 블록 학습 epoch 수")
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    if DEVICE == "cpu":
        print("[경고] GPU가 없다. A 블록은 매우 오래 걸린다. Colab 런타임 유형을 확인할 것.")

    for key in args.only or list(BLOCKS):
        if key == "A":
            block_a(args.epochs)
        else:
            BLOCKS[key]()
    print(f"\n완료 → {ASSETS}")
