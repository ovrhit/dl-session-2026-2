"""6주차 Transformer — 사전 실행 스크립트.

    !python week06_transformer/prep.py           # 전부 (T4 약 20분)

생성물:
    assets/A_speed.png / .json        RNN vs Transformer 학습 속도 — 논문의 '병렬화' 주장 재현
    assets/B_pos_encoding.png         sin/cos positional encoding 히트맵
    assets/B_no_pe.png / .json        위치 정보를 빼면 어떻게 되는가
    assets/B_head_patterns.png        GPT-2 head별 attention 패턴 (BertViz 대체용 백업)
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

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_corpus() -> str:
    """4주차와 같은 코퍼스를 쓴다 — 같은 데이터에서 비교해야 의미가 있다."""
    week4 = HERE.parent / "week04_rnn" / "assets" / "corpus.txt"
    if week4.exists():
        return week4.read_text(encoding="utf-8")
    local = ASSETS / "corpus.txt"
    if not local.exists():
        import urllib.request

        ASSETS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE, local)
    return local.read_text(encoding="utf-8")


# ════════════════════════════════════════════════════════════════════
# self-attention — 논문 3.2절을 그대로 옮긴 것
# ════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(QKᵀ / √d_k) V   ← 논문 식 (1)

    q: (B, H, T, d)  k: (B, H, S, d)  v: (B, H, S, d)

    √d_k 로 나누는 이유: d_k가 크면 내적값의 분산이 커져 softmax가 한 곳에 몰린다.
    그러면 기울기가 거의 0이 되어 학습이 안 된다 (논문 각주 4).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal: bool = True):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        # causal mask: 미래를 못 보게 한다. 이게 없으면 정답을 그대로 베낀다.
        mask = torch.tril(torch.ones(T, T, device=x.device)) if causal else None
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), attn


class Block(nn.Module):
    """논문 Figure 1의 디코더 블록. residual(3주차!) + LayerNorm + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # pre-LN (2017 원논문은 post-LN이었지만 지금은 pre-LN이 표준이다 — C 블록 참고)
        a, _ = self.attn(self.ln1(x))
        x = x + a                       # ← 3주차 residual
        return x + self.ff(self.ln2(x))


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """논문 식 (2)(3). 위치를 여러 주파수의 sin/cos 로 표현한다."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    i = torch.arange(0, d_model, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, max_len: int = 256, use_pe: bool = True):
        super().__init__()
        self.use_pe = use_pe
        self.emb = nn.Embedding(vocab, d_model)
        self.register_buffer("pe", sinusoidal_pe(max_len, d_model))
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.emb(x)
        if self.use_pe:
            h = h + self.pe[: x.size(1)]
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


class LSTMLM(nn.Module):
    """비교 대상. 파라미터 수를 Transformer와 비슷하게 맞춘다."""

    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.rnn = nn.LSTM(d_model, d_model * 2, n_layers, batch_first=True)
        self.head = nn.Linear(d_model * 2, vocab)

    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])


# ════════════════════════════════════════════════════════════════════
# A. 병렬화 — 같은 데이터, 같은 시간, 누가 더 배우나
# ════════════════════════════════════════════════════════════════════

SEQ = 128


def _data():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, len(chars), stoi, {i: c for c, i in stoi.items()}


def _iter(ids, batch: int, seq: int, steps: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(ids) - seq - 1, (batch,), generator=g)
        x = torch.stack([ids[i:i + seq] for i in ix])
        y = torch.stack([ids[i + 1:i + seq + 1] for i in ix])
        yield x.to(DEVICE), y.to(DEVICE)


def train_timed(model, ids, vocab, steps: int = 600, batch: int = 64, seq: int = SEQ) -> dict:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, t0 = [], None
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    for i, (x, y) in enumerate(_iter(ids, batch, seq, steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % 20 == 0:
            losses.append(loss.item())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    return {"seconds": elapsed, "losses": losses, "final_loss": losses[-1],
            "params": n_params, "steps": steps}


def block_a(steps: int = 600) -> None:
    print("\n[A] RNN vs Transformer 학습 속도")
    ids, V, _, _ = _data()

    out = {}
    for name, mk in (("LSTM", lambda: LSTMLM(V)), ("Transformer", lambda: TinyTransformerLM(V))):
        set_seed(0)
        out[name] = train_timed(mk(), ids, V, steps)
        r = out[name]
        print(f"  [{name:12s}] {r['seconds']:6.1f}초  "
              f"final_loss={r['final_loss']:.3f}  params={r['params'] / 1e6:.1f}M")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    viz.bars(list(out), [out[k]["seconds"] for k in out],
             ylabel="초", title=f"{steps} step 학습에 걸린 시간 (배치 64 × 길이 {SEQ})",
             colors=[viz.OLD, viz.NEW], ax=axes[0])
    for i, k in enumerate(out):
        axes[0].text(i, out[k]["seconds"], f"{out[k]['seconds']:.0f}s",
                     ha="center", va="bottom", fontsize=14)
    viz.curves({k: v["losses"] for k, v in out.items()},
               xlabel="step (×20)", ylabel="loss", title="같은 step에서의 손실",
               colors={"LSTM": viz.OLD, "Transformer": viz.NEW}, ax=axes[1])
    fig.tight_layout()
    viz.save(fig, ASSETS / "A_speed.png")
    plt.close(fig)

    (ASSETS / "A_speed.json").write_text(
        json.dumps({"seq_len": SEQ, "device": DEVICE, "results": out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════
# B. positional encoding / head 패턴
# ════════════════════════════════════════════════════════════════════

def block_b(steps: int = 400) -> None:
    print("\n[B] positional encoding")
    pe = sinusoidal_pe(128, 128).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    im = axes[0].imshow(pe, aspect="auto", cmap="RdBu_r")
    axes[0].set_xlabel("임베딩 차원")
    axes[0].set_ylabel("위치 (토큰 순서)")
    axes[0].set_title("sin/cos positional encoding — 논문 식 (2)(3)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    for p in (0, 10, 40):
        axes[1].plot(pe[p], label=f"위치 {p}")
    axes[1].set_xlabel("임베딩 차원")
    axes[1].set_title("위치마다 다른 '지문'이 찍힌다")
    axes[1].legend()
    fig.tight_layout()
    viz.save(fig, ASSETS / "B_pos_encoding.png")
    plt.close(fig)

    # ── 위치 정보를 빼면? ─────────────────────────────────────────
    # self-attention은 집합 연산이라 순서를 전혀 모른다.
    # PE를 빼면 "단어 가방"이 되어 언어모델 성능이 눈에 띄게 떨어져야 한다.
    print("  위치 정보 유/무 비교")
    ids, V, _, _ = _data()
    res = {}
    for use_pe, label in ((True, "PE 있음"), (False, "PE 없음")):
        set_seed(0)
        res[label] = train_timed(TinyTransformerLM(V, use_pe=use_pe), ids, V, steps)
        print(f"  [{label}] final_loss={res[label]['final_loss']:.3f}")

    fig, ax = plt.subplots()
    viz.curves({k: v["losses"] for k, v in res.items()},
               xlabel="step (×20)", ylabel="loss",
               title="위치 정보를 빼면 Transformer는 '단어 가방'이 된다",
               colors={"PE 있음": viz.NEW, "PE 없음": viz.OLD}, ax=ax, annotate_last=False)
    ax.legend()
    viz.save(fig, ASSETS / "B_no_pe.png")
    plt.close(fig)
    (ASSETS / "B_no_pe.json").write_text(
        json.dumps({k: {"final_loss": v["final_loss"], "losses": v["losses"]}
                    for k, v in res.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


DEMO_TEXT = "The cat sat on the mat because it was tired."


def block_b_heads(text: str = DEMO_TEXT, model_id: str = "gpt2") -> None:
    """BertViz가 Colab에서 안 뜰 때를 대비한 정적 백업 그림."""
    print("\n[B+] head별 attention 패턴")
    from transformers import AutoModel, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(model_id)
    m = AutoModel.from_pretrained(model_id, attn_implementation="eager",
                                  output_attentions=True).eval().to(DEVICE)
    ins = tk(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        attn = m(**ins).attentions          # (층) × (B, head, T, T)
    toks = [t.replace("Ġ", "") for t in tk.convert_ids_to_tokens(ins.input_ids[0])]

    picks = [(0, 0), (0, 5), (5, 3), (8, 7), (10, 2), (11, 9)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (layer, head) in zip(axes.ravel(), picks):
        A = attn[layer][0, head].cpu().numpy()
        viz.heatmap(A, xlabels=toks, ylabels=toks, ax=ax,
                    title=f"layer {layer} · head {head}", cmap="Blues")
    fig.suptitle(f'"{text}" — head마다 다른 것을 본다', fontsize=20, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "B_head_patterns.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# C. 2017 블록 vs 지금 LLM 블록  (에셋 없이 노트북에서 라이브)
# ════════════════════════════════════════════════════════════════════

MODERN_DIFF = [
    {"항목": "위치 정보", "2017 Transformer": "sin/cos 절대 위치를 더함",
     "지금 (Llama 계열)": "RoPE — Q·K를 각도만큼 회전", "왜 바뀌었나": "학습 때보다 긴 입력에 더 잘 버틴다"},
    {"항목": "정규화", "2017 Transformer": "LayerNorm, 블록 뒤 (post-LN)",
     "지금 (Llama 계열)": "RMSNorm, 블록 앞 (pre-LN)", "왜 바뀌었나": "깊게 쌓아도 학습이 안정적이고 더 싸다"},
    {"항목": "FFN 활성함수", "2017 Transformer": "ReLU", "지금 (Llama 계열)": "SwiGLU",
     "왜 바뀌었나": "같은 파라미터로 성능이 조금 더 좋다"},
    {"항목": "attention head", "2017 Transformer": "Q·K·V head 수가 모두 같음",
     "지금 (Llama 계열)": "GQA — K·V head를 공유", "왜 바뀌었나": "추론할 때 KV 캐시 메모리를 줄인다"},
    {"항목": "뼈대", "2017 Transformer": "residual + attention + FFN",
     "지금 (Llama 계열)": "residual + attention + FFN", "왜 바뀌었나": "— 안 바뀌었다"},
]


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "H": block_b_heads}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="6주차 Transformer 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    for key in args.only or list(BLOCKS):
        if key == "A":
            block_a(args.steps)
        elif key == "B":
            block_b(max(200, args.steps // 2))
        else:
            block_b_heads()
    print(f"\n완료 → {ASSETS}")
