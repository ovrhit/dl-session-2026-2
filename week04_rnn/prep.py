"""4주차 RNN 언어모델 — 사전 실행 스크립트.

    !python week04_rnn/prep.py                  # 전부 (T4 약 15분)
    !python week04_rnn/prep.py --only B

코퍼스: `assets/corpus.txt` 가 있으면 그걸 쓴다.
        → **한국어 텍스트(가사, 소설, 동아리 공지 아무거나)를 넣어두면 B 블록이 훨씬 재밌다.**
        없으면 tiny-shakespeare 를 내려받는다.

생성물:
    assets/A_perplexity.png / .json     n-gram vs RNN vs LSTM perplexity (논문 Table 재현)
    assets/B_samples.json               학습 epoch별 생성 샘플 (횡설수설 → 문장)
    assets/B_samples.png                위 샘플을 표로 렌더링한 것
    assets/B_vanishing.png              시점 거리별 gradient norm — RNN vs LSTM
    assets/C_modern.json                GPT-2 / 소형 LLM 과의 비교
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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


# ════════════════════════════════════════════════════════════════════
# 코퍼스
# ════════════════════════════════════════════════════════════════════

def load_corpus() -> str:
    path = ASSETS / "corpus.txt"
    if not path.exists():
        import urllib.request

        ASSETS.mkdir(exist_ok=True)
        print("  assets/corpus.txt 가 없다 → tiny-shakespeare 를 내려받는다.")
        print("  (한국어 텍스트 파일을 여기에 넣어두면 B 블록 생성 데모가 훨씬 재밌다)")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, path)
    return path.read_text(encoding="utf-8")


def split_text(text: str, val_frac: float = 0.1) -> tuple[str, str]:
    cut = int(len(text) * (1 - val_frac))
    return text[:cut], text[cut:]


# ════════════════════════════════════════════════════════════════════
# A. n-gram vs RNN — 논문 Table 재현
# ════════════════════════════════════════════════════════════════════
#
# Mikolov 2010의 핵심 표는 "RNN LM이 백오프 n-gram보다 perplexity가 낮다" 이다.
# perplexity = exp(평균 음의 로그가능도).
# 직관: "다음 단어를 고를 때 실질적으로 몇 개 중에서 헤매고 있는가". 낮을수록 좋다.

UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return text.lower().replace("\n", " \n ").split()


def build_vocab(tokens: list[str], max_size: int = 10000) -> dict[str, int]:
    common = [w for w, _ in Counter(tokens).most_common(max_size - 1)]
    return {w: i for i, w in enumerate([UNK] + common)}


class StupidBackoffNgram:
    """논문 시절 베이스라인(Katz/KN 백오프)의 단순화 버전.

    trigram → bigram → unigram 순으로 내려가며 볼 때마다 0.4를 곱해 벌점을 준다.
    (Brants et al. 2007) 진짜 Kneser-Ney는 아니지만, 요점인
    '못 본 n-gram은 더 짧은 문맥으로 후퇴한다'는 동작은 같다.
    """

    def __init__(self, order: int = 3, alpha: float = 0.4):
        self.order, self.alpha = order, alpha
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.totals = [Counter() for _ in range(order)]
        self.vocab_size = 0

    def fit(self, ids: list[int], vocab_size: int) -> "StupidBackoffNgram":
        self.vocab_size = vocab_size
        for n in range(self.order):
            for i in range(n, len(ids)):
                ctx = tuple(ids[i - n:i])
                self.counts[n][ctx][ids[i]] += 1
                self.totals[n][ctx] += 1
        return self

    def prob(self, ctx: tuple[int, ...], w: int) -> float:
        for n in range(min(self.order - 1, len(ctx)), -1, -1):
            c = ctx[len(ctx) - n:] if n else ()
            if self.totals[n][c] > 0:
                hit = self.counts[n][c][w]
                if hit > 0:
                    return (self.alpha ** (self.order - 1 - n)) * hit / self.totals[n][c]
        return 1.0 / self.vocab_size   # 그래도 못 봤으면 균등분포로 최후 후퇴

    def perplexity(self, ids: list[int]) -> float:
        nll = 0.0
        for i in range(self.order - 1, len(ids)):
            p = self.prob(tuple(ids[i - self.order + 1:i]), ids[i])
            nll -= math.log(max(p, 1e-12))
        return math.exp(nll / (len(ids) - self.order + 1))


class WordLM(nn.Module):
    """Mikolov의 모델 그대로: 임베딩 → 순환층 → 출력. cell만 갈아끼운다."""

    def __init__(self, vocab: int, emb: int = 200, hid: int = 300, cell: str = "rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        Cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = Cell(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, vocab)

    def forward(self, x, state=None):
        out, state = self.rnn(self.emb(x), state)
        return self.head(out), state


def _batchify(ids: list[int], batch: int, seq: int):
    t = torch.tensor(ids, dtype=torch.long)
    n = (len(t) - 1) // (batch * seq) * (batch * seq)
    x = t[:n].view(batch, -1)
    y = t[1:n + 1].view(batch, -1)
    for i in range(0, x.size(1), seq):
        yield x[:, i:i + seq], y[:, i:i + seq]


def train_word_lm(cell: str, train_ids, val_ids, vocab_size, epochs: int = 5) -> tuple[float, list[float]]:
    set_seed(0)
    model = WordLM(vocab_size, cell=cell).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    curve = []
    for ep in range(epochs):
        model.train()
        for xb, yb in _batchify(train_ids, 32, 35):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            # RNN은 기울기가 터지는 것도 문제다. 논문 시절부터 clip은 필수였다.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ppl = eval_word_lm(model, val_ids, vocab_size)
        curve.append(ppl)
        print(f"  [{cell:4s}] epoch {ep + 1}/{epochs}  val_ppl={ppl:.1f}")
    return curve[-1], curve


@torch.no_grad()
def eval_word_lm(model, ids, vocab_size) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in _batchify(ids, 32, 35):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="sum")
        total += loss.item()
        n += yb.numel()
    return math.exp(total / n)


def block_a(epochs: int = 5) -> None:
    print("\n[A] n-gram vs RNN perplexity")
    train_txt, val_txt = split_text(load_corpus())
    tr_tok, va_tok = tokenize(train_txt), tokenize(val_txt)
    vocab = build_vocab(tr_tok)
    enc = lambda toks: [vocab.get(w, 0) for w in toks]  # noqa: E731
    tr_ids, va_ids = enc(tr_tok), enc(va_tok)
    V = len(vocab)
    print(f"  단어 수 train={len(tr_ids):,} val={len(va_ids):,}  vocab={V:,}")

    results = {}
    for order, label in ((2, "bigram"), (3, "trigram")):
        ppl = StupidBackoffNgram(order).fit(tr_ids, V).perplexity(va_ids)
        results[label] = ppl
        print(f"  [{label}] val_ppl={ppl:.1f}")

    curves = {}
    for cell, label in (("rnn", "RNN"), ("lstm", "LSTM")):
        ppl, curve = train_word_lm(cell, tr_ids, va_ids, V, epochs)
        results[label] = ppl
        curves[label] = curve

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    order = ["bigram", "trigram", "RNN", "LSTM"]
    viz.bars(order, [results[k] for k in order],
             ylabel="validation perplexity (낮을수록 좋음)",
             title="논문의 주장: RNN LM이 n-gram보다 낫다",
             colors=[viz.OLD, viz.OLD, viz.NEW, "#009E73"], ax=axes[0])
    for i, k in enumerate(order):
        axes[0].text(i, results[k], f"{results[k]:.0f}", ha="center", va="bottom", fontsize=14)
    viz.curves(curves, xlabel="epoch", ylabel="val perplexity",
               title="학습이 진행되면서", ax=axes[1])
    fig.tight_layout()
    viz.save(fig, ASSETS / "A_perplexity.png")
    plt.close(fig)

    (ASSETS / "A_perplexity.json").write_text(
        json.dumps({"perplexity": results, "curves": curves, "vocab_size": V,
                    "train_tokens": len(tr_ids), "val_tokens": len(va_ids)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════
# B. 문자 단위 생성의 변화 + vanishing gradient
# ════════════════════════════════════════════════════════════════════

class CharLM(nn.Module):
    def __init__(self, vocab: int, emb: int = 64, hid: int = 384, cell: str = "rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        Cell = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = Cell(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, vocab)

    def forward(self, x, state=None):
        out, state = self.rnn(self.emb(x), state)
        return self.head(out), state


@torch.no_grad()
def sample(model, itos, stoi, prime: str = "\n", length: int = 260, temp: float = 0.8) -> str:
    model.eval()
    ids = [stoi.get(c, 0) for c in prime]
    state, out = None, list(prime)
    x = torch.tensor([ids], device=DEVICE)
    for _ in range(length):
        logits, state = model(x, state)
        probs = torch.softmax(logits[0, -1] / temp, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        out.append(itos[nxt])
        x = torch.tensor([[nxt]], device=DEVICE)
    return "".join(out)


def block_b(epochs: int = 12) -> None:
    print("\n[B] 문자 단위 생성 변화 + vanishing gradient")
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = [stoi[c] for c in text]
    V = len(chars)
    print(f"  문자 {len(ids):,}개, vocab={V}")

    # ── B-1. epoch별 샘플 ─────────────────────────────────────────
    set_seed(0)
    model = CharLM(V).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    snapshots = []
    for ep in range(epochs):
        model.train()
        for xb, yb in _batchify(ids, 64, 128):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # 0, 1, 2, 4, 8 ... epoch 에서만 찍는다. 변화가 큰 초반을 촘촘히 보기 위해.
        if ep in {0, 1, 2, 4, 7, epochs - 1}:
            s = sample(model, itos, stoi)
            snapshots.append({"epoch": ep + 1, "sample": s})
            print(f"  --- epoch {ep + 1} ---\n{s[:160]}\n")

    (ASSETS / "B_samples.json").write_text(
        json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(13, 1.35 * len(snapshots) + 1.2))
    ax.axis("off")
    for i, snap in enumerate(snapshots):
        y = 1 - i / len(snapshots)
        ax.text(0, y, f"epoch {snap['epoch']}", fontsize=15, fontweight="bold", color=viz.NEW)
        body = snap["sample"].replace("\n", " ⏎ ")[:190]
        ax.text(0.1, y, body, fontsize=12, family="monospace", wrap=True)
    fig.suptitle("같은 모델이 학습되면서 뱉는 문장의 변화", fontsize=19, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "B_samples.png")
    plt.close(fig)

    # ── B-2. vanishing gradient ──────────────────────────────────
    # 마지막 시점의 손실이 '거리 d 만큼 떨어진 과거 입력'에 대해 갖는 기울기 크기.
    # 거리가 멀수록 지수적으로 0에 가까워지면 = 그만큼 과거를 기억 못 한다는 뜻.
    fig, ax = plt.subplots()
    for cell, color, label in (("rnn", viz.OLD, "RNN (tanh)"), ("lstm", viz.NEW, "LSTM")):
        set_seed(0)
        m = CharLM(V, cell=cell).to(DEVICE).train()
        T = 120
        x = torch.randint(0, V, (1, T), device=DEVICE)
        e = m.emb(x).detach().requires_grad_(True)
        out, _ = m.rnn(e)
        logits = m.head(out)
        logits[0, -1].sum().backward()          # 마지막 시점에서만 신호를 준다
        g = e.grad[0].norm(dim=-1).detach().cpu().numpy()
        dist = np.arange(T)[::-1]               # 마지막 시점으로부터의 거리
        ax.plot(dist, g, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("마지막 시점으로부터 떨어진 거리 (토큰 수)")
    ax.set_ylabel("기울기 크기 (로그 스케일)")
    ax.set_title("얼마나 먼 과거까지 신호가 닿는가")
    ax.legend()
    viz.save(fig, ASSETS / "B_vanishing.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# C. 현대 모델과 비교
# ════════════════════════════════════════════════════════════════════

def block_c(model_id: str = "gpt2") -> None:
    print(f"\n[C] {model_id} 와 비교")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, val_txt = split_text(load_corpus())
    val_txt = val_txt[:20000]

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(DEVICE)

    ids = tok(val_txt, return_tensors="pt").input_ids.to(DEVICE)
    win = min(512, model.config.max_position_embeddings)
    total_nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.size(1) - 1, win):
            chunk = ids[:, i:i + win + 1]
            if chunk.size(1) < 2:
                break
            out = model(chunk[:, :-1])
            nll = F.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                chunk[:, 1:].reshape(-1), reduction="sum",
            )
            total_nll += nll.item()
            n_tok += chunk.size(1) - 1

    # 중요: GPT-2는 BPE 토큰, 우리 RNN은 단어 단위다. 토큰 수가 다르면 perplexity를
    # 그대로 비교할 수 없다. 그래서 총 NLL을 '단어 수'로 나눠 단위를 맞춘다.
    n_words = len(tokenize(val_txt))
    result = {
        "model": model_id,
        "ppl_per_bpe_token": math.exp(total_nll / n_tok),
        "ppl_per_word": math.exp(total_nll / n_words),
        "n_bpe_tokens": n_tok,
        "n_words": n_words,
        "sample": tok.decode(
            model.generate(
                **tok("The meaning of life is", return_tensors="pt").to(DEVICE),
                max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )[0],
            skip_special_tokens=True,
        ),
    }
    (ASSETS / "C_modern.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  단어 단위로 환산한 perplexity = {result['ppl_per_word']:.1f}")
    print(f"  (BPE 토큰 기준으로는 {result['ppl_per_bpe_token']:.1f} — 단위가 달라서 직접 비교 불가)")


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "C": block_c}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="4주차 RNN 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--epochs", type=int, default=5, help="A 블록 epoch")
    ap.add_argument("--char-epochs", type=int, default=12, help="B 블록 epoch")
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    for key in args.only or list(BLOCKS):
        if key == "A":
            block_a(args.epochs)
        elif key == "B":
            block_b(args.char_epochs)
        else:
            block_c()
    print(f"\n완료 → {ASSETS}")
