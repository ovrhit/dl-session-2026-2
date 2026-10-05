"""5주차 Bahdanau Attention — 사전 실행 스크립트.

    !python week05_attention/prep.py             # 전부 (T4 약 40분)
    !python week05_attention/prep.py --only B

생성물:
    assets/A_bleu_by_length.png / .json   문장 길이별 BLEU — 논문 Fig.2 재현 ★
    assets/ckpt_noattn.pt, ckpt_attn.pt   체크포인트 (세션에선 이것만 로드)
    assets/B_alignment.png                attention 정렬 히트맵 — 논문 Fig.3
    assets/C_modern_alignment.png         현대 번역 모델의 cross-attention

체크포인트가 커서 git에 부담되면 common.setup.download_assets 로 HF Hub에서 받게 바꿀 것.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
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

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
MAX_LEN = 40


# ════════════════════════════════════════════════════════════════════
# 데이터 — Multi30k (독일어 → 영어, 약 29,000 문장쌍)
# ════════════════════════════════════════════════════════════════════

def load_multi30k():
    from datasets import load_dataset

    ds = load_dataset("bentrevett/multi30k")
    def pairs(split):
        return [(r["de"], r["en"]) for r in ds[split]]
    return pairs("train"), pairs("validation"), pairs("test")


def tok(s: str) -> list[str]:
    # spacy 없이. 구두점을 떼어내는 정도면 이 데모에는 충분하다.
    out = []
    for w in s.lower().strip().split():
        head = w.rstrip(".,!?;:\"'")
        tail = w[len(head):]
        if head:
            out.append(head)
        out.extend(tail)
    return out


def build_vocab(sents, min_freq: int = 2) -> dict[str, int]:
    c = Counter(w for s in sents for w in tok(s))
    words = [w for w, n in c.items() if n >= min_freq]
    return {w: i for i, w in enumerate(SPECIALS + sorted(words))}


def encode(s: str, vocab: dict[str, int], add_bos_eos: bool = False) -> list[int]:
    ids = [vocab.get(w, UNK) for w in tok(s)][:MAX_LEN]
    return [BOS] + ids + [EOS] if add_bos_eos else ids


def make_batches(pairs, sv, tv, batch_size: int = 64, shuffle: bool = True):
    data = [(encode(s, sv), encode(t, tv, True)) for s, t in pairs]
    data = [(s, t) for s, t in data if 1 <= len(s) <= MAX_LEN and len(t) <= MAX_LEN + 2]
    # 길이가 비슷한 것끼리 묶으면 패딩이 줄어 학습이 빨라진다
    data.sort(key=lambda p: len(p[0]))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    if shuffle:
        random.shuffle(batches)
    for b in batches:
        yield _pad(b)


def _pad(batch):
    sl = max(len(s) for s, _ in batch)
    tl = max(len(t) for _, t in batch)
    src = torch.full((len(batch), sl), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), tl), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src.to(DEVICE), tgt.to(DEVICE)


# ════════════════════════════════════════════════════════════════════
# 모델 — 차이는 디코더가 문맥을 어떻게 받는가, 그것 하나
# ════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, vocab: int, emb: int = 256, hid: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(hid * 2, hid)

    def forward(self, src):
        out, h = self.rnn(self.emb(src))              # out: (B, S, 2H)
        final = torch.cat([h[0], h[1]], dim=-1)       # 양방향 마지막 상태
        return out, torch.tanh(self.bridge(final))    # (B,S,2H), (B,H)


class BahdanauAttention(nn.Module):
    """논문 식 (6): e_ij = vᵀ tanh(W s_{i-1} + U h_j)

    디코더가 단어 하나를 낼 때마다 인코더의 **모든 시점**을 다시 보고
    어디에 얼마나 집중할지 가중치를 계산한다.
    """

    def __init__(self, hid: int = 256):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)       # 디코더 상태 쪽
        self.U = nn.Linear(hid * 2, hid, bias=False)   # 인코더 출력 쪽
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_state, enc_out, mask):
        score = self.v(torch.tanh(self.W(dec_state).unsqueeze(1) + self.U(enc_out))).squeeze(-1)
        score = score.masked_fill(~mask, float("-inf"))   # 패딩은 보면 안 된다
        alpha = torch.softmax(score, dim=-1)              # (B, S) ← 이게 정렬 가중치
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab: int, emb: int = 256, hid: int = 256, attention: bool = True):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.attn = BahdanauAttention(hid) if attention else None
        self.rnn = nn.GRU(emb + hid * 2, hid, batch_first=True)
        self.head = nn.Linear(hid + hid * 2 + emb, vocab)

    def step(self, y_prev, state, enc_out, mask, fixed_context):
        e = self.emb(y_prev).unsqueeze(1)                       # (B,1,E)
        if self.attention:
            context, alpha = self.attn(state, enc_out, mask)    # 매 시점 새로 계산
        else:
            # attention 없는 쪽: 문장 전체를 담은 **고정 벡터 하나**를 매번 똑같이 쓴다.
            # 이게 논문이 말하는 bottleneck 그 자체다.
            context, alpha = fixed_context, None
        out, state = self.rnn(torch.cat([e, context.unsqueeze(1)], dim=-1), state.unsqueeze(0))
        state = state.squeeze(0)
        logit = self.head(torch.cat([state, context, e.squeeze(1)], dim=-1))
        return logit, state, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, attention: bool = True):
        super().__init__()
        self.encoder = Encoder(src_vocab)
        self.decoder = Decoder(tgt_vocab, attention=attention)
        self.attention = attention

    def _fixed_context(self, enc_out, mask):
        # attention 없는 모델의 '문장 벡터': 인코더 출력의 평균 (패딩 제외)
        m = mask.unsqueeze(-1).float()
        return (enc_out * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, src, tgt, teacher_forcing: float = 1.0):
        mask = src != PAD
        enc_out, state = self.encoder(src)
        fixed = self._fixed_context(enc_out, mask)
        logits = []
        y = tgt[:, 0]
        for t in range(1, tgt.size(1)):
            logit, state, _ = self.decoder.step(y, state, enc_out, mask, fixed)
            logits.append(logit)
            y = tgt[:, t] if random.random() < teacher_forcing else logit.argmax(-1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def translate(self, src, max_len: int = MAX_LEN + 2):
        mask = src != PAD
        enc_out, state = self.encoder(src)
        fixed = self._fixed_context(enc_out, mask)
        y = torch.full((src.size(0),), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        outs, alphas = [], []
        for _ in range(max_len):
            logit, state, alpha = self.decoder.step(y, state, enc_out, mask, fixed)
            y = logit.argmax(-1)
            outs.append(y)
            if alpha is not None:
                alphas.append(alpha)
            done |= y == EOS
            if bool(done.all()):
                break
        return torch.stack(outs, 1), (torch.stack(alphas, 1) if alphas else None)


# ════════════════════════════════════════════════════════════════════
# 학습 / 평가
# ════════════════════════════════════════════════════════════════════

def train_model(attention: bool, train_pairs, val_pairs, sv, tv, epochs: int = 12):
    set_seed(0)
    model = Seq2Seq(len(sv), len(tv), attention).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tag = "attention 있음" if attention else "attention 없음"
    for ep in range(epochs):
        model.train()
        total = n = 0
        for src, tgt in make_batches(train_pairs, sv, tv):
            opt.zero_grad(set_to_none=True)
            logits = model(src, tgt)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        print(f"  [{tag}] epoch {ep + 1:2d}/{epochs}  loss={total / n:.3f}")
    return model


def decode_ids(ids, itos) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i not in (PAD, BOS):
            out.append(itos[i])
    return " ".join(out)


BUCKETS = [(1, 10), (11, 15), (16, 20), (21, 25), (26, MAX_LEN)]


def bleu_by_length(model, pairs, sv, tv) -> dict[str, float]:
    import sacrebleu

    itos = {i: w for w, i in tv.items()}
    buckets = {f"{lo}-{hi}": ([], []) for lo, hi in BUCKETS}
    model.eval()
    for i in range(0, len(pairs), 64):
        chunk = pairs[i:i + 64]
        src, _ = _pad([(encode(s, sv), encode(t, tv, True)) for s, t in chunk])
        hyp, _ = model.translate(src)
        for (s, ref), h in zip(chunk, hyp):
            L = len(tok(s))
            for lo, hi in BUCKETS:
                if lo <= L <= hi:
                    buckets[f"{lo}-{hi}"][0].append(decode_ids(h, itos))
                    buckets[f"{lo}-{hi}"][1].append(ref.lower())
                    break
    return {
        k: (sacrebleu.corpus_bleu(h, [r]).score if len(h) >= 5 else float("nan"))
        for k, (h, r) in buckets.items()
    }


# ════════════════════════════════════════════════════════════════════
# A. 논문 Fig.2 재현
# ════════════════════════════════════════════════════════════════════

def block_a(epochs: int = 12) -> None:
    print("\n[A] 문장 길이별 BLEU — 논문 Fig.2 재현")
    train_pairs, val_pairs, test_pairs = load_multi30k()
    sv = build_vocab([s for s, _ in train_pairs])
    tv = build_vocab([t for _, t in train_pairs])
    print(f"  학습 {len(train_pairs):,}쌍  vocab de={len(sv):,} en={len(tv):,}")

    scores = {}
    for attention, key in ((False, "attention 없음"), (True, "attention 있음")):
        model = train_model(attention, train_pairs, val_pairs, sv, tv, epochs)
        scores[key] = bleu_by_length(model, test_pairs, sv, tv)
        torch.save(
            {"state": model.state_dict(), "src_vocab": sv, "tgt_vocab": tv, "attention": attention},
            ASSETS / ("ckpt_attn.pt" if attention else "ckpt_noattn.pt"),
        )
        print(f"  [{key}] BLEU = {scores[key]}")

    labels = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    fig, ax = plt.subplots()
    viz.curves(
        {k: [v[l] for l in labels] for k, v in scores.items()},
        x=list(range(len(labels))),
        xlabel="원문 문장 길이 (단어 수)", ylabel="BLEU",
        title="문장이 길어지면 무슨 일이 일어나는가 — 논문 Fig.2 재현",
        colors={"attention 없음": viz.OLD, "attention 있음": viz.NEW},
        ax=ax, annotate_last=False,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend()
    viz.save(fig, ASSETS / "A_bleu_by_length.png")
    plt.close(fig)

    (ASSETS / "A_bleu_by_length.json").write_text(
        json.dumps({"buckets": labels, "scores": scores}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════
# B. 정렬 히트맵 — 논문 Fig.3
# ════════════════════════════════════════════════════════════════════

def load_ckpt(attention: bool = True):
    ck = torch.load(ASSETS / ("ckpt_attn.pt" if attention else "ckpt_noattn.pt"),
                    map_location=DEVICE, weights_only=False)
    model = Seq2Seq(len(ck["src_vocab"]), len(ck["tgt_vocab"]), ck["attention"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["src_vocab"], ck["tgt_vocab"]


def alignment_figure(sentence: str, model=None, sv=None, tv=None):
    if model is None:
        model, sv, tv = load_ckpt(True)
    itos = {i: w for w, i in tv.items()}
    src, _ = _pad([(encode(sentence, sv), [BOS, EOS])])
    hyp, alpha = model.translate(src)
    words_src = tok(sentence)
    words_tgt = decode_ids(hyp[0], itos).split()
    A = alpha[0, :len(words_tgt), :len(words_src)].detach().cpu().numpy()
    ax = viz.heatmap(
        A, xlabels=words_src, ylabels=words_tgt,
        xaxis="원문 (독일어)", yaxis="번역 (영어)",
        title="단어를 낼 때마다 원문 어디를 봤나",
    )
    return ax.figure, words_tgt


DEMO_SENTENCES = [
    "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster",
    "zwei junge hunde spielen im gras",
]


def block_b() -> None:
    print("\n[B] attention 정렬 히트맵")
    model, sv, tv = load_ckpt(True)
    for i, s in enumerate(DEMO_SENTENCES):
        fig, out = alignment_figure(s, model, sv, tv)
        viz.save(fig, ASSETS / f"B_alignment_{i}.png")
        plt.close(fig)
        print(f"  {s}\n  → {' '.join(out)}")
    # 노트북이 기본으로 띄울 대표 그림
    import shutil
    shutil.copy(ASSETS / "B_alignment_0.png", ASSETS / "B_alignment.png")


# ════════════════════════════════════════════════════════════════════
# C. 현대 번역 모델의 cross-attention
# ════════════════════════════════════════════════════════════════════

def modern_alignment(sentence: str, model_id: str = "Helsinki-NLP/opus-mt-de-en"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval().to(DEVICE)
    ins = tk(sentence, return_tensors="pt").to(DEVICE)
    gen = m.generate(**ins, max_new_tokens=60, output_attentions=True,
                     return_dict_in_generate=True, num_beams=1)

    # cross_attentions: (생성 스텝, 층, B, head, 1, src_len) → 마지막 층의 head 평균
    steps = [step[-1][0].mean(0)[0].detach().cpu().numpy() for step in gen.cross_attentions]
    A = np.stack(steps, axis=0)
    src_tokens = tk.convert_ids_to_tokens(ins.input_ids[0])
    tgt_tokens = tk.convert_ids_to_tokens(gen.sequences[0][1:])
    n = min(len(tgt_tokens), A.shape[0])
    ax = viz.heatmap(
        A[:n, :len(src_tokens)],
        xlabels=[t.replace("▁", "") for t in src_tokens],
        ylabels=[t.replace("▁", "") for t in tgt_tokens[:n]],
        xaxis="원문", yaxis="번역",
        title=f"{model_id} 의 cross-attention (2014년과 같은 그림)",
        cmap="Purples",
    )
    return ax.figure, tk.decode(gen.sequences[0], skip_special_tokens=True)


def block_c() -> None:
    print("\n[C] 현대 번역 모델")
    fig, text = modern_alignment(DEMO_SENTENCES[0])
    viz.save(fig, ASSETS / "C_modern_alignment.png")
    plt.close(fig)
    print(f"  → {text}")


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "C": block_c}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5주차 Attention 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    print(f"device = {DEVICE}")
    for key in args.only or list(BLOCKS):
        block_a(args.epochs) if key == "A" else BLOCKS[key]()
    print(f"\n완료 → {ASSETS}")
