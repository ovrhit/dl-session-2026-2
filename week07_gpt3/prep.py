"""7주차 GPT-3 — 사전 실행 스크립트.

    !python week07_gpt3/prep.py                  # 전부 (T4 약 30분)
    !python week07_gpt3/prep.py --only A --n 30

왜 Pythia 시리즈인가:
    GPT-3 논문의 주장은 "**크기만** 키웠더니 능력이 생겼다"이다.
    그런데 대부분의 모델 시리즈는 크기별로 학습 데이터·레시피가 다르다.
    그러면 "크기 때문인지 데이터 때문인지" 구분이 안 된다.
    Pythia는 **같은 데이터를 같은 순서로** 먹인 크기 사다리라서, 이 재현에 정직하다.

생성물:
    assets/A_scaling.png / .json    모델 크기 × shot 수 격자 — 논문 Fig.1.2 재현 ★
    assets/B_prompts.json           실제로 넣은 프롬프트 원문
    assets/B_tokenizer.json         토크나이저 데모 결과
    assets/C_instruct.json          instruct 모델 0-shot 결과
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import viz  # noqa: E402
from common.setup import set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# fp16 기준 VRAM: 160m≈0.3GB / 410m≈0.8GB / 1.4b≈2.8GB / 2.8b≈5.6GB → T4(16GB) 안에 들어간다.
MODELS = [
    ("pythia-160m", "EleutherAI/pythia-160m", 0.16),
    ("pythia-410m", "EleutherAI/pythia-410m", 0.41),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", 1.4),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b", 2.8),
]
SHOTS = [0, 1, 2, 4, 8]


# ════════════════════════════════════════════════════════════════════
# 태스크 — 전부 GPT-3 논문에 실제로 나오는 것들
# ════════════════════════════════════════════════════════════════════

WORDS = [
    "banana", "guitar", "window", "pencil", "rocket", "yellow", "silver", "forest",
    "castle", "danger", "mirror", "planet", "summer", "wisdom", "bridge", "candle",
    "garden", "hunter", "island", "jungle", "ladder", "market", "narrow", "orange",
]


def task_addition(rng: random.Random) -> tuple[str, str]:
    """논문 3.9.1 Arithmetic — 2자리 덧셈."""
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"{a} + {b} =", str(a + b)


def task_reverse(rng: random.Random) -> tuple[str, str]:
    """논문 3.9.2 Word Scrambling — 철자 뒤집기."""
    w = rng.choice(WORDS)
    return f"{w[::-1]} =", w


def task_typo(rng: random.Random) -> tuple[str, str]:
    """논문 3.9.2 의 Random Insertion 변형 — 끼워넣은 문자를 지우고 원래 단어 복원."""
    w = rng.choice(WORDS)
    i = rng.randint(1, len(w) - 1)
    noisy = w[:i] + rng.choice("!@#$%") + w[i:]
    return f"{noisy} =", w


TASKS = {
    "2자리 덧셈": (task_addition, "Q: 12 + 34 =\nA: 46"),
    "철자 뒤집기": (task_reverse, "Q: rorrim =\nA: mirror"),
    "오타 고치기": (task_typo, "Q: mir#ror =\nA: mirror"),
}


def build_prompt(task_fn, rng: random.Random, k: int) -> tuple[str, str]:
    """k개의 예시 + 풀어야 할 문제 하나. 이게 in-context learning 의 전부다.

    가중치는 전혀 건드리지 않는다. **그냥 입력 텍스트가 길어질 뿐이다.**
    """
    lines = []
    for _ in range(k):
        q, a = task_fn(rng)
        lines.append(f"Q: {q}\nA: {a}")
    q, a = task_fn(rng)
    lines.append(f"Q: {q}\nA:")
    return "\n\n".join(lines), a


# ════════════════════════════════════════════════════════════════════
# 평가
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, tok, task_fn, k: int, n: int, seed: int = 0, batch: int = 8) -> float:
    rng = random.Random(seed)
    prompts, golds = zip(*(build_prompt(task_fn, rng, k) for _ in range(n)))
    correct = 0
    for i in range(0, n, batch):
        chunk, gold = list(prompts[i:i + batch]), golds[i:i + batch]
        ins = tok(chunk, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(
            **ins, max_new_tokens=8, do_sample=False,      # greedy — 재현 가능하게
            pad_token_id=tok.pad_token_id,
        )
        for j, g in enumerate(gold):
            gen = tok.decode(out[j][ins.input_ids.shape[1]:], skip_special_tokens=True)
            # 첫 줄의 첫 토큰만 본다. 모델은 멈추는 법을 배운 적이 없어서 계속 지껄인다.
            pred = gen.strip().split("\n")[0].strip().split(" ")[0].strip()
            correct += int(pred == g)
    return correct / n


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"     # decoder-only 배치 생성은 반드시 왼쪽 패딩
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).eval().to(DEVICE)
    return model, tok


# ════════════════════════════════════════════════════════════════════
# A. 논문 Fig.1.2 재현
# ════════════════════════════════════════════════════════════════════

def block_a(n: int = 40) -> None:
    print("\n[A] 모델 크기 × shot 수 격자 — 논문 Fig.1.2 재현")
    results: dict[str, dict[str, dict[int, float]]] = {}

    for label, model_id, _ in MODELS:
        print(f"\n  ── {label} 로드 중")
        model, tok = load_model(model_id)
        results[label] = {}
        for task_name, (fn, _) in TASKS.items():
            results[label][task_name] = {}
            for k in SHOTS:
                acc = evaluate(model, tok, fn, k, n)
                results[label][task_name][k] = acc
                print(f"     {task_name:10s} k={k}  acc={acc:.2f}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    sizes = {label: b for label, _, b in MODELS}
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 5.6), sharey=True)
    for ax, task_name in zip(axes, TASKS):
        for i, k in enumerate(SHOTS):
            ax.plot(
                [sizes[l] for l, _, _ in MODELS],
                [results[l][task_name][k] for l, _, _ in MODELS],
                marker="o", label=f"{k}-shot", color=viz.PALETTE[i % len(viz.PALETTE)],
            )
        ax.set_xscale("log")
        ax.set_xticks([b for _, _, b in MODELS])
        ax.set_xticklabels([f"{b}B" for _, _, b in MODELS])
        ax.set_xlabel("파라미터 수 (로그 스케일)")
        ax.set_title(task_name)
    axes[0].set_ylabel("정확도")
    axes[-1].legend(title="예시 개수")
    fig.suptitle("모델을 키워야만 '예시를 보고 배우는' 능력이 생긴다",
                 fontsize=20, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ASSETS / "A_scaling.png")
    plt.close(fig)

    (ASSETS / "A_scaling.json").write_text(
        json.dumps({"n_eval": n, "shots": SHOTS,
                    "sizes_b": {l: b for l, _, b in MODELS}, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════
# B. 프롬프트 원문 + 토크나이저
# ════════════════════════════════════════════════════════════════════

TOKENIZER_PROBES = [
    ("strawberry", "영어 단어 — r이 몇 개인지 세어보라고 하면 왜 틀리나"),
    ("1234567890", "숫자 — 자릿수대로 안 쪼개진다"),
    ("딸기가 좋아요", "한국어 — 영어보다 토큰을 훨씬 많이 먹는다"),
    ("     ", "공백 — 들여쓰기가 코드 모델에서 문제되는 이유"),
]


def block_b() -> None:
    print("\n[B] 프롬프트 원문 + 토크나이저")
    rng = random.Random(0)
    prompts = {}
    for task_name, (fn, _) in TASKS.items():
        prompts[task_name] = {str(k): build_prompt(fn, random.Random(0), k)[0] for k in (0, 1, 4)}
    (ASSETS / "B_prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  프롬프트 예시 저장 ({len(prompts)} 태스크)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    probes = []
    for text, note in TOKENIZER_PROBES:
        ids = tok(text).input_ids
        probes.append({
            "text": text, "note": note, "n_tokens": len(ids),
            "tokens": [tok.decode([i]) for i in ids],
        })
        print(f"  {text!r:22s} → {len(ids)}토큰  {[tok.decode([i]) for i in ids]}")
    (ASSETS / "B_tokenizer.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ════════════════════════════════════════════════════════════════════
# C. instruct 모델 — few-shot 없이도 되나
# ════════════════════════════════════════════════════════════════════

INSTRUCT_ID = "Qwen/Qwen2.5-1.5B-Instruct"

INSTRUCTIONS = {
    "2자리 덧셈": "Compute the sum. Reply with only the number.",
    "철자 뒤집기": "The input is a word spelled backwards. Reply with only the original word.",
    "오타 고치기": "One junk character was inserted into a word. Reply with only the corrected word.",
}


def block_c(n: int = 40, model_id: str = INSTRUCT_ID) -> None:
    print(f"\n[C] {model_id} — 0-shot")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).eval().to(DEVICE)

    out = {}
    for task_name, (fn, _) in TASKS.items():
        rng = random.Random(0)
        correct = 0
        for _ in range(n):
            q, gold = fn(rng)
            chat = [
                {"role": "system", "content": INSTRUCTIONS[task_name]},
                {"role": "user", "content": q.rstrip(" =")},
            ]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            ins = tok(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                gen = model.generate(**ins, max_new_tokens=12, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            pred = tok.decode(gen[0][ins.input_ids.shape[1]:], skip_special_tokens=True)
            correct += int(pred.strip().split("\n")[0].strip().strip(".") == gold)
        out[task_name] = correct / n
        print(f"  {task_name:10s} 0-shot acc={out[task_name]:.2f}")

    (ASSETS / "C_instruct.json").write_text(
        json.dumps({"model": model_id, "n_eval": n, "accuracy": out},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════

BLOCKS = {"A": block_a, "B": block_b, "C": block_c}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="7주차 GPT-3 데모 에셋 생성")
    ap.add_argument("--only", choices=list(BLOCKS), nargs="*")
    ap.add_argument("--n", type=int, default=40, help="태스크당 평가 문제 수")
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    set_seed(0)
    print(f"device = {DEVICE}")
    if DEVICE == "cpu":
        print("[경고] GPU 없이 A 블록은 현실적으로 불가능하다.")
    for key in args.only or list(BLOCKS):
        block_b() if key == "B" else BLOCKS[key](args.n)
    print(f"\n완료 → {ASSETS}")
