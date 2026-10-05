# %% [markdown]
# # 5주차 · Bahdanau Attention (2014) 데모
# ### *Neural Machine Translation by Jointly Learning to Align and Translate*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 7분 | 문장 길이별 BLEU — 논문 Fig.2 ★ |
# | **B. 시각화** | 6분 | attention 정렬 히트맵 — 논문 Fig.3 |
# | **C. 현대 연결** | 4분 | 2026년 번역 모델의 cross-attention |

# %%
REPO_URL = "https://github.com/ovrhit/dl-session-2026-2.git"

import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/dl-session"):
    !git clone -q $REPO_URL /content/dl-session
if os.path.exists("/content/dl-session"):
    %cd /content/dl-session

# %%
FAST_MODE = True   # False 면 두 모델을 처음부터 학습 (T4 약 35분)

from common.setup import bootstrap
ctx = bootstrap("week05_attention", fast_mode=FAST_MODE)

sys.path.insert(0, "week05_attention")
import prep
from common import viz
import json

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# 2014년 기계번역의 표준 구조는 **encoder–decoder** 였다.
#
# ```
# 원문 전체  →  [인코더 RNN]  →  벡터 하나 c  →  [디코더 RNN]  →  번역문
# ```
#
# 문제는 가운데 **벡터 하나**다.
# 5단어짜리 문장도, 50단어짜리 문장도 **똑같은 크기의 벡터 하나**에 밀어넣는다.
#
# > 논문 원문의 표현: *"a fixed-length vector is a bottleneck"*
#
# 게다가 4주차에서 봤듯 RNN은 오래된 입력을 잊는다.
# 인코더가 50번째 단어를 읽을 때쯤이면 1번째 단어는 거의 사라진 상태다.
#
# > **논문의 제안**: 하나로 압축하지 말자.
# > 인코더의 **모든 시점 출력을 전부 남겨두고**, 디코더가 단어를 낼 때마다
# > "지금은 원문의 어디를 봐야 하지?"를 매번 계산하게 하자.

# %% [markdown]
# ---
# # A. bottleneck이 진짜 있나 — 논문 Fig.2 재현 ★
#
# 완전히 같은 seq2seq 두 개를 Multi30k(독→영, 29,000 문장쌍)로 학습시켰다.
# 인코더·디코더·차원·옵티마이저 전부 같고, **디코더가 문맥을 받는 방식만** 다르다.
#
# - `attention 없음` → 문장 전체를 담은 **고정 벡터 하나**를 매 시점 똑같이 받는다
# - `attention 있음` → 매 시점 원문 전체를 다시 보고 가중합을 새로 만든다

# %%
import inspect
print(inspect.getsource(prep.BahdanauAttention.forward))

# %% [markdown]
# 논문 식 (6) 그대로다. `alpha` 가 **"원문 각 단어를 얼마나 볼지"** 의 확률분포다.
# softmax를 통과하므로 다 더하면 1이 된다. 이게 B 블록에서 그릴 히트맵의 정체다.

# %%
if FAST_MODE:
    data = json.loads(ctx.asset("A_bleu_by_length.json").read_text(encoding="utf-8"))
    viz.show(ctx.asset("A_bleu_by_length.png"), width=1000)
    print(f"{'길이':>8} {'attention 없음':>16} {'attention 있음':>16}")
    for b in data["buckets"]:
        no = data["scores"]["attention 없음"][b]
        yes = data["scores"]["attention 있음"][b]
        print(f"{b:>8} {no:>16.1f} {yes:>16.1f}")
else:
    prep.block_a(epochs=12)          # T4 약 35분
    viz.show(ctx.asset("A_bleu_by_length.png"), width=1000)

# %% [markdown]
# ### 짚어줄 포인트
#
# - 짧은 문장(1–10단어)에서는 **둘이 비슷하다.** 짧으면 벡터 하나에 다 들어가니까.
# - 길어질수록 `attention 없음` 쪽이 **혼자 무너진다.**
#   `attention 있음` 은 거의 평평하게 유지된다.
# - 논문 Figure 2가 정확히 이 모양이다. 이 한 장이 논문 전체를 설득한다.
#
# 💬 **논문 읽는 법 관점**: 좋은 논문은 "우리가 더 좋다"를 평균 점수 하나로 주장하지 않는다.
# **어떤 조건에서 기존 방법이 무너지는지를 보여주고, 그 조건에서만 개선되는 걸 증명한다.**
# 평균 BLEU만 봤으면 차이가 훨씬 작아 보였을 것이다.
#
# 💬 **부원에게 던질 질문**: 그럼 벡터 크기를 100배 키우면 해결되지 않을까?
# (→ 어느 정도는. 하지만 문장 길이는 무한하고 벡터는 유한하다. 게다가 큰 벡터는
# 짧은 문장에서 낭비다. attention은 *필요한 만큼만* 보게 한다는 점이 다르다.)

# %% [markdown]
# ---
# # B. 그래서 모델은 어디를 보고 있나 — 논문 Fig.3
#
# 번역 단어를 하나 낼 때마다 `alpha` 가 원문 위에 분포를 그린다.
# 그걸 세로로 쌓으면 **정렬 행렬(alignment matrix)** 이 된다.

# %%
viz.show(ctx.asset("B_alignment.png"), width=950)

# %% [markdown]
# 대체로 **대각선**이 나온다. 앞에서부터 차례로 번역하니 당연하다.
# 재밌는 건 **대각선이 꺾이는 지점**이다.
#
# 독일어와 영어는 어순이 다른 구간이 있다 (특히 동사 위치, 관계절).
# 그런 곳에서 모델이 **원문을 건너뛰어 뒤를 먼저 보고** 오는 게 그림에 그대로 찍힌다.
#
# 여기서 중요한 것: **아무도 "어떤 단어가 어떤 단어에 대응하는지" 알려주지 않았다.**
# 번역문만 정답으로 주고 학습시켰는데, 정렬이 부수적으로 따라 나왔다.
# 논문 제목의 *"Jointly Learning to Align and Translate"* 가 이 뜻이다.

# %%
# 라이브: 문장을 직접 넣어보기 (독일어)
SENTENCE = "ein kleines mädchen spielt mit einem roten ball im park"

fig, translation = prep.alignment_figure(SENTENCE)
print("번역 결과:", " ".join(translation))
fig.show()

# %% [markdown]
# > 모델이 Multi30k(사진 설명문) 로만 학습돼서, 그 도메인 밖 문장은 잘 못 한다.
# > 29,000 문장쌍은 번역 모델치고 매우 작다. **못 하는 걸 보는 것도 데모의 일부다** —
# > 논문의 수치가 어떤 조건에서 나온 건지 감을 잡는 게 중요하다.

# %% [markdown]
# ---
# # C. 그래서 지금은? — 2026년에도 같은 그림이 나온다

# %%
viz.show(ctx.asset("C_modern_alignment.png"), width=1000)

# %% [markdown]
# 현대 번역 모델(Transformer 기반)에서 뽑은 **cross-attention** 이다.
# 2014년 그림과 **형태가 같다.** 계산 방식만 바뀌었다.
#
# | | Bahdanau (2014) | Transformer (2017~) |
# |---|---|---|
# | 점수 계산 | `vᵀ tanh(Ws + Uh)` (작은 신경망) | `qᵀk / √d` (그냥 내적) |
# | 언제 쓰나 | 디코더가 인코더를 볼 때만 | **자기 자신을 볼 때도** (self-attention) |
# | 개수 | 1개 | 여러 개 동시에 (multi-head) |
#
# 핵심 아이디어 — **"필요한 곳을 골라 본다"** — 는 그대로다.
# 다음 주 논문 제목이 *Attention Is All You Need* 인 이유가 이거다:
# 이게 이렇게 잘 되면, **RNN 부분은 아예 빼버려도 되는 것 아닌가?**

# %%
# 라이브: 아무 독일어 문장이나
fig, text = prep.modern_alignment("der hund läuft schnell über die straße")
print(text)
fig.show()

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. 고정 길이 벡터 하나는 긴 문장에서 병목이 된다 — **길이별로 쪼개 보면 보인다** (A)
# 2. 매 시점 원문 전체를 다시 보고 가중합하면 해결된다. 정렬은 **공짜로 따라온다** (B)
# 3. 이 아이디어가 지금 모든 번역 모델 안에 그대로 있다 (C)
#
# ### 다음 주 예고 — Transformer (2017)
# attention이 이렇게 잘 된다면, **RNN을 아예 빼면 어떻게 될까?**
# 순차 계산이 사라지면 문장 전체를 한 번에 GPU에 올릴 수 있다.
# 4주차에서 "RNN을 버린 진짜 이유는 속도"라고 했던 그 이야기의 결말이다.
