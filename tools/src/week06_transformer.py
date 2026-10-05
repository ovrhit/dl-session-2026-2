# %% [markdown]
# # 6주차 · Transformer (2017) 데모
# ### *Attention Is All You Need*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 7분 | self-attention 직접 구현 + RNN 대비 학습 속도 |
# | **B. 시각화** | 6분 | head마다 다른 걸 본다 + 위치 정보를 빼면? ★ |
# | **C. 현대 연결** | 4분 | 2017 블록 vs 2026 LLM 블록, 뭐가 바뀌었나 |

# %%
REPO_URL = "https://github.com/ovrhit/dl-session-2026-2.git"

import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/dl-session"):
    !git clone -q $REPO_URL /content/dl-session
if os.path.exists("/content/dl-session"):
    %cd /content/dl-session

# %%
FAST_MODE = True

from common.setup import bootstrap
ctx = bootstrap("week06_transformer", fast_mode=FAST_MODE)

sys.path.insert(0, "week06_transformer")
import prep
from common import viz
import json, torch

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# 5주차에서 attention이 RNN의 병목을 풀어주는 걸 봤다.
# 그런데 그때 attention은 **RNN 위에 얹은 보조 장치**였다.
#
# > 이 논문의 질문: **RNN을 빼고 attention만 남기면?**
#
# 왜 빼고 싶었나 — 4주차 결론 그대로다. RNN은 t번째를 계산하려면 t−1이 끝나야 한다.
# **순차적이라 GPU를 못 쓴다.**
# attention은 모든 위치가 모든 위치를 동시에 본다. **전부 한 번에 행렬곱 한 번으로 끝난다.**
#
# 대신 잃는 게 하나 있다: **순서 정보**. 그건 따로 넣어줘야 한다 (→ B 블록).

# %% [markdown]
# ---
# # A-1. self-attention은 사실 한 줄이다
#
# 논문 식 (1):
#
# $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

# %%
import inspect
print(inspect.getsource(prep.scaled_dot_product_attention))

# %% [markdown]
# 읽는 순서는 이렇다.
#
# - **Q (query)** "나는 지금 뭘 찾고 있나"
# - **K (key)** "나는 이런 정보를 갖고 있다"
# - **V (value)** "실제로 전달할 내용"
#
# `QKᵀ` 는 모든 위치 쌍의 **궁합 점수**다. softmax를 씌워 확률로 만들고,
# 그 비율대로 V를 섞는다. 5주차 `alpha` 와 정확히 같은 역할인데,
# 작은 신경망(`vᵀtanh(...)`) 대신 **그냥 내적**을 쓴다 — 그래서 훨씬 빠르다.
#
# ### `√d_k` 로 왜 나누나 (논문 각주 4)
# 차원이 크면 내적값의 분산이 커진다 → softmax가 한 곳에 극단적으로 몰린다
# → 기울기가 거의 0이 된다. 나눠주면 분산이 1 근처로 유지된다.
# **논문의 각주 하나가 실제로는 학습 성공/실패를 가르는 경우가 많다.** 각주를 읽자.

# %%
# 직접 돌려보기 — 5개 토큰, 1개 head
torch.manual_seed(0)
q = k = v = torch.randn(1, 1, 5, 8)
out, attn = prep.scaled_dot_product_attention(q, k, v)
print("attention 가중치 (행마다 합이 1):")
print(attn[0, 0].round(decimals=2))

# %%
# causal mask 를 씌우면 — 미래를 못 본다 (오른쪽 위 삼각형이 0)
mask = torch.tril(torch.ones(5, 5))
_, attn_causal = prep.scaled_dot_product_attention(q, k, v, mask)
print(attn_causal[0, 0].round(decimals=2))

# %% [markdown]
# 언어모델은 "다음 단어 맞히기"로 학습한다.
# 마스크가 없으면 **정답을 그냥 보고 베낀다.** 손실은 0이 되지만 아무것도 못 배운다.
# 이런 종류의 유출(leakage)은 실제 코드에서 정말 자주 나는 버그다.

# %% [markdown]
# # A-2. 그래서 얼마나 빨라지나 — 논문의 '병렬화' 주장 재현
#
# 4주차와 **같은 코퍼스**에, 파라미터 수를 비슷하게 맞춘 LSTM과 Transformer를
# 똑같이 600 step 학습시키고 **벽시계 시간**을 쟀다.

# %%
if FAST_MODE:
    d = json.loads(ctx.asset("A_speed.json").read_text(encoding="utf-8"))
    viz.show(ctx.asset("A_speed.png"), width=1100)
    for name, r in d["results"].items():
        print(f"  {name:12s} {r['seconds']:6.1f}초   final_loss={r['final_loss']:.3f}"
              f"   params={r['params'] / 1e6:.1f}M")
else:
    prep.block_a(steps=600)          # T4 약 8분
    viz.show(ctx.asset("A_speed.png"), width=1100)

# %% [markdown]
# **같은 시간에 더 많이 배운다.** 이게 논문이 실제로 판 것이다.
#
# 한 가지 정직하게 짚을 것: 여기 쓴 시퀀스 길이는 128이다.
# 길이를 늘리면 격차가 더 벌어진다 (LSTM은 길이에 비례해 느려지고,
# Transformer는 행렬 크기만 커진다).
#
# 반대로 attention은 길이의 **제곱**에 비례하는 계산량이 든다.
# 길이가 수만이 되면 오히려 부담이 된다 — 이게 지금 롱컨텍스트 연구의 출발점이다.
#
# 💬 **부원에게 던질 질문**: 그럼 왜 2017년 전에는 이런 걸 안 했을까?
# (→ 아이디어보다 하드웨어가 먼저였다. "행렬곱 한 방"이 이득이 되려면 GPU가 충분히 커야 한다.)

# %% [markdown]
# ---
# # B-1. head마다 다른 걸 본다 ★
#
# multi-head는 attention을 여러 개 **동시에** 돌린다.
# 논문은 "서로 다른 표현 공간에서 다른 관계를 보게 하려고"라고만 썼는데,
# 실제로 학습된 모델을 열어보면 진짜로 역할이 갈린다.

# %%
# BertViz — 클릭해서 층/head를 바꿔가며 볼 수 있다
from bertviz import head_view
from transformers import AutoModel, AutoTokenizer

TEXT = "The cat sat on the mat because it was tired."

tk = AutoTokenizer.from_pretrained("gpt2")
m = AutoModel.from_pretrained("gpt2", attn_implementation="eager", output_attentions=True).eval()
ins = tk(TEXT, return_tensors="pt")
with torch.no_grad():
    attn = m(**ins).attentions
head_view(attn, tk.convert_ids_to_tokens(ins.input_ids[0]))

# %% [markdown]
# **이렇게 보면 된다**: 왼쪽 토큰에서 오른쪽 토큰으로 가는 선의 진하기가 attention 세기다.
# 위쪽 색깔 막대에서 head를 켜고 끌 수 있다.
#
# 찾아볼 만한 패턴:
# - **직전 토큰만 보는 head** — 거의 모든 모델에 있다
# - **구두점/문장 시작 토큰에 몰리는 head** — 할 일 없을 때의 "주차장" 역할
# - **`it` → `cat` 처럼 지시어를 연결하는 head** — 문장 의미를 따라간다
#
# 마지막 게 흥미롭다. 아무도 대명사 해석을 가르치지 않았는데
# 다음 토큰 예측만으로 그런 head가 생겼다.

# %%
# BertViz가 안 뜨면 (Colab 위젯 문제) 이 정적 그림으로 대체
viz.show(ctx.asset("B_head_patterns.png"), width=1200)

# %% [markdown]
# # B-2. 그런데 self-attention은 순서를 모른다
#
# `softmax(QKᵀ/√d)V` 를 다시 보자. **토큰 순서를 뒤섞어도 결과가 같다.**
# 위치라는 개념이 식 어디에도 없다. 집합 연산이다.
#
# 그래서 논문은 위치마다 다른 "지문"을 임베딩에 **더해준다** (식 2, 3).

# %%
viz.show(ctx.asset("B_pos_encoding.png"), width=1100)

# %% [markdown]
# 왼쪽 그림: 세로가 위치, 가로가 차원. **위치마다 줄무늬 패턴이 다르다.**
# 낮은 차원은 빠르게 진동하고(가까운 위치 구분), 높은 차원은 천천히 진동한다(먼 위치 구분).
# 여러 주파수를 겹쳐서 위치를 부호화하는 것 — 아날로그 시계의 초침/분침/시침과 같은 발상이다.
#
# 굳이 sin/cos을 쓴 이유는 **상대 위치**가 선형변환으로 표현되기 때문이다.
# `PE(pos+k)` 를 `PE(pos)` 의 선형결합으로 쓸 수 있어서, 모델이 "3칸 앞"을 배우기 쉽다.

# %%
# 진짜로 필요한가 — 빼고 학습시켜봤다
viz.show(ctx.asset("B_no_pe.png"), width=1000)

# %% [markdown]
# 위치 정보를 빼면 손실이 눈에 띄게 높은 데서 멈춘다.
# 모델이 "어떤 글자들이 있었나"만 알고 **"어떤 순서였나"를 모르기 때문**이다.
# 언어에서 순서는 거의 전부다. (`개가 사람을 물었다` vs `사람이 개를 물었다`)

# %% [markdown]
# ---
# # C. 그래서 지금은? — 9년 동안 뭐가 바뀌었나

# %%
viz.table(prep.MODERN_DIFF)

# %%
# 실제 코드로 확인 — 현대 LLM 블록의 forward
from transformers.models.llama import modeling_llama
src = inspect.getsource(modeling_llama.LlamaDecoderLayer.forward)
print("\n".join(ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#"))[:1400])

# %% [markdown]
# 바뀐 건 전부 **부품 교체**다. 구조 자체는 그대로다.
#
# ```
# x = x + attention(norm(x))
# x = x + ffn(norm(x))
# ```
#
# 이 두 줄이 2017년 논문에도, 2026년 최신 LLM에도 있다.
# 딥러닝에서 9년은 아주 긴 시간인데도 살아남았다는 게 이 논문의 위상이다.
#
# 💬 **논문 읽는 법 관점**: 인용수가 많은 논문이 반드시 어려운 게 아니다.
# 이 논문의 핵심은 식 (1) 하나고, 나머지는 "그걸로 충분하다"는 걸 보이는 실험이다.
# **논문을 읽을 때 '이 논문이 실제로 새로 제안한 건 몇 줄인가'를 찾아보면** 구조가 잡힌다.

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. attention은 내적 + softmax + 가중합. RNN 없이 **전부 한 번에** 계산된다 (A-1)
# 2. 그래서 같은 시간에 더 많이 배운다 — 논문이 판 건 정확도가 아니라 **확장성**이다 (A-2)
# 3. head마다 다른 관계를 보고, 순서는 따로 넣어줘야 한다 (B)
# 4. 부품은 바뀌었지만 `x + attn(norm(x))` 구조는 9년째 그대로다 (C)
#
# ### 다음 주 예고 — GPT-3 (2020)
# 오늘 구조를 **그대로 두고 크기만 키우면** 어떻게 될까.
# 1750억 개 파라미터에서, 아무도 예상 못 한 일이 벌어진다:
# **모델을 고치지 않고 예시 몇 개만 보여줘도 새 문제를 푼다.**
