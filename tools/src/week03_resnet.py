# %% [markdown]
# # 3주차 · ResNet (2015) 데모
# ### *Deep Residual Learning for Image Recognition*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 8분 | plain-20/56 vs ResNet-20/56 — 논문 Fig.6 ★ |
# | **B. 시각화** | 4분 | skip 유무에 따른 층별 gradient norm |
# | **C. 현대 연결** | 5분 | residual은 지금도 살아있다 + Grad-CAM |
#
# > **세션 시작 10분 전에 셋업 셀 두 개 실행.**

# %%
REPO_URL = "https://github.com/ovrhit/dl-session-2026-2.git"

import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/dl-session"):
    !git clone -q $REPO_URL /content/dl-session
if os.path.exists("/content/dl-session"):
    %cd /content/dl-session

# %%
FAST_MODE = True   # False 로 바꾸면 네 모델을 전부 처음부터 학습한다 (T4 약 45분)

from common.setup import bootstrap
ctx = bootstrap("week03_resnet", fast_mode=FAST_MODE)

sys.path.insert(0, "week03_resnet")
import prep
from common import viz

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# AlexNet은 8층, VGG는 19층이었다. 자연스러운 다음 질문은 **"더 깊게 하면 되나?"** 였다.
#
# 해봤더니 **더 나빠졌다.** 그런데 이게 과적합이 아니었다.
# 과적합이면 학습 오차는 낮고 테스트 오차만 높아야 하는데,
# **학습 오차 자체가 더 높았다.** 논문은 이걸 *degradation problem* 이라고 부른다.
#
# 이게 왜 이상한지 생각해보자.
# 56층 망은 20층 망을 그대로 흉내낼 수 있다 — 뒤쪽 36층이 전부 항등함수(identity)면 된다.
# **원리적으로는 최소한 같아야 하는데**, 실제로는 그 해를 못 찾는다.
# 최적화 문제지 표현력 문제가 아니라는 뜻이다.
#
# > **해법**: 항등함수를 배우기 어렵다면, 그냥 **기본값으로 깔아주자.**
# > `H(x)` 를 직접 배우게 하는 대신 `F(x) = H(x) − x` 를 배우게 하고 `H(x) = F(x) + x` 로 쓴다.
# > 아무것도 안 배우면(F=0) 자동으로 항등함수가 된다.

# %% [markdown]
# ---
# # A. degradation problem 직접 보기 — 논문 Fig.6 재현 ★
#
# 네 모델을 **완전히 같은 조건**으로 학습시켰다.
# 층 수·채널·BatchNorm·초기화·옵티마이저 전부 동일하고,
# 차이는 `Block.forward` 의 마지막 한 줄뿐이다.

# %%
import inspect
print(inspect.getsource(prep.Block.forward))

# %% [markdown]
# `out = out + self.short(x)` — 이 한 줄이 오늘 논문 전부다.

# %%
import json

if FAST_MODE:
    data = json.loads(ctx.asset("A_degradation.json").read_text(encoding="utf-8"))
    viz.show(ctx.asset("A_degradation.png"), width=1100)
    print("최종 학습 오차 (낮을수록 좋음):")
    for k, v in data["final"].items():
        print(f"  {k:10s}  train={v['train_err']:.3f}   test={v['test_err']:.3f}")
else:
    prep.block_a(epochs=20)          # T4 약 45분
    viz.show(ctx.asset("A_degradation.png"), width=1100)

# %% [markdown]
# ### 왼쪽 그래프(학습 오차)만 보면 된다
#
# - `plain-56` 이 `plain-20` **보다 위에 있다.** 깊게 만들었는데 학습조차 더 못 한다.
# - `ResNet-56` 은 `ResNet-20` **보다 아래에 있다.** 깊이가 제대로 이득이 됐다.
#
# 다시 강조: 이건 **학습** 오차다. 과적합 이야기가 아니다.
# 데이터를 외우는 것조차 더 못 하고 있다.
#
# 💬 **부원에게 던질 질문**: 56층 망은 20층 망을 흉내낼 수 있다고 했는데, 왜 못 찾을까?
# (→ 항등함수를 여러 층의 곱으로 정확히 표현하는 건 경사하강법에게 의외로 어려운 목표다.
# 그래서 "배우게" 하지 말고 "깔아주자"가 답이 된다.)

# %%
# 라이브: 초반 몇 epoch만 돌려서 갈라지는 걸 직접 본다 (T4 약 3분)
RUN_LIVE = False     # 시간이 남으면 True

if RUN_LIVE:
    live = {n: prep.train(56, r, epochs=3) for n, r in [("plain-56", False), ("ResNet-56", True)]}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    viz.curves({k: v["train_err"] for k, v in live.items()},
               ylabel="학습 오차", title="초반 3 epoch만 봐도 갈라진다",
               colors=prep.COLORS, ax=ax)
    plt.show()

# %% [markdown]
# ---
# # B. 왜 깊으면 학습이 안 되나 — 층별 gradient norm
#
# 역전파를 한 번 돌리고, **각 conv 층의 가중치가 받은 기울기 크기**를 재본다.
# 입력 쪽(왼쪽)으로 갈수록 기울기가 살아있는지가 관건이다.

# %%
viz.show(ctx.asset("B_gradient_norm.png"), width=1050)

# %% [markdown]
# `plain-56` 은 입력 쪽으로 갈수록 기울기가 눈에 띄게 줄어든다.
# 기울기가 작다는 건 그 층이 **거의 학습되지 않는다**는 뜻이다.
#
# `ResNet-56` 은 덜 줄어든다. 이유는 덧셈의 미분이다.
# `y = F(x) + x` 를 `x` 로 미분하면 `F'(x) + 1` 이 되는데,
# **이 `+1` 덕분에 기울기가 층을 건너뛰어 곧장 아래로 흐른다.**
# skip connection을 "기울기 고속도로"라고 부르는 이유다.
#
# 💬 이건 2주차 ReLU 이야기의 연장이다. ReLU는 *한 층 안에서* 기울기가 죽는 걸 막았고,
# residual은 *층을 넘어가면서* 죽는 걸 막는다.

# %% [markdown]
# ---
# # C. 그래서 지금은? — residual은 아직도 모든 곳에 있다
#
# 2015년 아이디어가 2026년 LLM 안에 글자 그대로 들어있다.
# ResNet의 블록과 현대 LLM의 블록을 나란히 보자.

# %%
import re, torch, torchvision

src_cnn = inspect.getsource(torchvision.models.resnet.BasicBlock.forward)
print("── torchvision ResNet BasicBlock (2015) ──")
print(src_cnn)

# %%
from transformers.models.llama import modeling_llama

src_llm = inspect.getsource(modeling_llama.LlamaDecoderLayer.forward)
# 긴 주석/타입힌트를 걷어내고 residual 관련 줄만 남긴다
keep = [ln for ln in src_llm.splitlines() if "residual" in ln or "hidden_states =" in ln]
print("── HF LlamaDecoderLayer (2026년에도 쓰는 LLM 블록) ──")
print("\n".join(keep))

# %% [markdown]
# 둘 다 **`출력 = 입력 + 무언가`** 다.
# 이미지 분류기든 언어 모델이든, 깊게 쌓으려면 이 구조가 필요하다.
#
# 6주차 Transformer를 읽을 때 블록 그림에서 이 화살표를 찾아보면 된다. 반드시 있다.
#
# > 여담: residual은 "깊게 쌓기"의 전제조건이라서, 이게 없었으면
# > 100층짜리 GPT도, 확산모델의 U-Net도 없었다. 오늘 논문은 5주차 이후 전부의 바닥에 깔려 있다.

# %%
# Grad-CAM — 판단 근거 들여다보기
viz.show(ctx.asset("C_gradcam.png"), width=1100)

# %% [markdown]
# 빨간 부분이 "이 판단에 많이 기여한 영역"이다.
# 개라고 판단했으면 개 얼굴이 빨개야 정상인데, 가끔 **엉뚱한 데를 보고 맞히는** 경우가 있다
# (배경의 잔디만 보고 "양"이라고 하는 식). 그럴 때가 오히려 재밌다.

# %%
# 라이브: 부원 사진으로 Grad-CAM
if "google.colab" in sys.modules:
    from google.colab import files
    up = files.upload()
    dest = ctx.week_dir.parent / "week02_alexnet" / "assets" / "samples"
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in up.items():
        (dest / name).write_bytes(blob)
    prep.block_c()
    viz.show(ctx.asset("C_gradcam.png"), width=1100)

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. 깊게 쌓았더니 **학습 오차**가 더 높아졌다 — 표현력이 아니라 최적화 문제 (A)
# 2. 원인은 기울기가 입력 쪽까지 못 간다는 것. `+x` 가 고속도로를 놔준다 (B)
# 3. 이 한 줄이 지금 LLM 블록에도 그대로 들어있다 (C)
#
# ### 다음 주 예고 — RNN 언어모델 (2010)
# 지금까지는 전부 **이미지**였다. 이미지는 한 번에 다 보면 되지만
# 문장은 순서가 있다. 앞 단어를 기억하면서 읽어야 한다.
# 그 "기억"을 어떻게 만들 것인가, 그리고 그 기억이 왜 금방 사라지는가.
