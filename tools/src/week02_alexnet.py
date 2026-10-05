# %% [markdown]
# # 2주차 · AlexNet (2012) 데모
# ### *ImageNet Classification with Deep Convolutional Neural Networks*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 7분 | ReLU vs tanh 수렴 속도 — 논문 Fig.1 |
# | **B. 시각화** | 5분 | 1층 conv 필터 그리드 — 논문 Fig.3 ★ |
# | **C. 현대 연결** | 5분 | AlexNet vs ConvNeXt vs CLIP, 같은 사진으로 |
#
# > **세션 시작 10분 전에 아래 셀 두 개를 미리 실행해둘 것.**
# > 모델 가중치 다운로드를 세션 중에 시작하면 몇 분을 통째로 날린다.

# %%
# ── 셋업 (세션 시작 10분 전 실행) ──────────────────────────────────
# Colab에서 처음 열었다면 repo를 받아온다. 로컬이면 이 셀은 그냥 지나간다.
REPO_URL = "https://github.com/ovrhit/dl-session-2026-2.git"

import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/dl-session"):
    !git clone -q $REPO_URL /content/dl-session
if os.path.exists("/content/dl-session"):
    %cd /content/dl-session

# %%
FAST_MODE = True   # True  : 미리 만들어둔 에셋 로드 — 세션 진행용 (즉시)
                   # False : 처음부터 전부 재현 — 15분 이상 (사후 학습용)

from common.setup import bootstrap
ctx = bootstrap("week02_alexnet", fast_mode=FAST_MODE)

sys.path.insert(0, "week02_alexnet")
import prep
from common import viz

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# > 깊은 CNN을 **ReLU + Dropout + 데이터 증강 + GPU 2장**으로 훈련시켰더니
# > ImageNet top-5 오류가 26.2% → **15.3%** 로 떨어졌다.
#
# 2위와의 격차가 10%p 넘게 벌어진 대회였다. 이 논문 이후 컴퓨터 비전 연구는
# 사실상 전부 딥러닝으로 넘어갔다.
#
# **오늘 확인할 것 두 가지**
# 1. 논문이 "ReLU가 빠르다"고 한 게 진짜인가 → **A**
# 2. 네트워크가 대체 뭘 배운 건가 → **B**

# %% [markdown]
# ---
# # A. ReLU vs tanh — 논문 Fig.1 재현
#
# 논문 Figure 1은 **같은 4층 CNN에서 활성함수만 바꿨을 때** 학습 오차 25%에
# 도달하는 속도가 6배 차이난다는 그림이다.
#
# 여기서는 CIFAR-10 + 5층 CNN으로 축소해서 같은 실험을 돌렸다.
# 비교가 공정하려면 **활성함수 말고는 전부 같아야 한다** — 같은 seed, 같은 초기값,
# 같은 learning rate, 같은 배치 순서. (`week02_alexnet/prep.py` 의 `_train_one` 참고)

# %%
import json

if FAST_MODE:
    data = json.loads(ctx.asset("A_relu_vs_tanh.json").read_text(encoding="utf-8"))
    viz.show(ctx.asset("A_relu_vs_tanh.png"))
    print(f"학습 오차 25% 도달:  ReLU {data['reach_epoch']['relu']} epoch"
          f"  /  tanh {data['reach_epoch']['tanh']} epoch"
          f"  →  {data['speedup']:.1f}배 차이")
else:
    prep.block_a(epochs=25)          # T4에서 약 12분
    viz.show(ctx.asset("A_relu_vs_tanh.png"))

# %% [markdown]
# ### 짚어줄 포인트
#
# - **왜 ReLU가 빠른가**: tanh는 입력이 조금만 커져도 기울기가 0에 가까워진다(포화).
#   기울기가 0이면 그 층은 학습이 멈춘다. ReLU는 양수 구간에서 기울기가 항상 1이라
#   깊어져도 신호가 죽지 않는다.
# - 이건 "정확도가 높아졌다"는 얘기가 **아니다**. 같은 곳에 **빨리** 도달한다는 얘기다.
#   그리고 2012년에 속도는 곧 "더 큰 모델을 훈련시킬 수 있다"는 뜻이었다.
# - 3주차 ResNet은 바로 이 "깊어지면 신호가 죽는다" 문제의 다음 챕터다.
#
# 💬 **부원에게 던질 질문**: ReLU는 음수를 전부 0으로 만든다. 정보를 버리는 건데
# 왜 더 잘 되는 걸까? (→ 희소성, 그리고 기울기 소실 회피)

# %% [markdown]
# ---
# # B. 네트워크는 대체 뭘 배웠나 — 논문 Fig.3 재현 ★
#
# AlexNet의 **첫 번째 conv 층**은 `(64, 3, 11, 11)` 짜리 가중치다.
# 입력 채널이 3(RGB)이라서, 이 가중치를 **그대로 11×11 컬러 이미지로 볼 수 있다.**
#
# 아무도 "엣지 검출기를 만들어라"라고 가르치지 않았다.
# 손실을 줄이라고만 했는데 경사하강법이 스스로 저걸 만들었다.

# %%
model, weights = prep.load_alexnet()

fig = prep.conv1_filter_grid(model)      # 라이브 실행 — 10초
fig.show()

# %% [markdown]
# 위 64개를 보면 크게 두 종류다.
#
# - **흑백 줄무늬** (방향·주파수가 제각각) → 엣지 검출기. 생물의 1차 시각피질에서
#   발견된 Gabor 필터와 거의 같은 모양이다.
# - **컬러 블롭** → 색 대비 검출기.
#
# 논문에서는 GPU 2장에 모델을 쪼개 넣었는데, 신기하게도
# **한쪽 GPU는 흑백 필터만, 다른 쪽은 컬러 필터만** 학습했다 (논문 Fig.3 설명).
# 시키지 않았는데 역할 분담이 생긴 것이다.

# %%
# 층이 깊어지면 뭘 보는가 — 특징맵
viz.show(ctx.asset("B_featuremaps.png"))

# %% [markdown]
# 엣지 → 질감 → 부분 형태(눈, 바퀴, 털 패턴) 순으로 점점 추상적이 된다.
# **사람이 "이 단계에서는 질감을 봐라"라고 설계한 게 아니다.** 계층 구조가 저절로 생겼다.
#
# 💬 **부원에게 던질 질문**: 그럼 마지막 층은 뭘 볼까? (→ 거의 "개다/고양이다" 수준의 개념.
# 그래서 마지막 층 직전의 벡터를 떼어내 다른 문제에 갖다 쓰는 게 전이학습이다.)

# %% [markdown]
# ---
# # C. 그래서 지금은? — 같은 사진, 10년 차이
#
# 세 모델에게 같은 사진을 보여준다.
#
# | 모델 | 연도 | 파라미터 | ImageNet top-1 | 비고 |
# |---|---|---|---|---|
# | AlexNet | 2012 | 61M | 56.5% | 오늘의 논문 |
# | ConvNeXt-Tiny | 2022 | 28M | 82.5% | 파라미터는 **절반 이하**인데 훨씬 정확 |
# | CLIP ViT-B/32 | 2021 | 151M | — | 분류기가 아예 없다. 라벨을 문장으로 준다 |

# %%
viz.show(ctx.asset("C_predictions.png"))

# %% [markdown]
# CLIP이 특히 이상하다. **1000개 고정 클래스가 없다.**
# "a photo of a dog" 같은 문장 후보를 그때그때 주면 그중에서 고른다.
# 즉 **학습할 때 본 적 없는 라벨도 즉석에서 만들어 쓸 수 있다** (zero-shot).
#
# AlexNet은 "1000개 중 하나를 고르는 기계"였는데, CLIP은 "이미지와 문장이 어울리는지
# 판단하는 기계"다. 이 차이가 7주차 GPT-3의 few-shot 이야기와 곧장 이어진다.

# %%
# ── 라이브: 부원 사진으로 돌려보기 ────────────────────────────────
# Colab에서 파일 업로드 → assets/samples/ 에 넣고 다시 비교한다.
if "google.colab" in sys.modules:
    from google.colab import files
    up = files.upload()
    for name, blob in up.items():
        (ctx.assets / "samples" / name).write_bytes(blob)
    prep.block_c()
    viz.show(ctx.asset("C_predictions.png"))
else:
    print("assets/samples/ 에 사진을 넣고 prep.block_c() 를 실행하면 된다.")

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. **ReLU** — 깊은 망에서 신호가 죽지 않게 한다 (A)
# 2. **학습된 필터** — 특징을 사람이 설계하지 않고 데이터에서 얻는다 (B)
# 3. 10년 뒤 모델은 더 작으면서 더 정확하고, 아예 **고정된 클래스 목록을 버렸다** (C)
#
# ### 다음 주 예고 — ResNet (2015)
# AlexNet은 8층이었다. "그럼 100층으로 만들면 더 좋지 않나?"
# 해봤더니 **더 나빠졌다.** 그것도 테스트가 아니라 *학습* 오차가.
# 왜 그런지, 어떻게 고쳤는지가 다음 주 논문이다.
