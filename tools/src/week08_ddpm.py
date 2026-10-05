# %% [markdown]
# # 8주차 · DDPM (2020) 데모 — 마지막 주
# ### *Denoising Diffusion Probabilistic Models*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 7분 | 노이즈를 더하고 되돌리는 과정 + 직접 학습한 생성 모델 ★ |
# | **B. 시각화** | 4분 | 스텝 수 vs 품질, 잠재공간 보간 |
# | **C. 현대 연결** | 6분 | SD-Turbo 로 부원 프롬프트 즉석 생성 |
#
# > **세션 시작 10분 전 실행 필수.** SD-Turbo 가중치가 약 2.5GB 내려온다.
# > 마지막 주니까 C 블록에 시간을 넉넉히 주고 마무리하는 걸 권한다.

# %%
REPO_URL = "https://github.com/ovrhit/dl-session-2026-2.git"

import os, sys
if "google.colab" in sys.modules and not os.path.exists("/content/dl-session"):
    !git clone -q $REPO_URL /content/dl-session
if os.path.exists("/content/dl-session"):
    %cd /content/dl-session

# %%
FAST_MODE = True   # False 면 tiny DDPM 을 처음부터 학습 (T4 약 12분)

from common.setup import bootstrap
ctx = bootstrap("week08_ddpm", fast_mode=FAST_MODE)

sys.path.insert(0, "week08_ddpm")
import prep
from common import viz
import json, torch
from IPython.display import Image, display

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# 지금까지 본 모델은 전부 **주어진 것을 판단하거나 이어쓰는** 모델이었다.
# 오늘은 **없던 걸 만들어내는** 모델이다.
#
# 생성 모델은 원래 학습이 어렵기로 악명이 높았다.
# GAN은 생성자와 판별자가 싸우다가 자주 무너지고(mode collapse),
# VAE는 안정적이지만 결과가 흐릿했다.
#
# > **DDPM의 발상**: 만드는 걸 직접 배우지 말고, **망가뜨리는 걸 거꾸로 배우자.**
#
# ```
# forward  (학습 불필요, 그냥 정해진 규칙)
#   원본 이미지  ──노이즈 조금씩 1000번──▶  완전한 잡음
#
# reverse  (이걸 배운다)
#   완전한 잡음  ──노이즈 조금씩 제거 1000번──▶  새 이미지
# ```
#
# 여기서 결정적인 게 있다. **모델이 배우는 건 "이미지를 만드는 법"이 아니다.**
# "이 그림에 섞인 노이즈가 뭐였는지 맞히기" 하나뿐이다.
# 그런데 그걸 1000번 반복하면 이미지가 나온다.

# %% [markdown]
# ---
# # A-1. forward — 이미지가 잡음이 되는 과정 ★

# %%
display(Image(filename=str(ctx.asset("A_forward.gif")), width=520))

# %% [markdown]
# 이 과정에는 **학습이 없다.** 그냥 정해진 양만큼 가우시안 노이즈를 섞는 것뿐이다.
#
# 논문 식 (4)가 이걸 한 줄로 만든다:
#
# $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$
#
# **중요한 건 이 식이 "한 번에" 계산된다는 것이다.**
# t=700 단계 이미지를 보고 싶으면 700번 반복할 필요 없이 바로 구한다.
# 이 닫힌 형태가 없었으면 학습이 불가능했다.

# %%
import inspect
print(inspect.getsource(prep.q_sample))

# %% [markdown]
# # A-2. 그래서 뭘 학습하나 — 논문 Algorithm 1
#
# 놀랄 만큼 짧다.
#
# 1. 학습 이미지 하나를 뽑는다
# 2. t를 1~1000 중에서 **무작위로** 하나 뽑는다
# 3. 그 t만큼 노이즈를 섞는다 (위 식으로 한 번에)
# 4. 모델에게 `x_t` 와 `t` 를 주고 **섞은 노이즈 ε을 맞히게 한다**
# 5. MSE 손실. 끝.
#
# GAN처럼 두 네트워크가 싸우지도 않고, VAE처럼 복잡한 하한도 없다.
# **그냥 회귀 문제 하나다.** 그래서 안정적으로 학습된다 — 이게 DDPM이 이긴 이유다.

# %%
if FAST_MODE:
    model = prep.load_model()
    print("체크포인트 로드 완료")
else:
    prep.block_a(epochs=30)          # T4 약 12분
    model = prep.load_model()

# %% [markdown]
# # A-3. reverse — 잡음에서 이미지가 떠오른다 ★

# %%
display(Image(filename=str(ctx.asset("A_reverse.gif")), width=520))

# %%
viz.show(ctx.asset("A_samples.png"), width=620)

# %% [markdown]
# Fashion-MNIST로 12분 학습한 결과다. 신발, 가방, 티셔츠 모양이 보인다.
# **학습 데이터에 없던 새 이미지다** — 외운 게 아니다 (B-2에서 확인한다).
#
# 💬 **부원에게 던질 질문**: 왜 하필 1000단계일까? 10단계면 안 되나?
# (→ 한 번에 많이 지우려면 모델이 훨씬 어려운 문제를 풀어야 한다.
# 조금씩 지우면 각 단계가 쉬워진다. **어려운 문제를 쉬운 문제 1000개로 쪼갠 것**이다.
# 그런데 그래서 느리다 → B 블록.)

# %%
# 라이브: 지금 바로 생성 (100 step, 약 5초)
x, _ = prep.sample(model, n=16, steps=100, eta=0.0, seed=torch.randint(0, 9999, (1,)).item())
fig = viz.image_grid([prep.grid_uint8(x)], ncols=1, cell=6, cmap="gray", title="방금 만든 것")
fig.show()

# %% [markdown]
# ---
# # B-1. 1000번은 너무 느리다
#
# 이미지 한 장에 forward pass 1000번. GPT는 토큰 하나에 1번인데 말이다.
# **DDPM의 가장 큰 약점이 이거였고**, 이후 연구의 절반이 이걸 줄이는 것이었다.

# %%
viz.show(ctx.asset("B_steps.png"), width=1250)
timing = json.loads(ctx.asset("B_steps.json").read_text(encoding="utf-8"))
for k, v in timing.items():
    print(f"  {k:18s} {v:6.1f}초")

# %% [markdown]
# **DDIM**(2021)의 아이디어: reverse 과정에서 무작위성(`eta`)을 빼면
# 단계를 **건너뛰어도** 된다. 1000 → 50 으로 줄여도 품질이 크게 안 떨어진다.
#
# 10 step까지 가면 눈에 띄게 뭉개진다. **어디까지가 감당 가능한 손해인지**를
# 직접 보는 게 이 그림의 목적이다.

# %% [markdown]
# # B-2. 외운 게 아니라는 증거 — 보간

# %%
viz.show(ctx.asset("B_interpolation.png"), width=1150)

# %% [markdown]
# 시작 잡음 두 개를 정하고 그 사이를 조금씩 섞어가며 생성한 것이다.
# 결과가 **뚝뚝 끊기지 않고 매끄럽게 변한다.**
#
# 만약 모델이 학습 이미지를 외웠다면 중간 지점에서 이상한 게 나오거나 급격히 점프해야 한다.
# 매끄럽다는 건 **"잡음 → 이미지"라는 연속적인 대응을 배웠다**는 뜻이다.

# %% [markdown]
# ---
# # C. 그래서 지금은? — 6년 뒤
#
# 오늘 만든 모델과 지금 쓰는 모델의 차이는 세 가지다.
#
# | | 오늘 (DDPM 2020) | SD-Turbo (2023~) |
# |---|---|---|
# | 어디에 노이즈를 넣나 | 픽셀에 직접 | **압축된 잠재공간**에 (훨씬 쌈) |
# | 조건 | 없음 (아무거나 나옴) | **텍스트** (CLIP 텍스트 인코더) |
# | 스텝 수 | 1000 | **1~4** (distillation) |
#
# 가운데 칸이 제일 크다. 텍스트 조건이 붙는 순간 **"만들어달라고 말할 수 있게" 된다.**
# 그리고 그 텍스트 인코더가 2주차에 봤던 **CLIP**이다.

# %%
# SD-Turbo 로드 (약 2.5GB — 세션 10분 전에 미리 실행해둘 것)
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if ctx.has_gpu else torch.float32,
).to(ctx.device)
pipe.set_progress_bar_config(disable=True)
print("준비 완료")

# %%
# ── 라이브: 부원 프롬프트 받기 ────────────────────────────────────
PROMPT = "a cozy university club room at night, warm lamp light, laptops open, cinematic"

import time
t0 = time.perf_counter()
img = pipe(prompt=PROMPT, num_inference_steps=2, guidance_scale=0.0).images[0]
print(f"{time.perf_counter() - t0:.1f}초")
img

# %% [markdown]
# 방금 2 step으로 만들었다. 오늘 우리 모델은 100 step이 걸렸다.
# **같은 원리인데 6년 사이에 50배 빨라지고 텍스트로 조종할 수 있게 된 것이다.**

# %%
# 스텝 수를 바꿔가며 — 오늘 B-1과 같은 실험을 최신 모델로
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
for ax, steps in zip(axes, [1, 2, 4, 8]):
    im = pipe(prompt=PROMPT, num_inference_steps=steps, guidance_scale=0.0).images[0]
    ax.imshow(im); ax.set_title(f"{steps} step"); ax.axis("off")
fig.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. 만드는 걸 배우지 말고 **망가뜨리는 걸 거꾸로** 배운다 (A-1)
# 2. 실제로 배우는 건 "섞인 노이즈 맞히기" 하나. 그래서 학습이 안정적이다 (A-2)
# 3. 대신 느리다 — 이후 연구의 절반이 이 스텝 수를 줄이는 것이었다 (B-1)
# 4. 지금은 잠재공간 + 텍스트 조건 + 1~4 step (C)
#
# ---
# ## 8주 전체 정리
#
# | 주차 | 논문 | 남긴 것 |
# |---|---|---|
# | 2 | AlexNet | 특징을 **설계하지 말고 학습**시킨다 |
# | 3 | ResNet | `x + f(x)` — 깊게 쌓을 수 있게 됐다 |
# | 4 | RNN LM | 순서가 있는 데이터, 그리고 **기억의 한계** |
# | 5 | Attention | 하나로 압축하지 말고 **필요한 곳을 골라 본다** |
# | 6 | Transformer | recurrence를 버리니 **병렬화**가 됐다 |
# | 7 | GPT-3 | 키웠더니 **가중치를 안 고치고도** 새 문제를 푼다 |
# | 8 | DDPM | 역과정을 배워서 **만들어낸다** |
#
# 지금 쓰는 모델은 거의 전부 이 7개의 조합이다.
# ChatGPT = 6 + 7, 이미지 생성 = 2(CLIP) + 3 + 6 + 8.
#
# **논문 읽는 법 관점에서 남길 것 하나**:
# 8주 동안 본 논문들은 전부 *"기존 방법이 정확히 어디서 무너지는가"* 를
# 먼저 보여주고, 그 지점만 공격했다.
# 새 논문을 읽을 때도 **"이 논문이 공격하는 실패 지점이 뭔가"** 를 먼저 찾으면
# 나머지는 따라온다.
#
# 한 학기 수고하셨습니다 🎉
