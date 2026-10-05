# %% [markdown]
# # 7주차 · GPT-3 (2020) 데모
# ### *Language Models are Few-Shot Learners*
#
# | 블록 | 시간 | 내용 |
# |---|---|---|
# | **A. 재현** | 7분 | 모델 크기 × 예시 개수 격자 — 논문 Fig.1.2 ★ |
# | **B. 시각화** | 5분 | 프롬프트 원문 + 토크나이저의 함정 |
# | **C. 현대 연결** | 5분 | few-shot이 필요 없어진 세계 |
#
# > **세션 시작 10분 전 실행 필수.** 이번 주는 모델 가중치가 총 5GB 넘게 내려온다.

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
ctx = bootstrap("week07_gpt3", fast_mode=FAST_MODE)

sys.path.insert(0, "week07_gpt3")
import prep
from common import viz
import json, random

# %% [markdown]
# ---
# ## 오늘 논문이 주장하는 것
#
# 6주차 Transformer 구조를 **거의 그대로 두고 크기만 키웠다.**
# 파라미터 1,750억 개, 그때까지 가장 큰 모델의 10배.
#
# 여기까지는 "돈을 더 썼다"는 얘기다. 논문이 진짜로 말하려던 건 그 다음이다.
#
# > **모델을 고치지 않고, 프롬프트에 예시 몇 개만 넣어주면 새 문제를 푼다.**
#
# 이게 왜 이상한가. 그때까지 딥러닝에서 새 문제를 풀려면 반드시
# **fine-tuning**(그 문제의 데이터로 가중치를 다시 학습)이 필요했다.
# 그런데 GPT-3는 **가중치를 전혀 안 건드린다.** 그냥 입력 텍스트가 길어질 뿐이다.
#
# 논문은 이걸 **in-context learning** 이라고 부른다.
#
# | 이름 | 예시 개수 | 가중치 수정 |
# |---|---|---|
# | zero-shot | 0 | 없음 |
# | one-shot | 1 | 없음 |
# | few-shot | 10~100 | 없음 |
# | fine-tuning | 수천~ | **있음** |

# %% [markdown]
# ---
# # A. 크기를 키워야만 생기는 능력인가 — 논문 Fig.1.2 재현 ★
#
# ### 왜 Pythia 시리즈를 쓰는가
#
# 논문의 주장은 **"크기만 키웠더니"** 다. 이걸 확인하려면 크기 말고 전부 같아야 한다.
# 그런데 대부분의 모델 시리즈(GPT, Qwen, Llama…)는 크기별로 학습 데이터도, 레시피도 다르다.
# 그러면 **"크기 때문인지 데이터 때문인지 구분이 안 된다."**
#
# Pythia는 160M부터 12B까지 **완전히 같은 데이터를 같은 순서로** 먹인 시리즈다.
# 그래서 "크기만의 효과"를 보려는 이 실험에 정직하다.
#
# 💬 **논문 읽는 법 관점**: 논문의 비교표를 볼 때 항상
# **"통제된 변수가 정말 하나인가"** 를 확인하는 습관. 이게 안 지켜진 비교가 생각보다 많다.
#
# 태스크는 GPT-3 논문 3.9절에 실제로 나오는 것들을 썼다.

# %%
if FAST_MODE:
    d = json.loads(ctx.asset("A_scaling.json").read_text(encoding="utf-8"))
    viz.show(ctx.asset("A_scaling.png"), width=1250)
else:
    prep.block_a(n=40)               # T4 약 25분
    viz.show(ctx.asset("A_scaling.png"), width=1250)
    d = json.loads(ctx.asset("A_scaling.json").read_text(encoding="utf-8"))

# %%
# 숫자로 보기 — 철자 뒤집기 태스크
task = "철자 뒤집기"
print(f"{'모델':>12} " + " ".join(f"{k}-shot" for k in d["shots"]))
for label in d["results"]:
    row = d["results"][label][task]
    print(f"{label:>12} " + " ".join(f"{row[str(k)]:6.2f}" for k in d["shots"]))

# %% [markdown]
# ### 짚어줄 포인트
#
# - **작은 모델은 예시를 줘도 소용없다.** 가로선에 가깝다.
#   예시를 8개 줘도 0개 줬을 때와 별 차이가 없다 → "예시를 활용하는 능력" 자체가 없다.
# - **모델이 커지면 shot 곡선이 벌어지기 시작한다.** 이게 핵심이다.
#   정확도가 올라간 게 아니라 **"예시를 보고 배우는 능력"이 생긴 것**이다.
# - 논문 Figure 1.2가 정확히 이 모양이고, 이게 "scale이 새로운 능력을 만든다"는
#   주장의 근거가 됐다.
#
# > ⚠️ **정직하게 짚을 것**: 우리가 쓴 최대 모델은 2.8B다. GPT-3는 **175B**,
# > 60배 크다. 논문의 극적인 곡선(특히 산술)은 10B 넘어가야 제대로 보인다.
# > 여기서는 **추세의 시작**만 보이는 것이고, 그게 무료 GPU의 한계다.
# > 이런 한계를 숨기지 않고 말하는 게 논문을 읽는 자세이기도 하다.
#
# 💬 **부원에게 던질 질문**: 예시를 보고 배운다는데, 다음 문제를 풀 때
# 아까 예시를 기억할까? (→ **못 한다.** 매번 프롬프트에 다시 넣어줘야 한다.
# 가중치가 안 바뀌었으니 당연하다. 이게 지금 "메모리"와 RAG 연구의 출발점이다.)

# %% [markdown]
# ---
# # B-1. few-shot 프롬프트는 실제로 이렇게 생겼다
#
# "few-shot"이라는 말이 추상적으로 들리는데, 진짜 별거 없다.
# **그냥 텍스트다.**

# %%
prompts = json.loads(ctx.asset("B_prompts.json").read_text(encoding="utf-8"))

for k in ("0", "1", "4"):
    print(f"{'═' * 20} {k}-shot {'═' * 20}")
    print(prompts["철자 뒤집기"][k])
    print()

# %% [markdown]
# 모델이 하는 일은 여전히 **"다음 토큰 맞히기"** 뿐이다.
# 그런데 앞에 `Q:/A:` 패턴이 몇 번 반복되면, 다음에 올 법한 것이
# **"그 패턴을 따르는 답"** 이 된다. 별도의 학습 알고리즘이 도는 게 아니다.
#
# 4주차에서 문자 RNN이 "따옴표 짝 맞추기"를 배웠던 것과 같은 종류의 일이,
# 훨씬 추상적인 수준에서 일어나는 것이다.

# %%
# 라이브: 예시 개수를 바꿔가며 직접 본다
model, tok = prep.load_model("EleutherAI/pythia-1.4b")

for k in (0, 2, 8):
    p, gold = prep.build_prompt(prep.task_reverse, random.Random(7), k)
    ins = tok(p, return_tensors="pt").to(ctx.device)
    out = model.generate(**ins, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[{k}-shot] 정답={gold:10s} 모델={gen.strip().splitlines()[0].strip()!r}")

# %% [markdown]
# # B-2. 토크나이저 — LLM이 이상하게 실패하는 이유의 절반
#
# 모델은 글자를 보지 않는다. **토큰**을 본다.
# 이게 "LLM이 왜 저런 걸 못하지?" 싶은 실패의 상당수를 설명한다.

# %%
probes = json.loads(ctx.asset("B_tokenizer.json").read_text(encoding="utf-8"))
for p in probes:
    print(f"{p['text']!r}")
    print(f"   → {p['n_tokens']}개 토큰: {p['tokens']}")
    print(f"   {p['note']}\n")

# %% [markdown]
# - **"strawberry에 r이 몇 개?"** — 모델은 `st/raw/berry` 같은 덩어리로 본다.
#   글자 단위 정보가 애초에 입력에 없다. 못 세는 게 당연하다.
# - **숫자** — 자릿수대로 안 쪼개진다. 그래서 큰 수 덧셈이 유독 약하다.
# - **한국어** — 같은 내용이 영어보다 토큰을 2~3배 먹는다.
#   같은 컨텍스트 길이에 한국어는 훨씬 적게 들어가고, API 요금도 더 나온다.
#
# 💬 이건 "모델이 멍청해서"가 아니라 **입력 표현의 한계**다.
# 모델 실패를 볼 때 "능력 부족"과 "표현 문제"를 구분하는 게 중요하다.

# %% [markdown]
# ---
# # C. 그래서 지금은? — few-shot이 필요 없어졌다
#
# GPT-3(2020)는 예시를 보여줘야 했다. 지금 모델은 **그냥 시키면 된다.**
# 같은 태스크를 instruct 모델에 **0-shot, 자연어 지시만으로** 돌린 결과다.

# %%
inst = json.loads(ctx.asset("C_instruct.json").read_text(encoding="utf-8"))
base = json.loads(ctx.asset("A_scaling.json").read_text(encoding="utf-8"))["results"]

rows = []
for task in inst["accuracy"]:
    rows.append({
        "태스크": task,
        "pythia-1.4b 0-shot": f"{base['pythia-1.4b'][task]['0']:.2f}",
        "pythia-1.4b 8-shot": f"{base['pythia-1.4b'][task]['8']:.2f}",
        f"{inst['model'].split('/')[-1]} 0-shot": f"{inst['accuracy'][task]:.2f}",
    })
viz.table(rows)

# %% [markdown]
# 비슷한 크기(1.4B vs 1.5B)인데 **예시 없이도** 더 잘 한다.
# 크기가 아니라 **학습 방식**이 달라서다.
#
# ```
# GPT-3 (2020)          다음 토큰 예측만.  → 예시를 줘야 패턴을 잡는다
#   ↓ instruction tuning   "지시 → 답" 데이터로 추가 학습
#   ↓ RLHF                 사람이 선호하는 답을 강화
#   ↓ reasoning            답하기 전에 길게 생각하도록 학습
# 지금 모델               지시만 하면 됨
# ```
#
# 그러니까 GPT-3 논문은 **"크기를 키우면 된다"의 증명**이자,
# 동시에 **"그것만으로는 쓰기 불편하다"의 출발점**이기도 하다.
# ChatGPT가 GPT-3가 아니라 그 뒤에 나온 이유가 이것이다.

# %%
# 라이브: 부원이 문제를 직접 내본다
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mid = prep.INSTRUCT_ID
itok = AutoTokenizer.from_pretrained(mid)
imodel = AutoModelForCausalLM.from_pretrained(
    mid, torch_dtype=torch.float16 if ctx.has_gpu else torch.float32
).eval().to(ctx.device)

QUESTION = "다음 문장을 세 단어로 요약해: 딥러닝 세션에서 논문을 읽고 데모를 돌려봤다."

chat = [{"role": "user", "content": QUESTION}]
text = itok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
ins = itok(text, return_tensors="pt").to(ctx.device)
out = imodel.generate(**ins, max_new_tokens=120, do_sample=True, temperature=0.7,
                      pad_token_id=itok.eos_token_id)
print(itok.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True))

# %% [markdown]
# ---
# ## 오늘 정리
#
# 1. 구조는 그대로, 크기만 키웠다 — 그런데 **새로운 능력**이 생겼다 (A)
# 2. few-shot은 마법이 아니라 **그냥 긴 프롬프트**다. 가중치는 안 변한다 (B-1)
# 3. 토크나이저는 LLM 실패의 절반을 설명한다 (B-2)
# 4. 지금은 instruction tuning 덕에 예시 없이도 된다 (C)
#
# ### 다음 주 예고 — DDPM (2020), 마지막 주
# 지금까지 본 모델은 전부 **판별**(이건 개다) 이거나 **이어쓰기**(다음 단어는 이거다) 였다.
# 마지막 주는 **없던 걸 만들어내는** 모델이다.
# 접근이 아주 이상하다: **이미지를 완전히 망가뜨린 다음, 망가뜨리는 과정을 거꾸로 배운다.**
