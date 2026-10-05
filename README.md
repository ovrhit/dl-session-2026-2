# 2026-2 딥러닝 논문 세션 — 데모 자료

딥러닝 입문자를 위한 8주 논문 세션의 **주차별 데모** 저장소.
매주 `사전 설명 → 논문 읽기 → 논문 해설` 뒤에 **15~20분 데모**를 붙인다.

---

## 주차별 구성

모든 주차가 같은 3블록 구조다. 리듬이 생기고, 준비할 것도 예측 가능해진다.

| 블록 | 시간 | 역할 |
|---|---|---|
| **A. 재현** | 6~8분 | 논문의 대표 Figure를 직접 재현 — "논문이 진짜 그런지 확인" |
| **B. 시각화** | 4~5분 | 그 주차의 킬러 이미지. 발표 슬라이드에 그대로 쓸 자료 |
| **C. 현대 연결** | 4~5분 | 같은 아이디어가 지금 어떻게 살아있는지. 부원 입력 받는 라이브 파트 |

| 주차 | 논문 | A. 재현 | B. 시각화 | C. 현대 연결 |
|---|---|---|---|---|
| 2 | AlexNet (2012) | ReLU vs tanh 수렴 속도 (Fig.1) | **1층 conv 필터 그리드** (Fig.3) | ConvNeXt / CLIP 과 예측 비교 |
| 3 | ResNet (2015) | **plain vs residual degradation** (Fig.6) | 층별 gradient norm | LLM 블록에도 있는 `x + f(x)` / Grad-CAM |
| 4 | RNN LM (2010) | n-gram vs RNN perplexity | **생성 문장의 변화 + vanishing gradient** | GPT-2 와 비교 |
| 5 | Bahdanau Attention (2014) | **문장 길이별 BLEU** (Fig.2) | attention 정렬 히트맵 (Fig.3) | opus-mt 의 cross-attention |
| 6 | Transformer (2017) | self-attention 구현 + 속도 비교 | **head 패턴 + positional encoding** | 2017 블록 vs Llama 블록 diff |
| 7 | GPT-3 (2020) | **모델 크기 × shot 수 격자** (Fig.1.2) | 프롬프트 원문 + 토크나이저 | instruct 모델 0-shot |
| 8 | DDPM (2020) | **forward/reverse 애니메이션 + 생성** | 스텝 수 vs 품질, 보간 | SD-Turbo 라이브 생성 |

---

## Colab에서 바로 열기

| | | |
|---|---|---|
| [![2주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week02_alexnet/demo.ipynb) 2주차 AlexNet | [![3주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week03_resnet/demo.ipynb) 3주차 ResNet | [![4주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week04_rnn/demo.ipynb) 4주차 RNN |
| [![5주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week05_attention/demo.ipynb) 5주차 Attention | [![6주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week06_transformer/demo.ipynb) 6주차 Transformer | [![7주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week07_gpt3/demo.ipynb) 7주차 GPT-3 |
| [![8주차](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovrhit/dl-session-2026-2/blob/main/week08_ddpm/demo.ipynb) 8주차 DDPM | | |

**런타임 유형을 T4 GPU로 바꾼 뒤** 실행할 것.

---

## 세션 당일 진행 방법

1. **세션 10분 전**: 해당 주차 `demo.ipynb` 를 Colab에서 열고 **셋업 셀 두 개를 실행**해둔다.
   (모델 가중치 다운로드를 세션 중에 시작하면 몇 분을 통째로 날린다)
2. `FAST_MODE = True` 인지 확인. 미리 만들어둔 에셋을 로드하므로 전체가 5분 이내로 끝난다.
3. 위에서 아래로 실행하면서 markdown 셀의 대본을 그대로 읽으면 된다.

**런타임 유형은 T4 GPU.** GPU 할당이 안 되면 `FAST_MODE=True` 경로는 CPU로도
그림이 다 뜨므로 데모 자체는 죽지 않는다 (라이브 생성 파트만 건너뛴다).

---

## 사전 준비 (세션 1주일 전)

각 주차의 무거운 학습·평가는 `prep.py` 로 분리돼 있다. Colab T4에서 미리 돌려 `assets/` 를 채운다.

```bash
python week03_resnet/prep.py            # 전체
python week03_resnet/prep.py --only A   # 블록 하나만
```

| 주차 | 예상 소요 (T4) | 비고 |
|---|---|---|
| 2 | 약 15분 | `assets/samples/` 에 **본인/부원 사진**을 넣어두면 C 블록이 훨씬 재밌다 |
| 3 | **약 50분** | 가장 오래 걸린다. 최소 3일 전에 돌릴 것 |
| 4 | 약 15분 | `assets/corpus.txt` 에 **한국어 텍스트**를 넣으면 B 블록 생성 데모가 훨씬 재밌다 |
| 5 | **약 40분** | 체크포인트 2개 생성 |
| 6 | 약 20분 | 4주차 코퍼스를 재사용한다 (같은 데이터라야 비교가 성립) |
| 7 | **약 30분** | 모델 가중치 총 5GB 이상 다운로드 |
| 8 | 약 20분 | GIF 2개 + 체크포인트 |

---

## 폴더 구조

```
.
├─ common/               # 공통 유틸
│  ├─ setup.py           #   Colab 셋업: 의존성, 한글 폰트, seed, device
│  ├─ viz.py             #   발표 화면용 matplotlib 설정 + 플롯 헬퍼
│  └─ requirements.txt   #   핵심 의존성 (torch 는 여기 넣지 말 것 — 아래 참고)
├─ tools/
│  ├─ src/weekNN_*.py    # 노트북 원본 (percent 포맷)
│  ├─ nbgen.py           #   → .ipynb 변환기
│  └─ build.py           #   전체 재생성
├─ weekNN_*/
│  ├─ demo.ipynb         # 세션 진행용 노트북 (해설 대본이 markdown 셀에 들어있다)
│  ├─ prep.py            # 사전 실행 → assets/ 생성
│  ├─ requirements.txt   # 그 주차 추가 의존성 (있는 주차만)
│  └─ assets/            # 미리 뽑아둔 PNG/GIF/체크포인트
└─ README.md
```

### 노트북을 고칠 때

`demo.ipynb` 는 **생성물**이다. 구조를 바꿀 거면 `tools/src/` 를 고치고:

```bash
python tools/build.py
```

Colab에서 직접 고친 내용은 `tools/src` 로 돌아오지 않는다.
세션 당일의 즉흥 수정은 그날만 쓰고 버리는 용도로 생각하면 된다.

---

## 설정해야 할 것

1. **체크포인트 배포** — 5·8주차 체크포인트는 용량이 있어서 `.gitignore` 에 들어있다.
   git에 올리려면 `git add -f`, 아니면 HF Hub에 올리고 `common/setup.py` 의
   `HF_ASSET_REPO` 를 본인 계정으로 바꿔 `download_assets()` 를 쓴다.

> ⚠️ `common/requirements.txt` 에 **torch / torchvision 을 추가하지 말 것.**
> Colab의 CUDA 빌드가 CPU 빌드로 덮어써져서 GPU가 죽는다.

---

## 검증 체크리스트

각 주차 노트북은 세션 전에 아래 두 가지를 통과시킨다.

- [ ] **세션 시뮬레이션** — Colab 무료 티어 **새 런타임**에서 `FAST_MODE=True` 로 전체 셀 실행
  - [ ] 총 실행 시간 5분 이내
  - [ ] 그래프의 한글 라벨이 안 깨짐
  - [ ] GPU 없는 런타임에서도 에셋 그림은 전부 뜸 (장애 대비)
- [ ] **재현 검증** — `FAST_MODE=False` 로 1회 완주, `prep.py` 결과와 일치
