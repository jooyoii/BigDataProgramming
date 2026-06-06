# BigDataProgramming

Yelp 리뷰 텍스트와 수치형 핸드크래프트 특징을 함께 사용해 **fake review**를 분류하는 멀티모달 딥러닝 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 리뷰 본문 텍스트만 보는 대신, 텍스트와 함께 다음과 같은 보조 특징을 결합해 성능을 높이는 것을 목표로 합니다.

- 텍스트 특징: `review_text`
- 수치형 특징 그룹:
  - `basic_linguistic_list` 13개
  - `readability_list` 6개
  - `sentiment_list` 7개
  - `behavioral_list` 9개

총 35개의 수치형 특징을 사용하며, 텍스트와 수치형 정보를 **Gated Fusion** 구조로 결합합니다.

## 모델 구조

### 1. 텍스트 분기

- `bert-base-uncased` tokenizer 사용
- `MAX_LEN = 256`
- `padding='max_length'`, `truncation=True`
- `TFBertModel.from_pretrained('bert-base-uncased')` 사용
- BERT는 현재 `freeze` 상태(`bert.trainable = False`)
- `last_hidden_state`를 `Dense(128)`으로 투사 후 masked mean pooling 적용

### 2. 수치형 특징 분기

- `StandardScaler`로 정규화
- 특징을 3개 그룹으로 분리
  - linguistic: basic + readability + sentiment
  - behavioral: behavioral
- 각 분기에서 MLP를 통과시킨 뒤 결합

### 3. Gated Fusion

- 텍스트 벡터와 수치형 특징 벡터를 concat
- sigmoid gate로 두 표현의 기여도를 조절
- 최종적으로 `Dense(256) -> Dense(64) -> Dense(1)` classifier로 fake 여부 예측

## 학습 설정

- Optimizer: Adam
- Learning rate: `1e-4`
- Loss: `binary_crossentropy`
- Metrics: `accuracy`
- Batch size: `32`
- Epochs: `20`
- Early Stopping:
  - monitor: `val_loss`
  - patience: `5`
  - restore best weights: `True`

## 성능 결과

실행 결과 기준 성능은 다음과 같습니다.

- Accuracy: `0.7604`
- Precision: `0.7149`
- Recall: `0.8660`
- F1-score: `0.7833`

## 환경

`requirements.txt` 기준 주요 패키지:

- `tensorflow[and-cuda]==2.21.0`
- `transformers==4.46.3`
- `pandas==2.3.3`
- `scikit-learn==1.6.1`
- `torch==2.1.0`

## Repository Overview
- `baseline_BERT.ipynb`
  - BERT 기반 baseline 실험
- `baseline_CNN.ipynb`
  - CNN 기반 baseline 실험
- `baseline_LSTM.ipynb`
  - LSTM 기반 baseline 실험
- `baseline_RNN.ipynb`
  - RNN 기반 baseline 실험
- `baseline_RoBERTa.ipynb`
  - RoBERTa 기반 baseline 실험
- `gated_fusion.ipynb`
  - 텍스트 + 수치형 feature를 결합한 Gated Fusion 모델
- `preprocess.ipynb`
  - 데이터 전처리 및 feature 추출

## Main Ideas

- Review text classification
- Fake review detection
- BERT embeddings
- Multi-modal feature fusion
- Gated Fusion architecture

## Model Comparison

이 저장소는 단일 모델만 다루지 않고, 다음 구조들을 비교하는 데 초점을 둡니다.

1. Baseline models
   - CNN
   - LSTM
   - RNN
   - BERT
   - RoBERTa
2. Proposed model
   - Gated Fusion with text and handcrafted features

## Data Flow

1. Text and feature preprocessing
2. Tokenization with `bert-base-uncased`
3. Numerical feature scaling
4. Baseline model training
5. Gated Fusion training and evaluation

이 프로젝트는 리뷰 텍스트만으로 분류하는 방식보다, 텍스트와 행동/문체 기반 특징을 함께 활용하는 멀티모달 분류 전략을 실험하기 위한 것입니다.
