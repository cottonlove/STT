# STT
Developing a specialized speech recognition model for Korean regional dialects.


# Dataset
AI-Hub에서 제공하는 중·노년층 한국어 방언 데이터 (경상도)를 학습데이터셋으로 선택

‘1인 발화-따라 말하기 유형’ 중 약 30시간, 3.2GB를 학습 및 검증 데이터셋으로 약 8시간, 850MB를 테스트 데이터셋으로 추출하여 구성함.

## Setup
실험 환경: 4 NVIDIA-GeForce-RTX-3090
Configuration – Deepspeech2
Number of encoders: 3, 5
Optimizer: Adam
Epoch: 50, 40
Batch size: 128, 64
Train:Val = 8:2

## CER
Deepspeech2 (#encoders: 3)
- Pronunciation 전사기준: 0.267
- Dialect 전사기준: 0.276
Deepspeech2 (#encoders: 5)
- Pronunciation 전사기준: 0.212
- Dialect 전사기준: 0.225

## Train loss, CER
<img src="/uploads/1848994ad25765da30fa8ef3684c67bc/캡처.PNG"  width="700" height="370">
<img src="/uploads/1848994ad25765da30fa8ef3684c67bc/캡처.PNG"  width="700" height="370">

## Validation loss, CER
<img src="/uploads/1848994ad25765da30fa8ef3684c67bc/캡처.PNG"  width="700" height="370">
<img src="/uploads/1848994ad25765da30fa8ef3684c67bc/캡처.PNG"  width="700" height="370">

베이스라인 코드는 김수환 님께서 개발해 공개하신 kospeech (https://github.com/sooftware/kospeech) 를 기반으로 하였습니다.
