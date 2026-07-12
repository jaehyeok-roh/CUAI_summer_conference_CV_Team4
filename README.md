# CUAI_summer_conference_CV_Team4
CUAI 하계 커너퍼런스 CV 4팀

## Lightweight Edge-Preserving JDC Network for Extreme Low-Light Environments 
> **CUAI 2026 하계 컨퍼런스 프로젝트**  
> 저조도 노이즈 환경 특화 End-to-End 동시 압축-복원 네트워크

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CompressAI](https://img.shields.io/badge/CompressAI-000000?style=for-the-badge&logo=github&logoColor=white)](https://interdigitalinc.github.io/CompressAI/)

### 📢 Overview
본 프로젝트는 통신망이 열악한 엣지(Edge) 환경(방범용 CCTV, 야간 드론 등)에서 촬영된 **저조도 노이즈 영상을 효율적으로 압축하고 선명하게 복원하기 위한 딥러닝 코덱**을 제안합니다. 

기존의 딥러닝 기반 이미지 압축 모델은 맑은 날씨의 데이터로만 학습되어, 야간 영상의 '노이즈'를 중요한 정보로 착각해 전송 용량(Bitrate)을 낭비하거나 디테일(글씨, 테두리 등)을 심하게 뭉개는(Oversmoothing) 한계가 있습니다. 이를 해결하기 위해 본 연구는 **CBAM(어텐션 모듈)** 과 **Edge-preserving Loss(윤곽선 보존 손실함수)** 를 결합한 새로운 아키텍처를 제안합니다.

<br>

### 🚨 Problem Statement
사전 학습된 기존 모델(`bmshj2018-hyperprior`)에 저조도 및 야간 거리를 입력했을 때 다음과 같은 치명적인 문제가 발생함을 확인했습니다.
1. **텍스트 정보 소실:** 간판의 작은 글씨들이 뭉개져 OCR 인식이 불가능해짐.
2. **고주파 텍스처(High-frequency Texture) 뭉개짐:** 아스팔트 바닥 등의 복잡한 질감이 찰흙처럼 매끈하게(Blur) 변형됨.
3. **노이즈로 인한 비트레이트 낭비:** 비선형적 센서 노이즈를 보존하려다 압축 효율이 저하됨.

| 원본 이미지 (Original) | 압축 복원 이미지 (Reconstructed - 기존 모델) |
| :---: | :---: |
| <img width="611" height="382" alt="image" src="https://github.com/user-attachments/assets/af7bf95e-30d2-499e-a719-bd8568ed3840" width="300"> | <img width="611" height="380" alt="image" src="https://github.com/user-attachments/assets/801ee66d-aa8f-4612-898b-8b4e7ccf72d0" width="300"> |
| *어둡고 노이즈가 낀 원본* | *글씨와 엣지가 녹아내린 복원 결과* |

<br>

### 💡 제안 방법 (Proposed Method)

#### 1. Data-Level: 저조도-정상조도 페어 학습
* **Dataset:** LOL (Low-Light) Dataset 활용
* 어둡고 노이즈 낀 사진을 Input으로, 밝고 깨끗한 사진을 Ground Truth로 설정하여 **"압축과 동시에 노이즈를 제거(Denoising)"** 하도록 유도.

#### 2. Architecture-Level: Attention Module (CBAM)
* CompressAI 인코더의 Bottleneck 직전에 **공간/채널 어텐션 모듈(CBAM)** 추가.
* 모델이 불필요한 노이즈 Feature는 억제하고, 보존해야 할 객체의 형태 Feature에 집중(Attention)하도록 구조 개선.

#### 3. Loss-Level: Edge-Preserving Loss
* 기존 압축 모델의 Rate-Distortion Loss (BPP + MSE)는 픽셀 간 단순 오차만 계산하여 Blur 현상을 유발.
* 형태 보존력을 극대화하기 위해 **Sobel Filter 기반의 Edge Loss**를 추가 도입하여 손실 함수 최적화.

<br>

## 🛠️ 개발 환경 및 설치 (Prerequisites & Installation)

```bash
# 1. 가상환경 생성 및 활성화 (선택)
conda create -n neural_comp python=3.10
conda activate neural_comp

# 2. 필수 라이브러리 설치
pip install torch torchvision
pip install compressai
pip install matplotlib numpy opencv-python
```
<br>

## 🚀 Quick Start
(추후 작성 예정)

<br>

## 👥 Team  - 추후 수정예정
인선우 - Data Preprocessing, Loss Function 커스텀, Training Pipeline 구축 

노재혁 - CompressAI Architecture 분석, CBAM 결합, Evaluation & Visualization

<br>

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
본 프로젝트는 중앙대학교 인공지능 학회 CUAI 2026 하계 컨퍼런스의 일환으로 진행되었습니다.

Base code is built upon CompressAI.
