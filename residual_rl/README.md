# Residual RL Finetuning for PlanarPeg

이 디렉토리는 `gym-PlanarPeg` 환경에서 Out-of-Distribution (OOD) 문제를 해결하기 위한 **Residual RL 파인튜닝 파이프라인**을 포함하고 있습니다. 이 파이프라인은 사전 학습된 Diffusion Policy를 Base Policy로 사용하고, 그 위에 추가적인 변위(Residual)를 예측하는 **RLPD (TD3 기반) 에이전트**를 학습시킵니다. 알고리즘 및 신경망 아키텍처는 원작 `@resfit` 논문의 구현을 100% 동일하게 차용하였습니다.

또한, 오프라인 시연 데이터를 활용하여 DINOv2 기반의 잠재 공간(Latent Space)을 학습하고, 이를 바탕으로 Dense Reward(LaNE 방법론)를 제공하여 강화학습의 탐색 효율성을 극대화합니다.

## 📂 디렉토리 구조

- `configs/`: 설정 파일 (`train.yaml` 등)
- `models/`: 핵심 신경망 모듈
  - `actor.py`: Residual TD3 Actor (시각 특징(Vision) + State를 입력받아 잔차를 결정론적으로 예측, L2 Penalty 적용)
  - `critic.py`: REDQ Critic 앙상블 (Q-value 과대추정 방지, 시각 특징(Vision) 포함)
  - `latent_encoder.py`: DINOEncoder 및 MLPE2C (Front/Top-view 잠재 공간 및 국소적 선형 동역학 학습)
- `scripts/`: 실행 스크립트
  - `train_latent_encoder.py`: 1단계 - 잠재 공간 학습 및 데모 캐싱
  - `train_residual_rl.py`: 2단계 - Residual RL 파인튜닝 (RLPD 방식 적용)
- `utils/`: 유틸리티 모듈
  - `normalization.py`: 액션 및 상태 정규화
  - `replay_buffer.py`: Symmetric Replay Buffer (온라인/오프라인 50:50 유지)
  - `reward.py`: LaNE 기반 잠재 공간 유사도 추가 보상 계산
- `wrappers/`: 환경 래퍼
  - `residual_env_wrapper.py`: Base Policy를 내재화하여 Residual Action과 결합하는 래퍼

---

## 🚀 사용 방법

### 0. 사전 준비 (Prerequisites)
1. **Base Policy 학습 완료**: `IL_policy` 폴더를 통해 Diffusion Policy가 학습되어 있어야 합니다. (예: `outputs/planar_peg-diffusion-...`)
2. **오프라인 데이터셋**: 기본 데이터셋 경로는 `data/ID_base/ID_base_v2.zarr` 로 설정되어 있습니다.
3. **설정 수정 (`configs/train.yaml`)**: 
   - `base_policy.run_dir` 값을 **학습된 Base Policy의 출력 디렉토리 경로**로 변경해주세요.

### 1단계: 잠재 공간 학습 및 데모 캐싱 (Phase 1)
Front-view (또는 모든 카메라) 이미지의 DINOv2 임베딩을 추출하고, MLPE2C 모델을 통해 16차원 잠재 공간을 학습합니다. 이 과정에서 Locally-Linear Dynamics(E2C)가 적용되어 궤적이 부드럽게 펴지게 됩니다.

```bash
cd /home/ysl2683/gym-PlanarPeg
python -m residual_rl.scripts.train_latent_encoder
```
- 실행이 완료되면 `residual_rl/outputs/latent_encoder-YYYYMMDD_HHMMSS/` 폴더가 생성되며, 그 안에 `latent_cache.pt`가 저장됩니다.
- 멀티 카메라(Front+Top)를 활용하시려면 `--use_all_cameras` 인자를 추가하세요.

### 2단계: Residual RL 파인튜닝 (Phase 2)
1단계에서 생성된 `latent_cache.pt` 파일의 경로를 `--latent_cache_path` 인자로 전달하여 훈련을 시작합니다.

```bash
cd /home/ysl2683/gym-PlanarPeg
python -m residual_rl.scripts.train_residual_rl \
    --config residual_rl/configs/residual_sac.yaml \
    --base_policy_run_dir <IL_policy_결과_디렉토리_경로> \
    --latent_cache_path <1단계에서_생성된_latent_cache.pt_경로>
```

#### 📌 핵심 훈련 로직 (@resfit 100% 반영)
1. **결정론적 탐험 (TD3/RLPD)**: Actor는 평균(Mean) 행동만 출력하며, 훈련 시 스케줄러에 따라 외부 가우시안 노이즈(TruncatedNormal)를 주입하여 탐험합니다.
2. **Action L2 Regularization**: 잔차 행동이 너무 커지지 않도록 Actor의 Loss에 행동 제곱합을 페널티로 명시적으로 더합니다.
3. **Visual 피쳐 입력**: Actor와 Critic 모두 DINOv2 시각 임베딩을 압축한 피쳐(50차원)와 로봇 상태, Base 행동을 병합(Concat)하여 입력으로 사용합니다.
4. **REDQ 앙상블 Critic**: 여러 개의 Q-네트워크 중 무작위 서브셋의 최솟값을 취해 과대추정을 방지합니다.
5. **Dense Reward**: 매 스텝 환경 보상(Sparse) 외에, 현재 시야가 캐싱된 데모 궤적 중 얼마나 유사한지에 따라 시각적 잠재 거리 기반 보상(LaNE)이 부여됩니다.
