# Residual RL Finetuning for PlanarPeg

이 디렉토리는 `gym-PlanarPeg` 환경에서 Out-of-Distribution (OOD) 문제를 해결하기 위한 **Residual RL 파인튜닝 파이프라인**을 포함하고 있습니다. 이 파이프라인은 사전 학습된 Diffusion Policy를 Base Policy로 사용하고, 그 위에 추가적인 변위(Residual)를 예측하는 SAC(Soft Actor-Critic) 에이전트를 학습시킵니다. 

또한, 오프라인 시연 데이터를 활용하여 DINOv2 기반의 잠재 공간(Latent Space)을 학습하고, 이를 바탕으로 Dense Reward(LaNE 방법론)를 제공하여 강화학습의 탐색 효율성을 극대화합니다.

## 📂 디렉토리 구조

- `configs/`: 설정 파일 (`residual_sac.yaml`)
- `models/`: 핵심 신경망 모듈
  - `actor.py`: Residual SAC Actor (잔차 예측, `action_scale`로 출력 범위 제한)
  - `critic.py`: REDQ Critic 앙상블 (Q-value 과대추정 방지)
  - `latent_encoder.py`: DINOFrontEncoder 및 MLPE2C (Front-view 잠재 공간 학습)
- `scripts/`: 실행 스크립트
  - `train_latent_encoder.py`: 1단계 - 잠재 공간 학습 및 데모 캐싱
  - `train_residual_sac.py`: 2단계 - Residual RL 파인튜닝
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
2. **오프라인 데이터셋**: 기본 데이터셋 경로는 `data/ID_base/ID_base_v2.zarr` 로 설정되어 있습니다. (약 50개의 에피소드 분량)
3. **설정 수정 (`configs/residual_sac.yaml`)**: 
   - `base_policy.run_dir` 값을 **학습된 Base Policy의 출력 디렉토리 경로**로 변경해주세요.

### 1단계: 잠재 공간 학습 및 데모 캐싱 (Phase 1)
Front-view 이미지의 DINOv2 임베딩을 추출하고, MLPE2C 모델을 통해 16차원 잠재 공간을 학습합니다. 그 후 오프라인 데모의 잠재 궤적을 캐싱합니다.

```bash
cd /home/ysl2683/gym-PlanarPeg
python -m residual_rl.scripts.train_latent_encoder
```
- 실행이 완료되면 `residual_rl/outputs/latent_encoder-YYYYMMDD_HHMMSS/` 폴더가 생성되며, 그 안에 `latent_cache.pt`가 저장됩니다.
- 이 과정은 실시간으로 **WandB**에 로깅되며 (KL, Recon, Trans Loss 등) CPU/GPU 환경을 자동으로 감지하여 실행됩니다.

### 2단계: 캐시 경로 업데이트 및 Residual RL 파인튜닝 (Phase 2)
**⚠️ 중요**: 2단계를 실행하기 전, 반드시 `configs/residual_sac.yaml` 파일을 열고 `latent_cache_path`를 1단계에서 생성된 실제 폴더 경로로 변경해야 합니다.

```yaml
# configs/residual_sac.yaml 예시
latent_cache_path: "residual_rl/outputs/latent_encoder-20260626_205800/latent_cache.pt"
```

그 후 아래 명령어로 온라인 파인튜닝을 시작합니다:

```bash
cd /home/ysl2683/gym-PlanarPeg
python -m residual_rl.scripts.train_residual_sac \
    --config residual_rl/configs/residual_sac.yaml \
    --base_policy_run_dir <IL_policy_결과_디렉토리_경로>
```
- 실행 시 `residual_rl/outputs/residual_sac-YYYYMMDD_HHMMSS/` 형태의 단일 폴더에 WandB 로그, 네트워크 체크포인트, 설정 백업 파일 등이 모두 일괄 정리되어 저장됩니다.

#### 📌 주요 훈련 로직 (Phase 2)
1. **Critic Warmup**: 초기 10,000 스텝 동안은 Actor를 업데이트하지 않고 무작위 Residual Action으로 Critic만 학습하여 Q-value를 안정화합니다.
2. **Dense Reward**: 매 스텝 환경 보상(Sparse) 외에, 현재 시야가 캐싱된 데모 궤적 중 얼마나 목표에 가깝고 유사한지에 따라 추가 보상이 부여됩니다. (PyTorch 2.6+ 호환 가능)
3. **Symmetric Replay Buffer**: 배치 샘플링 시 항상 오프라인 시연 데이터(50%)와 온라인 수집 데이터(50%)를 섞어서 학습하여 Base Policy가 잘 하던 영역을 잊어버리는 것을 방지합니다.
4. **Action Bounding**: `action_scale` (기본값 0.15)을 통해 Residual Action의 크기를 제한하여 안전한 탐색을 보장합니다.
