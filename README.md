# gym-PlanarPeg

MuJoCo와 Gymnasium을 기반으로 구축된 평면 쐐기 삽입(Planar Peg Insertion)을 위한 2.5D 물리 시뮬레이션 환경입니다. 본 환경은 분포 외(OOD) 환경에서 분포 내(ID) 환경으로의 일반화 성능 검증 및 텔레오퍼레이션(원격 조종)을 통한 고품질 사람 시연 데이터 수집을 목적으로 설계되었습니다.

## 개요
본 환경의 태스크는 가상의 모캡(Mocap) 타겟에 제약 조건으로 묶인 3-DoF 직사각형 peg(에이전트)를 조종하여, 격자형 미로 내 좁은 장애물 틈새를 통과한 뒤 반대편에 위치한 C자형 골 박스(Goal Box)에 도킹하는 것입니다.

시뮬레이션은 MuJoCo 3D 물리 엔진을 기반으로 실행되지만, 에이전트의 물리적 움직임은 2D 평면($X, Y, \theta$) 상으로 완벽히 구속됩니다.

---

## 미로 레이아웃 및 구성
미로 지도는 2D 텍스트 격자 형태로 표현됩니다. 격자 인코딩에는 5가지 핵심 기호가 사용됩니다:
* `W` 또는 `#`: **외곽 경계 벽** (회색)
* `O`: **내부 장애물 벽** (파란색)
* `S` 또는 `P`: **에이전트 스폰 위치** (시작 위치)
* `C` 또는 `G`: **C자형 골 박스** (녹색, 안정적인 도킹을 위해 1.5배 두꺼워진 외벽 적용)
* `.`: **빈 공간**

---

## 액션 공간 (Action Space)
액션 공간은 `Box([-x_limit, -y_limit, -pi], [x_limit, y_limit, pi], (3,), float32)`입니다.
각 원소는 제어 타겟의 **순수 절대 물리 좌표계(m, rad)** 수치를 의미합니다. 

| 번호 | 액션 | 단위 | 설명 |
| :--- | :--- | :---: | :--- |
| **0** | 절대 X 좌표 | 미터 (m) | 월드 좌표계 기준 절대 X축 목표 위치 |
| **1** | 절대 Y 좌표 | 미터 (m) | 월드 좌표계 기준 절대 Y축 목표 위치 |
| **2** | 절대 회전각 (Yaw) | 라디안 (rad) | Z축 기준 절대 목표 회전 각도 |

---

## 관측 공간 (Observation Space)
* `image_top`: 탑뷰 RGB 이미지 `(224, 224, 3), uint8`.
* `image_front`: 에이전트 전면(`0.04m` 앞) 부착 1인칭 RGB 이미지 `(224, 224, 3), uint8` (충돌 시 벽면 투시 방지 적용).
* `proprioception`: 에이전트의 실제 3-DoF 물리 상태 `[X, Y, Theta]` (액션 공간과 **100% 동일한 절대 물리 좌표계 스케일** 적용).

---

## 보상 체계 및 성공 조건 (Rewards)
* **도킹 성공 보상:** 골 박스 내부에 완벽히 안착했을 때 **`+1.0`** 보상 지급 후 에피소드 성공 종료(`terminated=True`).
  * 성공 판정 기준: 에이전트의 중심점($lx$)이 골 박스 로컬 좌표계 기준 `0.02m` 이상 깊숙이 진입할 것. (Goal 박스의 두께 변화와 무관하게 안쪽 진입 깊이만 검사)
* **타임 페널티:** 신속한 동작 유도를 위해 도킹 성공 전까지 매 step마다 **`-0.01`** 비용 차감.

---

## 사용 방법 (Usage)

### 1. 설치 (Installation)
```bash
git clone https://github.com/YSL2683/gym-PlanarPeg.git
cd gym-PlanarPeg
conda create -n planar_peg python=3.10 -y
conda activate planar_peg
pip install -e .
```

### 2. 테스트 조종 (Play Mode)
환경 로드 여부와 에이전트 조작감을 편하게 테스트하기 위한 스크립트입니다. 데이터는 저장되지 않습니다.
```bash
python scripts/teleop.py
```
* **단축키**: `W/S` (상/하), `A/D` (좌/우), `Q/E` (회전), `R` (현재 에피소드 초기화), `ESC` (종료)

### 3. 데이터 기록 (Record Data)
실제 오프라인 RL 및 Diffusion Policy 학습을 위한 Zarr 데이터셋 수집 스크립트입니다.
Task(상위 폴더) 단위로 분류하며, 동일한 dataset_name에 에피소드를 누적 저장(`resume` 기능)합니다.
```bash
python scripts/record.py --task ID_base --dataset_name my_dataset --num_episodes 50
```
* **명령어 옵션 (Options)**:
  * `--task` : 데이터가 저장될 상위 폴더 이름 (기본값: `ID_base`). 환경의 종류나 난이도를 구분할 때 사용합니다.
  * `--dataset_name` : 생성될 Zarr 데이터셋의 이름 (기본값: `dataset`). 확장자(`.zarr`)는 내부적으로 자동 추가됩니다.
  * `--num_episodes` : 수집을 완료할 총 성공 에피소드 개수 (기본값: `50`). 해당 목표치에 도달하면 스크립트가 자동 종료됩니다.

### 4. 수집 데이터 뷰어 (Data Viewer)
기록된 Zarr 데이터의 품질과 물리량 무결성을 검증하는 시각화 뷰어입니다. 
X, Y, Theta 축별로 3개의 서브플롯(Sub-plot)을 나누어 렌더링하며, 각각의 Raw 스케일 변화량을 가장 직관적으로 비교할 수 있습니다.
```bash
python scripts/data_viewer.py --dataset data/ID_base/my_dataset.zarr
```
* **명령어 옵션 (Options)**:
  * `--dataset` : 시각화할 `.zarr` 데이터셋 폴더의 상대적/절대적 경로를 지정합니다. (필수 입력)
* **단축키**: `SPACE` (재생 / 일시정지), `A` (이전 프레임 역재생), `D` (다음 프레임 재생), `Q` / `ESC` (뷰어 종료)

---

## 데이터 저장 형식 (Zarr Data Format)
데이터는 `data/<task_name>/<dataset_name>.zarr` 경로에 Zarr 포맷으로 안전하게 저장 및 Append 됩니다. 각 에피소드는 다음과 같은 필드를 포함합니다:

| 데이터셋 이름 | 차원 (Shape) | 타입 | 설명 |
| :--- | :--- | :---: | :--- |
| `image_top` | $(N, 224, 224, 3)$ | `uint8` | 탑뷰(Top-view) RGB 이미지 관측값 |
| `image_front` | $(N, 224, 224, 3)$ | `uint8` | 1인칭 전방뷰 RGB 이미지 관측값 |
| `proprioception` | $(N, 3)$ | `float32` | 3-DoF 물리 상태 `[X, Y, Theta]` (절대 미터/라디안) |
| `action` | $(N, 3)$ | `float32` | 목표 타겟 좌표 `[Target_X, Target_Y, Target_Theta]` (절대 미터/라디안) |
| `reward` | $(N,)$ | `float32` | 해당 스텝 발생 보상 (`+1.0` 성공, `-0.01` 페널티) |
| `terminated` | $(N,)$ | `bool` | 에피소드 성공 종료 여부 플래그 (`True`/`False`) |

* **메타데이터 검증(`attrs`):** Zarr 저장소의 루트 속성에 `task`와 `dataset_name`을 하드 코딩하여, 명령어 인자 실수로 인한 이종 데이터 덮어쓰기(Data Corruption)를 근본적으로 방지합니다.
