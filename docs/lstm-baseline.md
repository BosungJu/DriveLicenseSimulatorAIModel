# LSTM baseline

구간 종료 후 시계열에서 여러 개선점 라벨의 확률을 예측하는 CPU/CUDA baseline입니다. 단방향 LSTM(은닉 크기 64)과 선형 출력층, BCE 손실을 사용합니다. 공식 감점·합격 판정과 설명 생성은 기존 시스템의 책임입니다. 현재 실제 주행 데이터와 전문가 라벨이 없어 학습된 모델이나 실제 성능 결과는 제공하지 않습니다.

## 설치와 실행

Python 3.10 이상을 설치한 Windows 환경에서 프로젝트 루트에서 실행합니다. 프로젝트에는 기존 Python 의존성이 없어 PyTorch 하나를 추가했습니다. 아래 명령은 CPU wheel을 설치합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src/baseline.py train --data res/train/sections.jsonl --schema res/schema.json --output artifacts/lstm.pt
.\.venv\Scripts\python.exe src/baseline.py predict --data res/test/unlabeled.jsonl --checkpoint artifacts/lstm.pt
```

파일 경로는 사용자가 수집할 데이터의 예시이며 해당 파일은 아직 없습니다. 테스트는 임시 디렉터리에 합성 데이터를 생성합니다.

## 모델 클래스와 CUDA

`src/model.py`의 `SectionLSTM(nn.Module)`은 신경망 구조와 forward를 정의합니다. `src/baseline.py`의 `LSTMBaseline`은 장치 선택, 학습, 정규화, 평가, 추론 및 checkpoint를 관리합니다. 기존 train/predict 명령도 이 클래스를 사용합니다.

설정 파일의 `device`는 auto/cpu/cuda/cuda:0 등입니다. `auto`는 CUDA가 가능하면 첫 GPU를 선택하고 그렇지 않으면 CPU를 사용합니다. 명시적인 cuda 요청은 사용할 수 없으면 오류입니다. 지정 GPU 인덱스도 검사합니다.

```powershell
# CPU wheel을 사용 중이라면 CUDA wheel로 교체; 환경에 맞는 공식 설치 명령 확인
.\.venv\Scripts\python.exe -m pip install --force-reinstall -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe src/baseline.py train --config res/train-config.example.json --device cuda:0
.\.venv\Scripts\python.exe src/baseline.py predict --data res/test/unlabeled.jsonl --checkpoint artifacts/lstm-augmented.pt --device cuda:0
```

CUDA 12.8 wheel은 설치 예시입니다. 호환 GPU/드라이버가 필요하며 설치 가능 버전은 인덱스에 따라 다릅니다. 공식 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)를 기준으로 선택하세요. 현재 프로젝트 검증에는 RTX 4060 Ti와 이 인덱스를 사용합니다.

클래스를 Python에서 직접 사용하려면 프로젝트 루트에서 다음처럼 호출합니다. fit은 설정에 따라 새 모델을 학습하고 best checkpoint를 output에 저장합니다. 설정은 다음 fit부터 적용됩니다.

```python
from src.baseline import LSTMBaseline, parse_args

settings = parse_args(["train", "--config", "res/train-config.example.json"])
settings.learning_rate = 0.0005
settings.output = "artifacts/lstm-lr0005.pt"
baseline = LSTMBaseline(device=settings.device)
metrics = baseline.fit(settings)

loaded = LSTMBaseline.load(settings.output, device="cuda:0")
predictions = loaded.predict("res/test/unlabeled.jsonl")
loaded.save("artifacts/lstm-copy.pt")
```

`baseline.model`과 `baseline.device`는 읽기 전용 속성입니다. fit은 생성자의 장치를 사용하며 settings.device는 CLI에서 생성자를 구성할 때 사용합니다. checkpoint에는 실제 장치를 기록합니다. `evaluate(라벨있는_JSONL, batch_size=32)`는 전달한 파일 전체를 평가하며, 원하는 split의 파일을 지정하세요. 학습 또는 load 전에 평가/추론/저장을 시도하면 오류입니다. 기존 format_version 1 checkpoint를 계속 로드할 수 있습니다.

증대·정규화·padding과 길이 텐서는 CPU에서 준비하고, 입력·라벨·모델 연산은 선택한 장치에서 수행합니다. 저장 가중치와 정규화 통계는 CPU 텐서라 CUDA에서 학습한 모델도 CPU에서 읽을 수 있습니다. 출력에 실제 장치를 기록하고 checkpoint에 CUDA 버전을 저장합니다.

`deterministic` 기본값은 true입니다. `--no-deterministic` 또는 JSON false로 변경할 수 있습니다. CUDA 텐서 생성 전에 미설정된 `CUBLAS_WORKSPACE_CONFIG`에 `:4096:8`을 설정합니다. 외부에서 CUDA 연산을 먼저 수행하는 프로그램은 시작 전에 이 환경변수를 설정하세요. 학습 종료/실패 시 PyTorch 결정론 및 cuDNN 설정은 이전 값으로 복원합니다. CPU/CUDA 간 비트 단위 동일 결과는 보장하지 않습니다. [PyTorch 재현성 안내](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## 조절 가능한 학습 설정과 데이터 증대

`res/train-config.example.json`을 복사해 데이터 경로, 모델 저장 경로와 다음 변수를 수정하세요. 예시 스키마는 `res/schema.example.json`에 있습니다. 데이터 수집 스키마에 맞춰 feature와 label 이름을 변경해야 합니다.

```powershell
.\.venv\Scripts\python.exe src/baseline.py train --config res/train-config.example.json
# JSON 설정에서 learning rate와 저장 경로만 덮어쓰는 실험
.\.venv\Scripts\python.exe src/baseline.py train --config res/train-config.example.json --learning-rate 0.0005 --output artifacts/lstm-lr0005.pt
```

| 변수 | 의미 |
| --- | --- |
| learning_rate | Adam learning rate (`--learning-rate`) |
| device / deterministic | 실행 장치 및 결정론 사용 여부 (`--device`, `--deterministic` / `--no-deterministic`) |
| epochs / batch_size / hidden_size | epoch 수, batch 크기, LSTM 은닉 크기 |
| num_layers | 쌓는 LSTM 층 수 (`--num-layers`); 기본값 1, 양의 정수 |
| seed | 모델 초기화, 섞기, 증대의 재현성 seed |
| augmentation_seed | 증대용 seed (`--augmentation-seed`); 생략 시 seed + 1,000,003을 2^63으로 나눈 나머지 |
| output | 모델마다 지정하는 checkpoint 저장 경로; 같은 경로는 덮어씀 |
| augmentation.probability | 구간을 증대할 확률; 0은 증대 없음 |
| features.<이름>.bias_max | 구간 전체에서 일정한 bias의 최대 절댓값, 원본 feature 단위 |
| features.<이름>.noise_max | 매 시점 노이즈 후보의 최대 절댓값, 원본 feature 단위 |
| features.<이름>.smoothing | 0 초과 1 이하; 작을수록 노이즈가 완만하게 변함 |
| features.<이름>.min / max | feature의 물리적 허용 범위 |
| features.<이름>.protected_values | 넘거나 이탈하면 안 되는 판정 경계값 목록 |
| features.<이름>.fixed_values | 선택; 원본이 이 값과 정확히 같은 프레임은 변형하지 않음(정지 0, 페달 0/1, 최대 조향 ±1) |
| features.<이름>.on_violation | 선택; `reject_feature`(기본) 또는 `keep_frame`. 범위 이탈·경계 통과 시 구간의 feature 증대 전체를 취소할지, 위반 프레임만 원본으로 둘지 |
| augmentation.augmented_copies | 선택; 지정하면 epoch마다 train 구간 하나를 원본 1개 + 증대본 N개로 학습 |

설정 파일의 data/schema/output 상대 경로는 **설정 파일 폴더** 기준입니다. 명령줄에서 지정하는 상대 경로는 현재 작업 폴더 기준이며, 명령줄 값이 JSON 값보다 우선합니다. JSON 변수 이름은 위처럼 snake_case입니다. 알 수 없는 설정 키와 잘못된 강도/학습 인자는 오류로 처리합니다. 설정 변경은 다음 학습 실행에 적용됩니다. 각 checkpoint에 실제 하이퍼파라미터와 증대 설정을 저장합니다.

예를 들어 `"hidden_size": 64, "num_layers": 2`는 은닉 크기 64인 LSTM을 2층 쌓습니다. 명령줄에서는 `train --config res/train-config.example.json --num-layers 2 --output artifacts/lstm-2layers.pt`로 변경할 수 있습니다. Python에서는 `SectionLSTM(feature_count=6, label_count=2, hidden_size=64, num_layers=2)` 또는 fit 전 `settings.num_layers = 2`를 사용합니다. checkpoint에 층 수를 저장하고 load 시 같은 구조를 복원합니다. num_layers가 없는 기존 checkpoint와 설정은 1층으로 처리합니다.

증대는 train batch를 만들 때 원본 단위에서 새로 생성하고 이후 원본 train 통계로 정규화합니다. 파일이나 라벨은 수정하지 않습니다. val/test 평가와 추론은 원본을 사용합니다. 증대 표본을 파일로 여러 배 복제하지 않으며, epoch마다 다른 변형을 제공합니다. 증대 난수는 별도 generator를 사용합니다.

feature별 bias는 `[-bias_max, +bias_max]` 균등분포로 구간당 한 번 뽑습니다. 노이즈 후보도 대칭 균등분포이며 `noise[t] = (1 - smoothing) * noise[t-1] + smoothing * candidate[t]`로 평활화합니다. 첫 노이즈는 첫 후보 값입니다. 따라서 총 변동의 절댓값은 bias_max + noise_max 이내입니다. `smoothing=1`은 독립 노이즈입니다. 경계 보호/범위 검사로 거절된 증대를 제외하면 대칭 분포이며, 거절 때문에 최종 적용 분포는 비대칭이 될 수 있습니다.

증대할 항목은 스키마 `continuous_features`에 명시된 연속형 feature만 가능합니다. 이진 제동·명령·구간 ID는 이 목록에 넣지 마세요. 설정에 없는 feature는 그대로 유지됩니다. 후보가 범위를 벗어나거나 보호 경계의 아래/같음/위 관계를 바꾸면 기본값(`reject_feature`)에서는 **그 구간의 해당 feature 증대를 취소**합니다. `on_violation: "keep_frame"`이면 위반한 프레임만 원본 값으로 두고 나머지 프레임은 증대합니다. 두 방식 모두 **기록된 프레임**에서 설정한 범위와 보호 경계의 관계만 유지합니다. Unity 판정은 `Update` 주기와 구간 통과 순간의 값을 쓰고 텔레메트리는 `FixedUpdate`로 기록하므로, 프레임 사이 값이나 보호 경계로 지정하지 않은 판정 기준까지 라벨 보존을 보장하지는 않습니다. `keep_frame`은 원본으로 돌아간 프레임 앞뒤에 노이즈 크기만큼의 불연속이 생겨, 속도처럼 매끄러운 신호에서는 물리적으로 불가능한 급변(예: 0.02초에 2 km/h 이상)이 생길 수 있습니다. 원본이 설정 범위를 벗어나면 오류입니다. 다른 feature는 별도로 처리합니다.

예시 설정은 speed_kmh에 bias 최대 0.05 km/h, noise 최대 0.02 km/h를 적용하고 0/1/20 km/h를 보호합니다. 이 값들은 효과가 입증된 값이 아니라 조정용 시작 예시입니다. 실제 판정 설정에 맞게 경계를 바꾸세요. 조향도 `features`에 같은 구조로 추가할 수 있지만, 조향 안정성처럼 변화율에 의존하는 라벨은 경계값 검사만으로 보존을 보장할 수 없습니다. 모든 라벨의 의미가 유지되는 feature/강도만 허용해야 합니다. 이 증대는 작은 관측 변동에 대한 강건성을 실험하는 용도이며 물리적으로 다른 주행 생성은 Unity 재시뮬레이션이 필요합니다.

### 원본 + N개 증대본 (`augmented_copies`)

`augmented_copies`가 없으면 위처럼 batch마다 구간을 probability 확률로 변형본 하나로 바꿉니다. `augmented_copies: 4`를 지정하면 epoch마다 train 구간마다 원본 1개와 새로 뽑은 증대본 4개, 즉 5개 표본을 섞어 학습합니다. 증대본은 파일로 저장하지 않고 epoch마다 새로 만들며, 각 증대본도 probability 확률로 변형되므로 보통 1.0으로 둡니다. 한 epoch의 학습 step이 (1+N)배가 되므로 epoch 수 비교에 주의하세요. val/test는 원본만 사용합니다.

`res/train-config.bias-copies.json`은 연속형 feature 6개(speed_kmh, accel_mps2, yaw_rate_dps, steering_input, throttle_input, brake_input)에 원본+4개를 적용한 시작 예시입니다. 구간 전체에 일정한 bias는 운전자 입력(steering_input, throttle_input, brake_input)에만 주고, 차량 상태(speed_kmh, accel_mps2, yaw_rate_dps)는 `bias_max: 0`으로 시점별 노이즈만 줍니다. accel_mps2 범위 ±1000은 충돌 순간 같은 급변 값(실측 최대 602 m/s², 2026-11-25 세션)을 포함하기 위한 값입니다. 원본이 범위를 벗어나면 학습이 중단되고, 교차 검증은 val 세션도 train fold로 쓰므로 모든 split의 값을 포함해야 합니다. 페달·조향·속도는 대부분의 프레임이 정확히 0이어서 `fixed_values` 없이 대칭 노이즈를 주면 범위 이탈로 거의 모든 구간의 증대가 취소됩니다. 그래서 정지·미조작·포화 값은 `fixed_values`로 그대로 두고 조작 중인 프레임만 변형합니다. 모든 프레임이 고정값인 구간은 unchanged로 집계됩니다. 강도는 효과가 검증된 값이 아니며, feature 간 물리적 일관성(예: 속도와 가속도)은 보장하지 않습니다.

이 설정의 변화량은 train feature 표준편차의 0.07~1.4%로, 프레임 간 변화보다도 작아 증대본이 원본과 거의 같습니다. 또 속도는 0 근처 프레임 때문에 약 절반의 구간에서 증대가 취소됩니다. `res/train-config.strong-copies.json`은 같은 보호 경계(속도 0/1/20 km/h, Unity 정지·제한 속도 기준)를 유지하면서 `keep_frame`으로 바꾸고 강도를 표준편차의 약 1~5%로 키운 설정입니다(취소 0%, 경계 근처 속도 프레임 약 5%만 원본 유지). 조향에는 Unity 깜빡이 자동 취소 기준(`VehicleSystems`의 준비 0.5, 취소 0.05; 스크립트 기본값)을 ±0.5, ±0.05 보호 경계로 넣어, 증대가 깜빡이 상태와 모순되는 조향을 만들지 않게 합니다. 아래 학습 결과는 이 조향 경계를 넣기 전 설정으로 얻은 값입니다. 이 설정으로 한 번 학습한 결과(seed 42)는 val BCE 0.147, test BCE 0.452로, 기존 설정(0.155, 0.284)보다 나아지지 않았습니다. 양성 라벨 대부분(turn_signal_missing, vehicle_device_error)이 증대하지 않는 이진 feature에 좌우되므로 연속형 feature 증대만으로는 과적합을 줄이기 어렵습니다. val 양성이 3개뿐이라 단일 seed 비교는 신뢰도가 낮습니다. `res/train-config.strong-nobias-copies.json`은 같은 설정에서 운전자 입력의 bias만 0으로 둔 비교용입니다. 같은 seed에서 val BCE 0.146, test BCE 0.478로 bias가 있는 설정과 차이가 거의 없었습니다.

### 증대 데이터 파일로 남기기 (`augment`)

`augment` 명령은 학습과 같은 설정 파일로 train 구간마다 원본 1개와 `augmented_copies`개의 증대본을 고정 seed(`augmentation_seed`)로 만들어 JSONL로 저장합니다. val/test는 원본 그대로 넣어 파일 하나로 바로 학습할 수 있습니다. train 레코드에는 `augmentation_copy`(0 = 원본)가 붙고, 증대된 연속형 feature의 바뀐 값만 다시 쓰며 나머지 값은 원본 텍스트를 유지합니다. 같은 설정·seed·원본이면 같은 파일이 나옵니다. 옆에 `.meta.json`으로 원본 경로·SHA-256, 증대 설정, seed, split별 레코드 수, 적용 통계를 기록합니다.

```powershell
.\.venv\Scripts\python.exe src/baseline.py augment --config res/train-config.bias-copies.json --output res/data/augmented_sections.jsonl
# 저장된 증대 데이터로 학습할 때는 augmentation 없이(이중 증대 방지)
.\.venv\Scripts\python.exe src/baseline.py train --data res/data/augmented_sections.jsonl --schema res/schema.example.json --output artifacts/lstm-from-file.pt
```

학습 중 증대(`augmented_copies`)는 epoch마다 새 변형을 만들고, 파일 증대는 한 번 뽑은 변형을 고정해 재사용합니다. 학습 중 증대와 같은 seed라도 뽑는 순서가 달라 표본은 같지 않습니다. 파일은 원본의 약 (1+N)배 크기입니다(현재 데이터 기준 약 127MB). GitHub는 100MB가 넘는 파일을 거부하므로 저장소에 넣으려면 Git LFS를 쓰거나 커밋에서 제외하세요.

증대 없음(probability 0), noise만(bias_max 0), bias+noise를 별도 output과 동일 원본 split으로 비교하세요. test 성능으로 설정을 반복 선택하지 말고 val에서 선택한 뒤 test를 평가하세요. `augmentation`을 생략하거나 null로 설정하면 기존 학습 경로를 그대로 사용합니다. 증대만 별도 JSON 파일로 분리하려면 `--augmentation 파일경로`로 덮어쓸 수도 있습니다.

학습 종료 출력과 checkpoint의 `augmentation_stats`에서 feature별 attempted/applied/unchanged/skipped/rejected_bounds/rejected_threshold/kept_frames 횟수를 확인할 수 있습니다. attempted는 확률 선택 후 후보를 만든 횟수, applied는 실제 값이 달라진 횟수입니다. kept_frames만 구간이 아니라 프레임 단위이며, `keep_frame`에서 원본으로 둔 프레임 수입니다. 시도 대비 적용률이 10%보다 낮으면 경고합니다. 정지 프레임처럼 보호 경계와 정확히 같은 값이 있으면 해당 구간의 feature 증대가 모두 취소될 수 있으므로, 이 통계로 강도 조정 효과를 확인하세요. 원본 train 범위는 학습 시작 전에 전체 검사합니다.

## 데이터 계약 v1

`schema.json` 예시:

```json
{"schema_version":1,"sample_rate_hz":50,"feature_names":["speed_kmh","accel_mps2","yaw_rate_dps","steering_input","throttle_input","brake_input","gear_park","gear_reverse","gear_neutral","gear_drive","parking_brake_on","engine_on","left_signal_on","right_signal_on","hazard_on","headlight_on","high_beam_on","seat_belt_on","horn_on","section_start","section_acceleration","section_emergency","section_hill_stop","section_signal","section_left_turn","section_right_turn","section_parking","section_finish_approach"],"label_names":["stop_failure","hill_rollback","over_speed","insufficient_acceleration","section_time_over","turn_signal_missing","hazard_light_error","vehicle_device_error"]}
```

JSONL의 각 행은 구간 하나입니다. 학습 행 예시:

```json
{"schema_version":1,"sample_rate_hz":50,"run_id":"anonymous-run-001","section_id":2,"split":"train","feature_names":["speed_kmh","accel_mps2","yaw_rate_dps","steering_input","throttle_input","brake_input","gear_park","gear_reverse","gear_neutral","gear_drive","parking_brake_on","engine_on","left_signal_on","right_signal_on","hazard_on","headlight_on","high_beam_on","seat_belt_on","horn_on","section_start","section_acceleration","section_emergency","section_hill_stop","section_signal","section_left_turn","section_right_turn","section_parking","section_finish_approach"],"features":[[4.0,-0.5,3.0,-0.25,0.5,1.0,0,0,0,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1],[0.0,0.0,0.0,0.0,0.0,0.75,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1]],"label_names":["stop_failure","hill_rollback","over_speed","insufficient_acceleration","section_time_over","turn_signal_missing","hazard_light_error","vehicle_device_error"],"labels":[0,0,0,0,0,1,0,0]}
```

- 관측은 고정 Hz로 수집하거나 사전에 리샘플합니다. 시간 간격은 메타데이터만으로 검증할 수 없으므로 수집 계층이 보장해야 합니다. 단위와 실제 적용 입력의 범위를 feature 이름 및 수집 문서로 고정하세요. 50 Hz는 예시입니다.
- `features`는 패딩 없는 T × F 유한 수치 배열입니다. 빈 구간과 결측/NaN/Inf는 거부합니다. feature 이름과 순서는 schema 및 checkpoint와 정확히 일치해야 합니다.
- 라벨 이름은 구성 가능합니다. 기본 라벨은 아래 [감점 기반 라벨](#감점-기반-라벨)의 공식 감점 범주입니다. `1`은 확인된 양성, `0`은 확인된 음성입니다. 구간에서 판정됐지만 결과를 알 수 없는 라벨은 0으로 쓰고 선택 필드 `unjudged_labels`(라벨 이름 목록)에 넣으면 그 레코드에서만 손실·평가에서 제외됩니다. 이 목록의 라벨이 1이면 오류입니다.
- `split`은 train/val/test 중 하나입니다. train과 val은 필수, test는 선택입니다. 동일 run_id의 구간들이 split을 넘으면 오류입니다. 새 운전자에 대한 평가에는 익명 driver_id도 지정하세요. 지정된 driver_id의 split 일관성도 검증합니다. driver_id를 생략하면 운전자 단위 누수를 감지할 수 없습니다.
- 맵·물리·판정 설정 버전을 통일한 데이터로 시작하세요. 감점 결과나 라벨을 직접 드러내는 값을 feature로 넣으면 평가가 왜곡될 수 있습니다.
- 추론 행에는 split/label_names/labels가 필요 없습니다. run_id, feature_names, features, schema_version, sample_rate_hz는 필수입니다.

### Unity 자동차 입력 기록과 feature 매핑

`DriverLicenseSimulator`의 `CarController`(Input System 페달·핸들·기어·주차 브레이크)와 `VehicleSystems`(시동·등화·안전벨트·경음기) 상태를 읽기 전용 속성으로 노출합니다. `ReportManager`는 실행 순서 100으로 기본 순서의 차량 입력 처리 뒤에 기록하며, 같은 FixedUpdate 시점의 입력 표본이 있는 활성 컨트롤러만 사용합니다. 컨트롤러가 없거나 입력이 오래됐으면 물리 상태 기록은 유지하고 조작 열(조향~주차 브레이크)을 빈칸으로 내보냅니다. `CarController.systems`에 활성 `VehicleSystems`가 없으면 차량 상태 열을 빈칸으로 내보냅니다. 새 모델은 빈칸을 오류로 거부하며, 물리 feature만 사용하는 기존 모델은 계속 사용할 수 있습니다. Rigidbody까지 없으면 기록 시각을 진행하고 가속도 차분 이력을 초기화해 복구 첫 표본의 가속도를 0으로 기록합니다. Inspector에서 실행 순서를 별도 지정했다면 CarController가 ReportManager보다 앞서도록 확인하세요.

| 모델 feature 순서 | CSV / 원본 의미 |
| --- | --- |
| speed_kmh | Rigidbody 전체 속력, km/h, 진행 방향은 기어 feature로 구분 |
| accel_mps2 | 차량 forward에 투영한 속도의 시간 차분, m/s²; 세션 첫 표본은 0 |
| yaw_rate_dps | 월드 Y축 각속도, 도/초 |
| steering_input | CarMoveInput.Steering, -1~1의 조향 명령 |
| throttle_input | CarMoveInput.Throttle, 0~1 가속 페달; 구동 방향은 기어가 결정 |
| brake_input | CarMoveInput.Brake, 0~1 브레이크 페달 |
| gear_park / gear_reverse / gear_neutral / gear_drive | CarController.CurrentGear 원-핫, 매 표본 정확히 하나가 1 |
| parking_brake_on | CarController.ParkingBrakeOn, 0/1 |
| engine_on | VehicleSystems.EngineOn, 0/1 |
| left_signal_on / right_signal_on | VehicleSystems.LeftSignalOn / RightSignalOn, 0/1; 깜빡임 위상이 아니라 켜짐 상태 |
| hazard_on | VehicleSystems.HazardOn, 0/1; 시동이 꺼져도 켤 수 있음 |
| headlight_on / high_beam_on | VehicleSystems.HeadlightOn / HighBeamOn, 0/1 |
| seat_belt_on | VehicleSystems.SeatBeltOn, 0/1 |
| horn_on | VehicleSystems.HornOn, 0/1; 누르고 있는 동안 1 |
| section_start / section_acceleration / section_emergency / section_hill_stop / section_signal / section_left_turn / section_right_turn / section_parking | 현재 구간에 적용되는 채점 규칙(Unity `DrivingSectionTraits`), 0/1; 동시 명령 구간은 여러 개가 1 |
| section_finish_approach | 다음 경로 지점이 종료 지점이면 1; 종료 방향지시등 규칙이 이 구간을 판정함 |

steering_input은 부드럽게 제한된 실제 휠 조향각이 아니며, throttle_input은 실제 모터 토크가 아닙니다. 시동이 꺼져 있거나 P/N 기어여도 사용자의 가속 입력은 기록될 수 있습니다. 기어·주차 브레이크·차량 상태 토글은 Update에서 바뀌므로 표본에는 해당 물리 스텝 직전의 최신 상태가 기록됩니다. 상태는 차량 제어 코드 이후, 다음 물리 시뮬레이션 이전에 기록하므로 speed_kmh에는 CarController.Brake의 페달 비례 감속(끝까지 밟으면 `linearVelocity *= 0.98`)이 이미 반영됩니다. 와이퍼(`WiperController`)는 차량 컨트롤러와 연결된 참조가 없어 포함하지 않았습니다. 이진·원-핫 feature는 `continuous_features`에 넣지 마세요.

구간 feature는 명령 문자열이 아니라 채점기가 그 구간에 적용하는 규칙입니다. Unity 평가기(`DrivingTestScoreEvaluator`)가 판정기를 고를 때 쓰는 `DrivingSectionTraits`를 `ReportManager`도 그대로 사용하므로 두 기준이 어긋나지 않습니다. 예를 들어 section_hill_stop은 `DrivingScoringSettings.HillStopStep` 단계(기본 hill_1)에서만, section_acceleration은 종료 단계 fast_1을 제외하고, section_parking은 종료 단계 T_TEST_3을 제외하고 1입니다. 구간 진입과 함께 정해지는 값이라 결측이 없습니다. 이 9개 열이 추가되면서 이전 `training_sections.jsonl`(19 feature)은 새 스키마로 학습할 수 없으니 다시 수집하세요. 19-feature checkpoint는 열 이름으로 선택하므로 새 CSV로도 추론할 수 있습니다.

CSV 열 순서는 `time_s,point_number,heading_deg,pos_x,pos_z` 뒤에 위 feature 순서입니다. Unity의 `DrivingFeatureColumns`가 CSV와 학습 JSONL에 같은 열 정의·형식을 사용합니다. heading_deg/pos_x/pos_z는 기본 모델 feature에서 제외하고, time_s/point_number는 시간과 구간 메타데이터로 사용합니다. AI의 `session_predict.py`는 열 이름으로 checkpoint feature를 선택하며, 조향(-1~1)·페달(0~1) 범위, 이진 값, 기어 원-핫(정확히 하나 활성)을 검사합니다. 입력 feature를 사용하는 모델에서는 time_s 간격도 검사해 누락 프레임을 조용히 압축하지 않습니다(기록의 3자리 반올림 오차 허용). 아날로그 브레이크 이전 CSV(`brake_applied`, 부호 있는 throttle)에 0이나 추정 입력을 채우지 않으므로 새 입력 feature 모델에서는 오류입니다. `brake_applied`를 feature로 쓰는 이전 checkpoint는 이전 로그 추론을 위해 throttle_input -1~1 범위를 유지합니다. 새 모델은 라벨을 준비한 새 기록으로 재학습해야 하며 기존 checkpoint의 스키마와 가중치를 자동으로 변경하지 않습니다. 예시 라벨 정의는 계속 전문가 검토가 필요합니다.

입력을 요구하는 모델에서 한 표본이라도 입력이 비어 있으면 세션 전체 추론을 실패시키며 부분 결과를 만들지 않습니다. 물리 feature만 사용하는 기존 모델은 이전 호환 경로를 유지하므로 time_s 간격 검사는 적용하지 않습니다.

### Unity 학습 데이터 수집

`TrainingDataCollector`(Unity `Assets/Scripts/AI_Report`)를 `ReportManager`와 같은 오브젝트에 붙이면, 코스 완료 후 세션 폴더에 `training_sections.jsonl`을 함께 씁니다. 각 행은 위 데이터 계약 v1의 라벨 없는 구간 레코드이며 run_id(세션 폴더 이름 + 무작위 8자리), 선택적 익명 driver_id, section_id, start_time_s를 포함합니다. 값은 driving_log.csv와 같은 소수 3자리 반올림입니다. 입력·상태 결측, NaN, 고정 간격(0.5~1.5스텝)을 벗어난 시각이 있는 구간은 레코드를 만들지 않고 건너뜁니다. 학습 데이터는 아래 `deduction_labels.py`로 세션 파일들을 합치면서 split/label_names/labels를 추가해 만듭니다. 코스를 완료하지 않고 초기화한 주행은 저장되지 않습니다.

#### 구간 반복 연습 모드 (구간별 수집)

특정 구간 데이터만 집중해서 모으려면 Unity `DrivingTestSession`의 **Section Practice**를 켭니다. 연습 시작 지점(CAD 점 번호)에 차량을 놓고 지정한 구간 수만큼 주행하면, 다음 지점에 닿는 순간 그 시도를 코스 완료처럼 내보낸 뒤(세션 폴더에 driving_log.csv, section_results.json, training_sections.jsonl) 지연 시간 후 같은 지점에서 다시 시작합니다. 따라서 아래 `deduction_labels.py` 명령을 그대로 사용합니다.

| Inspector 항목 | 의미 |
| --- | --- |
| Is Section Practice Enabled | 켜면 R 키 재시작과 자동 재시작이 전체 코스 대신 연습 범위로 동작 |
| Practice Start Point Number | 시도를 시작할 CAD 점 번호; 이 점의 구간을 먼저 주행 |
| Practice Section Count | 시도당 연속 구간 수(기본 1); 마지막(종료) 지점을 넘지 않음 |
| Practice Restart Delay Seconds | 저장 후 다음 시도까지 대기 시간(기본 2초) |

- 연습 종료 지점의 구간은 진입하지 않고 끝내므로, 주행하지 않은 구간의 판정기가 감점(예: 가속 미달, 경사로 미정지)을 만들지 않습니다. 종료 지점 통과 속도는 마지막 구간의 출구 속도로 판정합니다. 범위가 종료 지점까지 이어지면 종료 방향지시등도 평소처럼 판정합니다.
- 차량은 정지 상태로 시작 지점에 놓입니다. 목표 구간에 주행 속도로 진입한 데이터를 원하면 한 지점 앞에서 시작하고 구간 수를 2로 두세요(앞 구간 데이터도 함께 저장되며 정상적인 학습 데이터입니다).
- 직각주차(T_TEST_n)처럼 여러 지점에 걸친 판정은 범위가 전체 단계를 덮어야 합니다. 일부만 연습하면 주차 시간·주차 브레이크 판정이 잘려 보고서 점수가 실제와 달라집니다(두 사유는 라벨에서 제외되어 학습 라벨에는 영향 없음).
- 시동·안전벨트·기어 상태는 재시작해도 유지되므로 첫 시도에서만 준비하면 됩니다. 시작 절차 데이터가 필요하면 출발 지점(start)부터 연습하세요.
- section_results.json의 `isSectionPractice`가 true이면 코스 일부만 주행한 기록이라 score/isPassed는 코스 합격 판정이 아닙니다. `report.py`는 이 값을 `is_section_practice`로 넘겨 보고서에서 합격·불합격을 안내하지 않게 합니다. 완료하지 않고 R로 재시작한 시도는 저장되지 않습니다.
- 같은 운전자의 반복 시도가 train/val에 섞이지 않도록 `TrainingDataCollector`의 익명 driver_id를 지정하세요. 지정하지 않으면 시도마다 다른 그룹으로 나뉘어 운전자 단위 누수가 생깁니다.

Unity 수동 확인: 코스 주행 중 조향 양방향, 가속·브레이크 페달 부분 입력, P/R/N/D 변속, 주차 브레이크, 시동·방향지시등·비상등·전조등·상향등·안전벨트·경음기를 조작하고 코스 종료 후 AIReport 세션의 driving_log.csv와 training_sections.jsonl을 확인하세요. EditMode Test Runner의 `DrivingReportCsvTests`, `DrivingTrainingDatasetTests`가 열 순서, 원-핫 기어, 아날로그 페달, 결측 빈칸, 문화권에 무관한 숫자 형식과 구간 건너뛰기를 검증합니다.

### 감점 기반 라벨

`src/deduction_labels.py`는 각 세션 폴더의 `training_sections.jsonl`에 같은 세션 `section_results.json`의 공식 감점을 구간(pointNumber) 단위로 붙입니다. `TrainingDataCollector` 없이 기록되어 `training_sections.jsonl`이 없는 세션은 `driving_log.csv`에서 같은 형식의 구간 레코드를 만듭니다(run_id는 세션 폴더 이름, feature는 `time_s,point_number,heading_deg,pos_x,pos_z` 뒤의 열, sample_rate_hz는 Unity처럼 소수 3자리 반올림). 이 경우 driver_id가 없어 run 단위로 split하며, Unity와 달리 잘못된 표본이 있는 구간을 건너뛰지 않고 세션 전체를 오류로 처리합니다. 라벨 1은 "시뮬레이터 채점기가 이 구간에서 해당 범주로 감점 또는 실격 처리했다"는 뜻이고, 0은 그런 기록이 없다는 뜻입니다. 전문가 라벨이 아니라 공식 채점 결과를 모방하는 약지도 라벨입니다. 구간 종류는 section_* feature로 함께 입력되므로, 모델은 어떤 채점 규칙이 적용되는 구간인지 알고 라벨을 학습합니다.

```powershell
# AIReport 폴더(세션 폴더들의 부모) 또는 개별 세션 폴더를 여러 개 지정
.\.venv\Scripts\python.exe src/deduction_labels.py "$env:USERPROFILE\AppData\LocalLow\<회사>\<제품>\AIReport" --output res/train/sections.jsonl --val-ratio 0.2 --test-ratio 0.1 --seed 42
.\.venv\Scripts\python.exe src/baseline.py train --data res/train/sections.jsonl --schema res/schema.example.json --output artifacts/lstm-deduction.pt
```

| 라벨 | Unity `DrivingDeductionReason` |
| --- | --- |
| stop_failure | HillNotStopped, EmergencyStopLate |
| hill_rollback | HillRollback, HillRollbackOver |
| over_speed | OverSpeed, AccelerationEntryTooFast, AccelerationExitTooFast |
| insufficient_acceleration | AccelerationTooSlow |
| section_time_over | StartTimeOver, HillPassTimeOver |
| turn_signal_missing | TurnSignalMissing, StartSignalMissing, FinishSignalMissing, StartTurnSignalNotOperated |
| hazard_light_error | EmergencyHazardLate, EmergencyHazardNotOff, EmergencyHazardEarly |
| vehicle_device_error | SeatBeltNotFastened, SeatBeltNotWorn, EngineNotStarted, HeadlightNotOperated, EngineStalled, GearNotOperated |

#### OBS 2.1 패키지와 train 전용 세션 (`--train-only`)

OBS 2.1 패키지(`runs/<id>/label.json` + `runs/<id>/reports/`의 Unity 세션)는 패키지 폴더, `runs` 폴더, 개별 run 폴더 중 무엇을 지정해도 됩니다. run_id는 run 폴더 이름입니다. 각 run은 출발부터 목표 구간까지 자동 주행하므로 `label.json`의 `scope` [시작, 끝) 구간만 사용합니다. 끝 지점은 출구 경계일 뿐입니다. scope 앞 구간은 run마다 거의 같은 자동 주행이 반복되므로 쓰지 않습니다. 라벨은 run 전체의 `section_results.json`에서 만들고, 실격 마스크도 같은 규칙을 적용합니다.

`--train-only`로 지정한 세션은 항상 train에 들어가고 val/test 분할에서 빠집니다. 나머지 세션의 분할은 바뀌지 않습니다. OBS21처럼 한 자동 제어 출처에서만 나와 독립 평가 자료가 없는 데이터에 쓰세요. 같은 세션이나 run/driver 그룹이 양쪽에 있으면 오류입니다. 위치 인자(분할 세션)를 `--train-only`보다 먼저 쓰세요.

```powershell
$downloads = "$env:USERPROFILE\Downloads"
# 기존 데이터셋 구성 유지: 20261110_053702_177은 이전부터 제외
$local = Get-ChildItem "$env:USERPROFILE\AppData\LocalLow\DefaultCompany\DrivingTestSimulator\AIReport" -Directory |
    Where-Object Name -ne "20261110_053702_177" | ForEach-Object FullName
.\.venv\Scripts\python.exe src/deduction_labels.py "$downloads\AIReport" @local "$downloads\AIReport2_8개\AIReport" "$downloads\AIReport3" `
    --train-only "$downloads\Unity-OBS21-final-train-118-20261126" `
    --output res/train/sections.jsonl --val-ratio 0.2 --test-ratio 0.2 --seed 42
```

현재 `res/train/sections.jsonl`(2026-11-27)은 이 명령으로 만들었습니다. 사람 주행 45세션은 train 27 / val 9 / test 9로 나뉘고, OBS21 118 run(150구간)은 train에만 들어갑니다. 전체 627구간, 약 81MB입니다. 세션 수가 바뀌어 이전 26세션 데이터와 split이 다르므로, 위 학습·교차 검증 수치와 직접 비교하지 마세요. OBS21에서 target=1인데 모든 라벨이 0인 20개 run은 제외 사유(SignalViolation 8, SignalLongStop 4, ParkingLineTouch·ParkingBrakeNotApplied·ParkingTimeOver 각 2, WiperNotOperated 2)만 있는 경우입니다. train 전용 레코드에는 `train_only: true`가 붙습니다. `cross_validate.py`는 이 레코드를 fold 배정에서 빼고 모든 fold의 train으로만 사용합니다(결과 JSON의 `train_only_sections`). 따라서 fold val은 사람 주행 세션만으로 평가됩니다. `train_only` 레코드의 split이 train이 아니면 오류입니다.

GearNotOperated(N 기어 + 주차 브레이크)와 StartTurnSignalNotOperated(좌측 방향지시등)는 출발 구간의 무작위 기기조작 지시 판정이며, 각각 기어·주차 브레이크와 방향지시등 feature로 관측할 수 있습니다.

라벨에서 제외한 사유와 이유: LaneKeepingLineTouch·ParkingLineTouch(차선과 차량 위치가 feature에 없음), SignalViolation·SignalLongStop(신호등 상태가 feature에 없음), ParkingTimeOver·ParkingBrakeNotApplied(여러 T_TEST 구간에 걸친 판정), CourseTimeOver(코스 전체 시간), WiperNotOperated(와이퍼 미기록). 이 감점들도 보고서의 공식 감점과 능력 점수에는 그대로 반영됩니다.

귀속 규칙:

- 감점은 채점기가 기록한 pointNumber 구간의 라벨이 됩니다. 구간 종료 시점 판정도 끝나는 구간에 기록됩니다.
- FinishSignalMissing은 종료 지점 진입 시 기록되지만 직전 진입 구간을 판정한 결과이므로 직전 구간에 붙입니다. 종료 지점은 진입과 동시에 코스가 끝나 표본이 없습니다.
- Unity `DrivingScoreSheet`는 첫 실격만 기록하고, 이후에도 일반 감점은 계속 기록합니다. 그래서 첫 실격 구간과 그 이후 구간은 데이터셋에 **유지**하되, 실격 사유(`DISQUALIFYING_REASONS`: StartTimeOver, HillNotStopped, HillRollbackOver, HillPassTimeOver, SeatBeltNotWorn, SignalViolation)를 가진 라벨(stop_failure, hill_rollback, section_time_over, vehicle_device_error)의 **0만** `unjudged_labels`로 표시해 손실·평가에서 뺍니다. 기록된 1과 나머지 라벨(over_speed, insufficient_acceleration, turn_signal_missing, hazard_light_error)은 그대로 학습합니다. `tests/test_deduction_labels.py`가 Unity의 `Disqualify(DrivingDeductionReason.X)` 호출과 이 목록을 비교합니다.
- 출력 요약의 `masked_after_disqualification`은 판정 불가 라벨이 있는 구간 수입니다.
- split은 driver_id(없으면 run_id) 단위로 seed에 따라 재현 가능하게 나눕니다. 같은 run_id 안에서 driver_id가 섞여 있거나, 그룹 수가 train/val(/test)을 채우기에 부족하면 오류입니다.

### 감점 사유 동기화

Unity `DrivingScoreSheet.cs`의 `DrivingDeductionReason`을 기준으로 세 표를 맞춥니다. `tests/test_deduction_labels.py`가 enum을 직접 읽어 검사하므로, Unity에 사유를 추가하거나 이름을 바꾸면 테스트가 실패합니다. Unity 저장소가 이 프로젝트 옆 폴더가 아니면 `DRIVER_LICENSE_SIMULATOR_ROOT` 환경변수로 위치를 지정하세요. 없으면 해당 테스트는 건너뜁니다.

| 표 | 위치 | 규칙 |
| --- | --- | --- |
| 라벨 / 제외 사유 | `src/deduction_labels.py` `LABEL_REASONS`, `UNOBSERVABLE_REASONS` | 모든 사유가 정확히 한 곳 |
| 한국어 사유 이름 | `src/report.py` `DEDUCTION_REASON_NAMES` | enum 순서 그대로 |
| 능력 점수 분류 | `src/report.py` `CATEGORY_REASONS` | 모든 사유가 정확히 한 분류; 없으면 보고서 생성이 중단됨 |

능력 점수 분류는 조작(경사로 정지·밀림, 돌발 정지, 비상등·주차 브레이크·시동, 출발 전 기기조작), 도로 주행(차로·주차선 접촉, 목표속도 미달), 준법(속도·시간 규칙, 신호, 방향지시등, 주행 중 안전띠)입니다. 보고서 프롬프트의 AI 라벨 해석도 위 8개 라벨로 맞춰져 있습니다.

출력 요약의 `positives_by_split`로 라벨별 양성 수를, `unrecorded_positive_sections`로 Unity가 입력 결측 등으로 레코드를 만들지 않아 잃은 양성 구간 수를 확인하세요. 라벨 집합이 바뀌었으므로 기존 2-라벨 checkpoint는 재학습이 필요합니다. 증대 설정의 speed_kmh `protected_values`(0/1/20)는 Unity `DrivingScoringSettings`의 정지 판정 1 km/h, 지정속도·가속 목표 20 km/h와 맞춘 값이므로 판정 설정을 바꾸면 함께 바꾸세요.

### 구간별 라벨 마스크

채점기는 라벨마다 판정하는 구간이 정해져 있습니다(예: hill_rollback은 경사로 정지 구간에서만). 그 밖의 구간에서 라벨 0은 "정상"이 아니라 "판정 대상 아님"이므로, 스키마의 `label_sections`로 라벨별 판정 구간을 지정하면 해당 항목을 손실·평가·추론에서 제외합니다. 모델은 하나를 공유하고 구간 종류는 `section_*` feature로 입력합니다. 구간 유형별 별도 모델보다 데이터를 나누지 않아 적은 수집량에서 유리합니다.

| 키 | 의미 |
| --- | --- |
| section_features | 구간 종류를 나타내는 0/1 feature 목록; `continuous_features`에 넣을 수 없음 |
| label_sections | 모든 라벨 → `"all"`(모든 구간에서 판정) 또는 section_features 중 판정 구간 목록 |

구간의 section feature가 한 프레임이라도 1이면 그 종류로 봅니다. 예시 스키마의 매핑은 Unity 판정기 기준입니다: stop_failure(경사로 정지·돌발), hill_rollback(경사로 정지), insufficient_acceleration(가속), section_time_over(출발·경사로 정지), turn_signal_missing(출발·좌회전·우회전·종료 직전), hazard_light_error(돌발), over_speed·vehicle_device_error(all; 코스 전체 규칙 포함). 판정기를 바꾸면 이 매핑도 함께 바꾸세요.

- 학습 손실은 판정 항목의 평균 BCE입니다. `label_sections`가 없는 기존 스키마는 모든 항목을 판정하므로 이전 결과와 같습니다.
- 평가 출력의 `per_label`에 `judged_count`(판정된 구간 수)가 추가되고, `per_section`에 section_features 종류별 BCE·micro F1·라벨별 지표가 나옵니다. 여러 종류가 켜진 구간은 각 종류에 모두 집계됩니다. 판정 항목이 없으면 bce는 null입니다.
- 상수 기준선의 라벨 빈도도 판정 구간에서만 계산합니다. 어떤 train 구간에서도 판정되지 않는 라벨은 경고합니다.
- 추론의 `probabilities`에는 그 구간에서 판정되는 라벨만 들어갑니다. 스키마가 checkpoint에 저장되므로 추론 시 같은 마스크를 사용합니다. `label_sections`를 추가한 스키마는 재학습이 필요합니다.

## 학습과 결과 해석

`train`은 시작할 때 장치·train 구간 수·epoch당 표본/배치 수를, 매 epoch 끝에 train 평균 BCE, val BCE, 최저 val 갱신 표시(`*best`), 경과 시간과 남은 시간 추정(eta)을 stderr에 출력합니다. stdout에는 마지막 결과 JSON만 나오므로 결과를 파일로 받으면서도 진행을 볼 수 있습니다. eta는 지금까지의 평균 epoch 시간으로 계산합니다. Python에서 `LSTMBaseline.fit`을 직접 호출할 때는 `logging.basicConfig(level=logging.INFO)`로 `drive_lstm` 로그를 켜세요.

```powershell
# 터미널에서 진행 로그를 보면서 결과 JSON은 파일로 저장
.\.venv\Scripts\python.exe src/baseline.py train --config res/train-config.bias-copies.json > artifacts/train-result.json
# 다른 곳에서 실행 중인 학습의 stderr를 파일로 남겼다면 실시간으로 확인
Get-Content artifacts/train-progress.log -Wait
```

정규화는 train의 실제 프레임만 사용합니다. 표준편차가 1e-6보다 작은 feature는 scale 1을 사용해 상수 feature의 입력 증폭을 피합니다. 가변 길이 배치는 packing으로 패딩을 제외합니다. 최소 val BCE를 기록한 가중치를 저장하고 test는 선택 완료 후 평가합니다. CPU/CUDA에서 seed와 결정적 알고리즘을 설정합니다. 환경 간 비트 단위 재현성은 보장하지 않습니다.

### 세션 단위 교차 검증 (`cross_validate.py`)

val은 세션 5개에 양성이 몇 개뿐이라, 한 번의 학습으로 고른 epoch이나 설정 비교는 우연에 크게 좌우됩니다. `src/cross_validate.py`는 학습 설정을 세션 단위 k-fold로 평가합니다.

- **입력**: `baseline.py train`과 같은 설정 파일과 명령줄 덮어쓰기(`--hidden-size` 등), fold 수(`--folds`, 기본 5), 서로 다른 seed 목록(`--seeds`, 기본 42 43). 원본 데이터여야 합니다. split이 train/val/test가 아니거나, `augment`로 만든 증대본(`augmentation_copy` ≥ 1)이 있거나, 한 세션이 test와 train/val에 걸쳐 있으면 오류입니다. `--results`가 입력 데이터·스키마 경로면 거부합니다.
- **처리**: test 구간은 제외하고 train·val 구간을 합칩니다. 세션(`driver_id`, 없으면 `run_id`)을 나누지 않고 fold에 배정하며, 양성이 많은 세션부터 그 라벨 양성이 가장 적은 fold에 넣어 양성을 고르게 퍼뜨립니다. 같은 데이터면 항상 같은 fold가 나옵니다. fold × seed마다 새 모델을 그 fold를 val로 학습하고 epoch별 val BCE를 기록합니다. 증대 seed는 학습 seed에서 유도하므로(`augmentation_seed` 무시) seed마다 다른 증대본을 씁니다. checkpoint는 임시 폴더에 쓰고 지웁니다.
- **출력**: fold 구성(세션, 구간 수, 라벨별 양성 수), run별 epoch 곡선, 요약 JSON. 요약의 `selected_epochs`는 모든 run의 epoch별 val BCE 평균이 가장 낮은 epoch 수이고, `selected_mean_val_bce`/`selected_std_val_bce`가 그때의 평균·표준편차입니다. 설정 비교에는 이 값을 쓰고, 같은 fold의 상수 예측 기준선 `mean_prevalence_val_bce`보다 낮은지 확인하세요. `mean_best_epoch_val_bce`는 run마다 자기 val로 고른 최저값이라 낙관적입니다. 모든 run은 fold 크기와 관계없이 같은 가중치입니다.
- **F1·정밀도·재현율**: 선택된 epoch에서 확률 0.5 기준으로 계산합니다. fold마다 양성이 몇 개뿐이라 fold별 F1을 평균하지 않고, 모든 run의 TP/FP/FN을 합친 뒤 나눕니다(`selected_val_micro`, 라벨별 `selected_val_per_label`, fold별 `selected_val_micro_f1_by_fold`, epoch별 `pooled_val_micro_f1_by_epoch`). 분모가 0이면(표시한 것도, 양성도 없음) null입니다. 같은 fold val로 라벨별 기준값을 고르면 낙관적이므로 기준값 조정은 하지 않습니다. 각 seed가 모든 세션을 한 번씩 검증하므로 개수는 seed 수만큼 곱해져 있습니다.

```powershell
.\.venv\Scripts\python.exe src/cross_validate.py --config res/train-config.bias-copies.json --folds 5 --seeds 42 43 --results artifacts/cv-bias-copies.json
# 설정 덮어쓰기는 train과 같은 옵션으로; --seed와 --output은 run마다 정하므로 쓸 수 없음
.\.venv\Scripts\python.exe src/cross_validate.py --config res/train-config.bias-copies.json --hidden-size 32 --results artifacts/cv-h32.json
```

`train-config.bias-copies.json`을 5 fold × seed 42, 43으로 평가한 결과(`artifacts/cv-bias-copies.json`)는 9 epoch에서 평균 val BCE 0.184 ± 0.138로, 모든 fold에서 상수 예측 기준선(평균 0.290)보다 낮았습니다. 평균 곡선은 5~13 epoch에서 0.184~0.201로 평탄하고 이후 천천히 오릅니다. 표준편차는 대부분 fold 간 난이도 차이(양성이 몰린 세션이 있는 fold 0: 0.446, fold 4: 0.057)이며 같은 fold의 seed 간 차이는 작으므로, 설정 비교는 같은 fold끼리 짝지어 보세요. 9 epoch의 확률 0.5 기준 합산 val F1은 0.581(정밀도 0.818, 재현율 0.450)입니다. 라벨별로는 turn_signal_missing 0.717, stop_failure 0.545, over_speed 0.476이고 vehicle_device_error와 section_time_over는 하나도 맞히지 못했습니다(0.000). epoch별 F1은 2~9 epoch에서 0.17에서 0.58로 오른 뒤 0.55~0.60에 머뭅니다.

### 크기·learning rate 격자 탐색 (`sweep.py`)

`src/sweep.py`는 고정된 train/val split과 같은 seed로 hidden_size × num_layers × learning_rate 조합마다 새 모델을 학습하고 val BCE로 순위를 매깁니다. 교차 검증보다 훨씬 빠르지만 val 세션이 적어 차이가 작으면 우연일 수 있으므로, 상위 설정만 `cross_validate.py`로 다시 확인하세요.

- **입력**: `baseline.py train`과 같은 설정 파일과 명령줄 덮어쓰기(`--epochs` 등), 격자 값 `--hidden-sizes`(기본 32 64 128), `--layer-counts`(기본 1 2), `--learning-rates`(기본 0.0005 0.001 0.002). `--hidden-size`/`--num-layers`/`--learning-rate`/`--output`은 시도마다 정해지므로 쓸 수 없습니다.
- **처리**: 시도마다 최저 val BCE epoch를 고르며 checkpoint는 임시 폴더에 쓰고 지웁니다. 시도가 끝날 때마다 결과 파일을 저장하고, 같은 결과 파일·설정·데이터로 다시 실행하면 끝난 시도는 건너뜁니다(중단 후 재개, 격자 확장). 설정이나 데이터가 다르면 다른 `--results`를 요구합니다.
- **출력**: 시도별 설정, best epoch, val BCE·micro F1(확률 0.5)·라벨별 F1, epoch별 train/val BCE와 val BCE 순위(`ranking`). 설정을 test로 고르지 않도록 test 지표는 넣지 않습니다. 기본은 epoch별 로그도 출력하며 `--quiet`는 시도 결과만 출력합니다.

```powershell
.\.venv\Scripts\python.exe src/sweep.py --config res/train-config.bias-copies.json --epochs 12 --results artifacts/sweep-20261127.json 2> artifacts/sweep-progress.20261127.log
Get-Content artifacts/sweep-progress.20261127.log -Wait
```

학습 결과 JSON에도 `best_epoch`와 `epoch_history`(epoch별 train/val BCE와 확률 0.5 기준 val 라벨별 TP/FP/FN)가 들어가고, 평가 지표의 `per_label`에도 라벨별 `true_positive`/`false_positive`/`false_negative`가 들어갑니다. test는 설정을 모두 고른 뒤 마지막에 한 번만 평가하세요.

### 라벨별 판정 기준값

학습이 끝나면 선택된 모델로 val 확률을 계산해 라벨마다 F1이 가장 높은 기준값을 고릅니다. 기준값은 감점으로 표시한 가장 낮은 확률과 그다음 확률의 중간에 두어 val 표본 위에 걸치지 않게 합니다. val에 그 라벨의 판정된 양성과 음성이 모두 있어야 하며, 없으면 0.5를 유지합니다. 같은 F1이면 높은 기준값(덜 표시)을 고릅니다. 기준값은 checkpoint의 `label_thresholds`에 저장되고, 학습 출력의 `lstm_tuned_thresholds`가 이 기준값으로 계산한 split별 지표입니다. 기준값을 val로 골랐으므로 val 지표는 낙관적이며, **test 지표가 독립 추정치**입니다. val 양성이 몇 개뿐이면 기준값이 그 몇 개에 맞춰지므로 데이터가 늘면 다시 학습하세요.

`session_predict.py`와 `report.py`는 `--threshold`를 생략하면 checkpoint의 라벨별 기준값을 쓰고, 각 구간의 `threshold`에 판정 라벨별 기준값을 기록합니다. `--threshold 0.5`처럼 숫자를 주면 모든 라벨에 그 값을 씁니다. 기준값이 없는 이전 checkpoint는 모든 라벨 0.5입니다.

출력은 split별 BCE, micro F1, 라벨별 양성 수/F1이며 `lstm`의 F1은 확률 0.5 기준입니다. `per_label` 순서는 출력 label_names와 같습니다. train 라벨 빈도의 상수 예측 baseline도 같은 split에서 평가합니다. 단일 클래스 train 라벨은 경고합니다. 불균형/양성 없는 평가 라벨의 F1은 단독 성능 근거로 사용하지 마세요. 임계값 튜닝, PR 기반 평가, 가중 손실은 실제 라벨 분포 확인 후 추가할 항목입니다.

Checkpoint는 스키마, 정규화, 가중치, 선택 epoch, 실행 인자, 버전, 데이터 SHA-256 및 평가 결과를 포함합니다. 추론은 checkpoint 통계를 재사용하고 개선점 후보 확률만 JSONL로 출력합니다. 이 확률은 보정된 신뢰도나 관측 근거가 아닙니다. 실제 안내는 별도 근거 확인과 규칙 경로가 필요합니다.

데이터 전체를 메모리에 읽는 작은 baseline입니다. 긴 구간은 수집 단계에서 메모리 예산을 정하고 길이를 관리하세요. 앞부분 사건을 놓치는지와 불균형 라벨 성능은 실제 데이터에서 확인해야 합니다. Unity 기록기 CSV와 오프라인 구간 추론을 연결하며 실시간 모델 호출은 포함하지 않습니다.
