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

설정 파일의 data/schema/output 상대 경로는 **설정 파일 폴더** 기준입니다. 명령줄에서 지정하는 상대 경로는 현재 작업 폴더 기준이며, 명령줄 값이 JSON 값보다 우선합니다. JSON 변수 이름은 위처럼 snake_case입니다. 알 수 없는 설정 키와 잘못된 강도/학습 인자는 오류로 처리합니다. 설정 변경은 다음 학습 실행에 적용됩니다. 각 checkpoint에 실제 하이퍼파라미터와 증대 설정을 저장합니다.

예를 들어 `"hidden_size": 64, "num_layers": 2`는 은닉 크기 64인 LSTM을 2층 쌓습니다. 명령줄에서는 `train --config res/train-config.example.json --num-layers 2 --output artifacts/lstm-2layers.pt`로 변경할 수 있습니다. Python에서는 `SectionLSTM(feature_count=6, label_count=2, hidden_size=64, num_layers=2)` 또는 fit 전 `settings.num_layers = 2`를 사용합니다. checkpoint에 층 수를 저장하고 load 시 같은 구조를 복원합니다. num_layers가 없는 기존 checkpoint와 설정은 1층으로 처리합니다.

증대는 train batch를 만들 때 원본 단위에서 새로 생성하고 이후 원본 train 통계로 정규화합니다. 파일이나 라벨은 수정하지 않습니다. val/test 평가와 추론은 원본을 사용합니다. 증대 표본을 파일로 여러 배 복제하지 않으며, epoch마다 다른 변형을 제공합니다. 증대 난수는 별도 generator를 사용합니다.

feature별 bias는 `[-bias_max, +bias_max]` 균등분포로 구간당 한 번 뽑습니다. 노이즈 후보도 대칭 균등분포이며 `noise[t] = (1 - smoothing) * noise[t-1] + smoothing * candidate[t]`로 평활화합니다. 첫 노이즈는 첫 후보 값입니다. 따라서 총 변동의 절댓값은 bias_max + noise_max 이내입니다. `smoothing=1`은 독립 노이즈입니다. 경계 보호/범위 검사로 거절된 증대를 제외하면 대칭 분포이며, 거절 때문에 최종 적용 분포는 비대칭이 될 수 있습니다.

증대할 항목은 스키마 `continuous_features`에 명시된 연속형 feature만 가능합니다. 이진 제동·명령·구간 ID는 이 목록에 넣지 마세요. 설정에 없는 feature는 그대로 유지됩니다. 후보가 범위를 벗어나거나 보호 경계의 아래/같음/위 관계를 바꾸면 **그 구간의 해당 feature 증대를 취소**합니다. 원본이 설정 범위를 벗어나면 오류입니다. 다른 feature는 별도로 처리합니다.

예시 설정은 speed_kmh에 bias 최대 0.05 km/h, noise 최대 0.02 km/h를 적용하고 0/1/20 km/h를 보호합니다. 이 값들은 효과가 입증된 값이 아니라 조정용 시작 예시입니다. 실제 판정 설정에 맞게 경계를 바꾸세요. 조향도 `features`에 같은 구조로 추가할 수 있지만, 조향 안정성처럼 변화율에 의존하는 라벨은 경계값 검사만으로 보존을 보장할 수 없습니다. 모든 라벨의 의미가 유지되는 feature/강도만 허용해야 합니다. 이 증대는 작은 관측 변동에 대한 강건성을 실험하는 용도이며 물리적으로 다른 주행 생성은 Unity 재시뮬레이션이 필요합니다.

증대 없음(probability 0), noise만(bias_max 0), bias+noise를 별도 output과 동일 원본 split으로 비교하세요. test 성능으로 설정을 반복 선택하지 말고 val에서 선택한 뒤 test를 평가하세요. `augmentation`을 생략하거나 null로 설정하면 기존 학습 경로를 그대로 사용합니다. 증대만 별도 JSON 파일로 분리하려면 `--augmentation 파일경로`로 덮어쓸 수도 있습니다.

학습 종료 출력과 checkpoint의 `augmentation_stats`에서 feature별 attempted/applied/unchanged/skipped/rejected_bounds/rejected_threshold 횟수를 확인할 수 있습니다. attempted는 확률 선택 후 후보를 만든 횟수, applied는 실제 값이 달라진 횟수입니다. 시도 대비 적용률이 10%보다 낮으면 경고합니다. 정지 프레임처럼 보호 경계와 정확히 같은 값이 있으면 해당 구간의 feature 증대가 모두 취소될 수 있으므로, 이 통계로 강도 조정 효과를 확인하세요. 원본 train 범위는 학습 시작 전에 전체 검사합니다.

## 데이터 계약 v1

`schema.json` 예시:

```json
{"schema_version":1,"sample_rate_hz":50,"feature_names":["speed_kmh","accel_mps2","yaw_rate_dps","steering_input","throttle_input","brake_input","gear_park","gear_reverse","gear_neutral","gear_drive","parking_brake_on","engine_on","left_signal_on","right_signal_on","hazard_on","headlight_on","high_beam_on","seat_belt_on","horn_on"],"label_names":["insufficient_stop","unstable_steering"]}
```

JSONL의 각 행은 구간 하나입니다. 학습 행 예시:

```json
{"schema_version":1,"sample_rate_hz":50,"run_id":"anonymous-run-001","section_id":2,"split":"train","feature_names":["speed_kmh","accel_mps2","yaw_rate_dps","steering_input","throttle_input","brake_input","gear_park","gear_reverse","gear_neutral","gear_drive","parking_brake_on","engine_on","left_signal_on","right_signal_on","hazard_on","headlight_on","high_beam_on","seat_belt_on","horn_on"],"features":[[4.0,-0.5,3.0,-0.25,0.5,1.0,0,0,0,1,0,1,1,0,0,1,0,1,0],[0.0,0.0,0.0,0.0,0.0,0.75,0,1,0,0,1,1,0,0,0,0,0,1,0]],"label_names":["insufficient_stop","unstable_steering"],"labels":[1,0]}
```

- 관측은 고정 Hz로 수집하거나 사전에 리샘플합니다. 시간 간격은 메타데이터만으로 검증할 수 없으므로 수집 계층이 보장해야 합니다. 단위와 실제 적용 입력의 범위를 feature 이름 및 수집 문서로 고정하세요. 50 Hz는 예시입니다.
- `features`는 패딩 없는 T × F 유한 수치 배열입니다. 빈 구간과 결측/NaN/Inf는 거부합니다. feature 이름과 순서는 schema 및 checkpoint와 정확히 일치해야 합니다.
- 라벨 이름은 구성 가능하며 예시 라벨의 판정 기준은 아직 확정하지 않았습니다. 전문가가 정의한 계약과 근거 기록이 필요합니다. `1`은 확인된 양성, `0`은 확인된 음성입니다. 미평가·적용 불가 항목을 0으로 채우지 마세요. v1은 결측 라벨을 지원하지 않으므로 모든 항목이 평가된 표본만 사용합니다.
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

steering_input은 부드럽게 제한된 실제 휠 조향각이 아니며, throttle_input은 실제 모터 토크가 아닙니다. 시동이 꺼져 있거나 P/N 기어여도 사용자의 가속 입력은 기록될 수 있습니다. 기어·주차 브레이크·차량 상태 토글은 Update에서 바뀌므로 표본에는 해당 물리 스텝 직전의 최신 상태가 기록됩니다. 상태는 차량 제어 코드 이후, 다음 물리 시뮬레이션 이전에 기록하므로 speed_kmh에는 CarController.Brake의 페달 비례 감속(끝까지 밟으면 `linearVelocity *= 0.98`)이 이미 반영됩니다. 와이퍼(`WiperController`)는 차량 컨트롤러와 연결된 참조가 없어 포함하지 않았습니다. 이진·원-핫 feature는 `continuous_features`에 넣지 마세요.

CSV 열 순서는 `time_s,point_number,heading_deg,pos_x,pos_z` 뒤에 위 feature 순서입니다. Unity의 `DrivingFeatureColumns`가 CSV와 학습 JSONL에 같은 열 정의·형식을 사용합니다. heading_deg/pos_x/pos_z는 기본 모델 feature에서 제외하고, time_s/point_number는 시간과 구간 메타데이터로 사용합니다. AI의 `session_predict.py`는 열 이름으로 checkpoint feature를 선택하며, 조향(-1~1)·페달(0~1) 범위, 이진 값, 기어 원-핫(정확히 하나 활성)을 검사합니다. 입력 feature를 사용하는 모델에서는 time_s 간격도 검사해 누락 프레임을 조용히 압축하지 않습니다(기록의 3자리 반올림 오차 허용). 아날로그 브레이크 이전 CSV(`brake_applied`, 부호 있는 throttle)에 0이나 추정 입력을 채우지 않으므로 새 19-feature 모델에서는 오류입니다. `brake_applied`를 feature로 쓰는 이전 checkpoint는 이전 로그 추론을 위해 throttle_input -1~1 범위를 유지합니다. 새 모델은 라벨을 준비한 새 기록으로 재학습해야 하며 기존 checkpoint의 스키마와 가중치를 자동으로 변경하지 않습니다. 예시 라벨 정의는 계속 전문가 검토가 필요합니다.

입력을 요구하는 모델에서 한 표본이라도 입력이 비어 있으면 세션 전체 추론을 실패시키며 부분 결과를 만들지 않습니다. 물리 feature만 사용하는 기존 모델은 이전 호환 경로를 유지하므로 time_s 간격 검사는 적용하지 않습니다.

### Unity 학습 데이터 수집

`TrainingDataCollector`(Unity `Assets/Scripts/AI_Report`)를 `ReportManager`와 같은 오브젝트에 붙이면, 코스 완료 후 세션 폴더에 `training_sections.jsonl`을 함께 씁니다. 각 행은 위 데이터 계약 v1의 라벨 없는 구간 레코드이며 run_id(세션 폴더 이름 + 무작위 8자리), 선택적 익명 driver_id, section_id, start_time_s를 포함합니다. 값은 driving_log.csv와 같은 소수 3자리 반올림입니다. 입력·상태 결측, NaN, 고정 간격(0.5~1.5스텝)을 벗어난 시각이 있는 구간은 레코드를 만들지 않고 건너뜁니다. 학습 데이터는 세션 파일들을 합친 뒤 각 행에 split/label_names/labels를 추가해 만듭니다. 코스를 완료하지 않고 초기화한 주행은 저장되지 않습니다.

Unity 수동 확인: 코스 주행 중 조향 양방향, 가속·브레이크 페달 부분 입력, P/R/N/D 변속, 주차 브레이크, 시동·방향지시등·비상등·전조등·상향등·안전벨트·경음기를 조작하고 코스 종료 후 AIReport 세션의 driving_log.csv와 training_sections.jsonl을 확인하세요. EditMode Test Runner의 `DrivingReportCsvTests`, `DrivingTrainingDatasetTests`가 열 순서, 원-핫 기어, 아날로그 페달, 결측 빈칸, 문화권에 무관한 숫자 형식과 구간 건너뛰기를 검증합니다.

## 학습과 결과 해석

정규화는 train의 실제 프레임만 사용합니다. 표준편차가 1e-6보다 작은 feature는 scale 1을 사용해 상수 feature의 입력 증폭을 피합니다. 가변 길이 배치는 packing으로 패딩을 제외합니다. 최소 val BCE를 기록한 가중치를 저장하고 test는 선택 완료 후 평가합니다. CPU/CUDA에서 seed와 결정적 알고리즘을 설정합니다. 환경 간 비트 단위 재현성은 보장하지 않습니다.

출력은 split별 BCE, micro F1, 라벨별 양성 수/F1이며 F1은 확률 0.5 기준입니다. `per_label` 순서는 출력 label_names와 같습니다. train 라벨 빈도의 상수 예측 baseline도 같은 split에서 평가합니다. 단일 클래스 train 라벨은 경고합니다. 불균형/양성 없는 평가 라벨의 F1은 단독 성능 근거로 사용하지 마세요. 임계값 튜닝, PR 기반 평가, 가중 손실은 실제 라벨 분포 확인 후 추가할 항목입니다.

Checkpoint는 스키마, 정규화, 가중치, 선택 epoch, 실행 인자, 버전, 데이터 SHA-256 및 평가 결과를 포함합니다. 추론은 checkpoint 통계를 재사용하고 개선점 후보 확률만 JSONL로 출력합니다. 이 확률은 보정된 신뢰도나 관측 근거가 아닙니다. 실제 안내는 별도 근거 확인과 규칙 경로가 필요합니다.

데이터 전체를 메모리에 읽는 작은 baseline입니다. 긴 구간은 수집 단계에서 메모리 예산을 정하고 길이를 관리하세요. 앞부분 사건을 놓치는지와 불균형 라벨 성능은 실제 데이터에서 확인해야 합니다. Unity 기록기 CSV와 오프라인 구간 추론을 연결하며 실시간 모델 호출은 포함하지 않습니다.
