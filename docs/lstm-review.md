# Claude 검토와 검증 기록

2026-10-02에 `claude-review` MCP로 계획과 구현 코드/테스트를 각각 검토했습니다. 주행 데이터는 제공하지 않았고 Claude는 전달된 개발 문맥만 정적으로 검토했습니다.

반영 사항:

- feature/label 이름과 순서, schema version, 고정 샘플링 주파수를 검사합니다.
- 실행 ID 및 제공된 익명 운전자 ID가 split을 넘으면 거부합니다.
- 상수 feature는 scale 1을 사용합니다. 극단적인 val/test feature로 train-only 통계를 검증합니다.
- 결측 라벨/feature와 잘못된 타입을 행 번호가 있는 오류로 거부합니다.
- 같은 데이터 바이트를 파싱하고 해시하여 학습 중 파일 변경으로 provenance가 달라지는 문제를 피합니다.
- train 빈도 기준선 및 라벨별 양성 수/F1을 기록합니다. 경로는 checkpoint 실행 인자에서 제외합니다.
- 학습/추론 테스트 파일을 분리하고 결정론 전역 설정을 테스트 종료 시 복원합니다.
- 학습 sanity 테스트는 실제 production train 경로를 실행하고 합성 데이터에서 상수 기준선보다 낮은 BCE를 확인합니다.

판단 및 후속 항목:

- 작은 baseline은 기존 단일 파일에서 유지합니다. pooling, class weight, PR 기반 평가, 임계값 튜닝은 실제 라벨 분포와 실패 사례를 확인한 후 결정합니다.
- 고정 Hz 수집, 전문가 라벨, 중복 표본 관리 및 Unity 연결은 데이터 수집 계층에서 준비해야 합니다.
- test split이 없는 경우 경고하며 val 결과는 모델 선택에 사용됐으므로 최종 일반화 성능으로 해석하지 않습니다.

검증 명령은 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`입니다. Python 3.12.15 / PyTorch 2.14.1+cpu에서 테스트 6개가 통과했습니다. 합성 데이터 검증이며 실제 주행 성능을 입증하지 않습니다. NumPy가 없는 환경에서 PyTorch 초기화 경고가 발생하지만 이 경로는 NumPy를 사용하지 않습니다.

vexp `verify_done`은 `.git`에 저장소 메타데이터가 없어 변경 집합 검증을 제공하지 못했습니다. 테스트는 직접 실행했습니다.

## 데이터 증대와 학습 설정 추가 검토

같은 날 Claude로 증대 및 JSON 설정 처리 코드를 검토했습니다. train만 증대하는 경로, 원본 정규화, 입력 보존, 평활 노이즈 상한, CLI 우선순위와 checkpoint 호환성을 확인했습니다. 추가 지적은 다음과 같이 반영했습니다.

- 범위 밖 원본 train은 probability와 관계없이 모델 생성 전에 거부합니다.
- 보호 경계 때문에 실제 증대가 취소될 수 있어 feature별 적용/거절 통계를 출력하고 checkpoint에 저장합니다. 적용률 10% 미만은 경고합니다.
- 증대 seed를 모델 seed에서 분리하고 augmentation_seed 설정/CLI로 노출합니다.
- 강도 단위와 경계 보호에 따른 비대칭 수락 가능성을 문서화했습니다.

JSON에서 feature별 bias/noise/smoothing/범위/보호 경계 및 lr/epoch/batch/hidden/seed/output을 조절할 수 있습니다. 변경은 다음 학습 실행부터 적용됩니다. 자동 라벨 재생성 또는 진행 중 학습의 실시간 파라미터 변경은 구현 범위에 포함되지 않습니다.

기존 명령으로 테스트 19개가 통과했습니다. 증대 상한, 구간 고정 bias, 노이즈 연속성, 임계값 보존, 원본 보존, seed 재현성, config/CLI override, 학습 전 범위 검사, 실제 학습·추론과 checkpoint 경로를 검증했습니다. 실제 데이터의 증대 효과는 아직 측정하지 않았습니다.

## 모델 클래스와 CUDA 검토

`SectionLSTM`을 src/model.py에 분리하고 LSTMBaseline에 fit/evaluate/predict/save/load 및 장치 선택을 추가한 변경을 Claude로 정적 검토했습니다. 테스트 환경변수 복원, 실제 사용 장치의 metadata 기록, CUDA wheel 설치 안내 및 cuBLAS 설정 타이밍 지적을 반영했습니다. 기존 CLI와 format_version 1 checkpoint는 유지합니다.

CUDA 12.8 인덱스에서 제공하는 requirements 범위의 wheel이 PyTorch 2.11.0+cu128이므로 기존 2.14.1+cpu를 해당 빌드로 교체했습니다. uv 캐시 rename은 Windows 파일 잠금으로 실패했고, pip 직접 설치로 완료했습니다. 프로젝트 가상환경에만 설치했습니다.

검증 환경: Python 3.12.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8, NVIDIA GeForce RTX 4060 Ti. `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`로 테스트 30개 모두 통과했으며 CUDA 테스트도 실제 실행했습니다. GPU 학습/평가/추론, 동일 seed 재학습, CPU checkpoint snapshot, GPU에서 학습한 모델의 CPU 재로드, CLI auto GPU 선택과 CPU 추론을 확인했습니다. 합성 데이터 검증이며 실제 주행 성능 평가는 별도입니다.

## num_layers 설정 추가

LSTM 층 수를 생성자, JSON 설정 및 `--num-layers`로 조절할 수 있게 추가했습니다. 기본값과 기존 Namespace/checkpoint fallback은 1입니다. checkpoint에 실제 층 수를 저장하고 같은 구조로 복원합니다. Claude 정적 검토에서 변경 범위의 correctness 결함은 발견되지 않았습니다.

기존 검증 명령으로 테스트 32개가 모두 통과했습니다. CPU 2층 모델 저장/로드, CUDA 2층 학습·재현성·CPU 재로드, 설정/CLI 우선순위, 잘못된 층 수 거부 및 층 수 필드 없는 기존 checkpoint를 검증했습니다.
