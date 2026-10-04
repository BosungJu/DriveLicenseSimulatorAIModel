# 운전 분석 AI 개발

Unity 시뮬레이터 repository: `DriverLicenseSimulator`

## LSTM baseline

신경망은 `src/model.py`의 `SectionLSTM`, 학습·평가·추론·저장은 `LSTMBaseline` 클래스가 관리합니다. 설정의 `device` 또는 `--device cuda:0`으로 CUDA를 선택할 수 있습니다. 기본 `auto`는 사용 가능한 GPU를 우선합니다. CUDA wheel 설치와 클래스 사용 예시는 아래 사용 문서에 있습니다.

구간별 주행 시계열을 받아 여러 개선점 후보의 확률을 예측하는 PyTorch baseline입니다. 학습·검증 실행 분리, train 정규화, 가변 길이 처리, checkpoint 저장 및 추론을 제공합니다. [학습 설정 예시](res/train-config.example.json)에서 learning rate 등 하이퍼파라미터와 feature별 bias/noise 증대 강도를 조절할 수 있습니다. 설치, JSONL 데이터 계약, 실행 명령과 제한은 [LSTM 사용 문서](docs/lstm-baseline.md)를 참고하세요. 실제 데이터 및 Unity 기록기 연결은 아직 준비되지 않았습니다.
