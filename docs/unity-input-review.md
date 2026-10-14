# Unity 입력 feature 연동 검증

2026-10-14 사용자 요청으로 Unity 자동차 입력 기록과 AI feature를 맞췄습니다. 사용자는 조향·가속·제동 입력 기록 확장도 명시적으로 요청했습니다. 차량 제어 소유 영역의 CarController는 읽기 전용 입력 속성과 수집 시각만 추가했고, 물리 제어 분기는 유지했습니다.

## 변경 계약

- CSV 기존 8개 열을 유지하고 steering_input/throttle_input/brake_applied를 추가했습니다.
- AI 기본 feature는 속도, 종방향 가속도, Y축 회전율 및 위 세 입력의 6개입니다.
- 사용할 수 없는 입력은 CSV 빈칸입니다. 물리 상태는 계속 기록하며 새 입력 모델은 결측을 명시적으로 거부합니다.
- 새 입력 feature를 사용하는 추론은 시간 간격도 검사합니다. 기존 checkpoint를 자동 수정하지 않으며 새 feature로 재학습해야 합니다.

## 교차 리뷰

cross-review MCP로 두 프로젝트의 변경을 검토받았습니다. 1차 CHANGES_NEEDED의 System.Text import 누락, 입력 불가 시 물리 기록 회귀, 누락 구간의 시간/가속도 차분 문제를 확인하고 수정했습니다. 한 번의 재검토 결과는 APPROVE입니다.

재검토의 추가 지적 중 세션 전체 실패와 기존 물리 feature의 시간 간격 미검사는 의도한 호환 동작이며 사용 문서에 명시했습니다. 1024초 이후 float 시각 때문에 0.0011초 허용 오차를 넘는다는 지적은 50 Hz float32 누적과 CSV 반올림을 1024/2048/16384초에서 각 10,000스텝 재현해 확인했습니다. 최대 간격 오차는 0.001초로 허용 범위였으므로 해당 근거로 시각 필드 타입이나 허용 오차를 변경하지 않았습니다.

## 실제 검증과 제한

- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`: Python 테스트 49개 통과. 새 입력의 순서/범위/이진값, 누락 입력/시간 간격 거부, 기존 checkpoint, CSV→학습→세션 추론을 검증했습니다. 기존 CUDA 테스트도 실행했습니다.
- production DrivingReportData.cs를 PowerShell Add-Type로 컴파일하고 실제 CSV를 생성했습니다. fr-FR 문화권에서도 해당 CSV→Python 변환의 부호 있는 입력, 제동 0/1, 결측 입력의 기존 물리 경로 호환을 확인했습니다. 이 검증은 Unity 런타임 테스트가 아닙니다.
- 두 저장소의 git diff --check 통과. 기존 AI 작업 변경은 유지했습니다.
- Unity 6000.0.70f1 배치 EditMode 테스트는 해당 프로젝트가 다른 Unity 인스턴스에서 열려 있어 실행하지 못했습니다. 로그는 artifacts/verification/unity-input-tests.log입니다. 추가한 DrivingReportCsvTests를 열린 편집기의 EditMode Test Runner에서 실행해야 합니다.
- 실제 주행에서 CarController→ReportManager 실행 순서와 입력/물리 상태의 연동은 사용 문서의 수동 확인 절차로 검증해야 합니다.
