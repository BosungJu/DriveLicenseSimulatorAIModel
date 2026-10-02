# 운전 분석 AI 개발

Unity 시뮬레이터: `C:\WorkSpace\DriverLicenseSimulator`

## LSTM baseline

신경망은 `src/model.py`의 `SectionLSTM`, 학습·평가·추론·저장은 `LSTMBaseline` 클래스가 관리합니다. 설정의 `device` 또는 `--device cuda:0`으로 CUDA를 선택할 수 있습니다. 기본 `auto`는 사용 가능한 GPU를 우선합니다. CUDA wheel 설치와 클래스 사용 예시는 아래 사용 문서에 있습니다.

구간별 주행 시계열을 받아 여러 개선점 후보의 확률을 예측하는 PyTorch baseline입니다. 학습·검증 실행 분리, train 정규화, 가변 길이 처리, checkpoint 저장 및 추론을 제공합니다. [학습 설정 예시](res/train-config.example.json)에서 learning rate 등 하이퍼파라미터와 feature별 bias/noise 증대 강도를 조절할 수 있습니다. 설치, JSONL 데이터 계약, 실행 명령과 제한은 [LSTM 사용 문서](docs/lstm-baseline.md)를 참고하세요. 실제 데이터 및 Unity 기록기 연결은 아직 준비되지 않았습니다.

## Codex ↔ Claude 개발 협업

주행 데이터 전송이 아닌, 개발 계획과 코드에 대한 검토 연결입니다.

- **Codex → Claude**: 전역 Codex MCP 설정의 `claude-review` 서버가 `tools/claude-review-mcp.mjs`를 실행합니다. `review_with_claude(context)`에 요구사항, 관련 코드 또는 diff, 검증 결과를 전달하면 Claude가 한국어 검토를 반환합니다. Claude는 파일을 읽거나 수정하지 않습니다. Codex가 의견을 판단하여 반영합니다.
- **Claude → Codex**: 이 폴더의 `.mcp.json`이 `codex mcp-server`를 등록합니다. 폴더에서 `claude`를 시작하고 `/mcp`에서 연결을 확인하세요. 프로젝트 MCP 신뢰 확인이 표시되면 해당 서버를 확인하세요. Codex에 작업을 위임할 때는 작업 범위와 수정 허용 여부를 명시하세요.

새 MCP 설정은 Codex 앱/세션을 다시 시작한 뒤 로드됩니다. 현재 열린 채팅에 도구가 자동으로 추가되는 것은 아닙니다. 연결 후 “이 구현안을 Claude로 검토하고 의견을 반영해줘”라고 요청하면 됩니다. 매 코드 변경마다 자동으로 검토를 호출하는 감시 기능은 없습니다.

Claude 검토는 로컬 Claude Code 로그인과 사용량을 이용합니다. API 키를 저장하지 않습니다. 전달한 개발 문맥은 Claude 서비스로 전송되므로 비밀 값은 포함하지 마세요. Node.js와 Claude Code가 필요하며 추가 npm 패키지는 없습니다. npm 설치 경로와 다르면 `CLAUDE_EXECUTABLE` 환경변수에 실행 파일 경로를 지정하세요.

```powershell
# 등록 재현 (이미 등록되어 있으면 다시 실행할 필요 없음)
codex mcp add claude-review -- node C:\WorkSpace\DriveLicenseSimulatorAIModel\tools\claude-review-mcp.mjs
codex mcp get claude-review

# 단위 테스트 (Claude 호출 및 네트워크 사용 없음)
node --test tools/claude-review-mcp.test.mjs

# Claude에서 프로젝트 시작
claude
```

검토 호출은 최대 120초이며 동시에 한 요청만 실행합니다. 실패는 MCP 오류 결과로 반환하며, 재시도는 호출자가 결정합니다. 코드 파일 변경이나 셸 실행 도구는 Claude 검토 프로세스에 제공하지 않습니다.

공식 참고: [Claude MCP](https://code.claude.com/docs/en/mcp), [프로그램에서 Claude 호출](https://code.claude.com/docs/en/headless).
