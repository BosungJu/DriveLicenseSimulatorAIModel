"""Writes a Korean driving coaching report from section deductions and LSTM predictions with the OpenAI API.

Input  (session folder): section_results.json (Unity ReportManager), predictions.json (session_predict.py).
Output (session folder): report_input.json (exact data sent to the model), report.md.
Requires the OPENAI_API_KEY environment variable.
"""

import argparse
import json
import os
from pathlib import Path

if __package__:
    from .session_predict import DEFAULT_LABEL_THRESHOLD, PREDICTION_FILE, SECTION_RESULT_FILE, predict_session
else:
    from session_predict import DEFAULT_LABEL_THRESHOLD, PREDICTION_FILE, SECTION_RESULT_FILE, predict_session

REPORT_INPUT_FILE = "report_input.json"
REPORT_FILE = "report.md"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"

# Korean names of DrivingDeductionReason in the Unity simulator (DrivingScoreSheet.cs).
DEDUCTION_REASON_NAMES = {
    "LaneKeepingLineTouch": "차로 유지 중 선 접촉",
    "ParkingLineTouch": "주차 중 선 접촉",
    "ParkingTimeOver": "주차 제한시간 초과",
    "StartTimeOver": "출발 제한시간 초과",
    "HillNotStopped": "경사로 미정지",
    "HillShortStop": "경사로 정지시간 부족",
    "HillRollback": "경사로 뒤로 밀림",
    "HillRollbackOver": "경사로 허용 거리 이상 밀림",
    "HillPassTimeOver": "경사로 통과시간 초과",
    "SignalLongStop": "교차로 장시간 정지",
    "SignalPassTimeOver": "교차로 통과시간 초과",
    "AccelerationTooSlow": "가속 구간 목표속도 미달",
    "OverSpeed": "과속",
    "EmergencyStopLate": "돌발 정지 지연",
    "CourseTimeOver": "코스 제한시간 초과",
}

REPORT_INSTRUCTIONS = """당신은 운전면허 기능시험 시뮬레이터의 운전 코치입니다.
입력 JSON은 한 번의 주행 결과이며, 이 데이터에 있는 사실만 근거로 한국어 Markdown 보고서를 작성하세요.

데이터 의미:
- official_deductions: 시뮬레이터 채점기가 확정한 공식 감점입니다. 점수와 합격 여부의 근거입니다.
- ai_analysis: LSTM 모델이 구간 시계열로 추정한 개선점 후보입니다. probabilities는 보정된 신뢰도가 아니며 공식 감점이 아닙니다.
  flagged_labels는 threshold 이상인 라벨이고, 비어 있으면 모델 기준 정상 구간입니다. ai_analysis가 null이면 예측이 없는 구간입니다.

작성 규칙:
1. "## 총평": 점수, 합격/불합격(실격 여부 포함), 가장 중요한 개선점 1~3개.
2. "## 구간별 분석": 공식 감점이나 flagged_labels가 있는 구간만 포인트 번호 순서로 작성합니다.
   각 구간에서 공식 감점과 AI 추정을 "공식 감점", "AI 분석(참고)"으로 구분해 쓰고, 둘이 같은 문제를 가리키면 함께 설명합니다.
3. "## 연습 방법": 위 문제를 줄이기 위한 구체적인 연습 방법.
4. 입력에 없는 수치, 원인, 사건을 만들어내지 마세요. 근거가 부족하면 "데이터로 확인 불가"라고 쓰세요.
"""


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_report_input(section_results, predictions):
    """Merges the official result of each section with its LSTM prediction, keyed by point number."""
    predictions_by_section = {}
    if predictions is not None:
        predictions_by_section = {prediction["section_id"]: prediction for prediction in predictions["sections"]}

    deductions_by_section = {}
    for deduction in section_results["deductions"]:
        deductions_by_section.setdefault(deduction["pointNumber"], []).append({
            "reason": deduction["reason"],
            "description": DEDUCTION_REASON_NAMES.get(deduction["reason"], deduction["reason"]),
            "command": deduction["command"],
            "points": deduction["points"],
            "is_disqualification": deduction["isDisqualification"],
        })

    sections = []
    for section in section_results["sections"]:
        point_number = section["pointNumber"]
        official_deductions = deductions_by_section.get(point_number, [])
        prediction = predictions_by_section.get(point_number)
        # Sections the vehicle never reached have neither samples nor deductions.
        if section["sampleCount"] == 0 and not official_deductions:
            continue

        ai_analysis = None
        if prediction is not None:
            ai_analysis = {
                "probabilities": prediction["probabilities"],
                "flagged_labels": prediction["flagged_labels"],
                "threshold": prediction["threshold"],
            }

        sections.append({
            "point_number": point_number,
            "instruction": section["instructionText"],
            "commands": section["commands"],
            "official_deductions": official_deductions,
            "section_deduction": sum(deduction["points"] for deduction in official_deductions),
            "ai_analysis": ai_analysis,
        })

    return {
        "score": section_results["score"],
        "max_score": section_results["maxScore"],
        "passing_score": section_results["passingScore"],
        "total_deduction": section_results["totalDeduction"],
        "is_passed": section_results["isPassed"],
        "is_disqualified": section_results["isDisqualified"],
        "driving_seconds": section_results["recordedSeconds"],
        "ai_model_available": predictions is not None,
        "sections": sections,
    }


def request_report(report_input, model):
    # Imported here so the data preparation can run and be tested without the OpenAI package.
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before generating a report")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=REPORT_INSTRUCTIONS,
        input=json.dumps(report_input, ensure_ascii=False),
    )
    return response.output_text


def generate_report(session_dir, model=DEFAULT_OPENAI_MODEL, require_predictions=True):
    session_dir = Path(session_dir)
    section_results = load_json(session_dir / SECTION_RESULT_FILE)

    prediction_path = session_dir / PREDICTION_FILE
    predictions = None
    if prediction_path.exists():
        predictions = load_json(prediction_path)
    elif require_predictions:
        raise FileNotFoundError(f"{prediction_path} not found; run session_predict.py or pass --checkpoint")

    report_input = build_report_input(section_results, predictions)
    (session_dir / REPORT_INPUT_FILE).write_text(json.dumps(report_input, ensure_ascii=False, indent=2), encoding="utf-8")

    report_text = request_report(report_input, model)
    report_path = session_dir / REPORT_FILE
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate an OpenAI driving report for a Unity session")
    parser.add_argument("session_dir", help="Session folder exported by ReportManager")
    parser.add_argument("--checkpoint", help="Run LSTM prediction with this checkpoint before writing the report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LABEL_THRESHOLD)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
                        help="OpenAI model name (default: OPENAI_MODEL environment variable or %(default)s)")
    parser.add_argument("--without-predictions", action="store_true",
                        help="Write the report from official deductions only when predictions.json is absent")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.checkpoint:
        predict_session(args.session_dir, args.checkpoint, args.device, args.threshold)

    report_path = generate_report(args.session_dir, args.model, require_predictions=not args.without_predictions)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
