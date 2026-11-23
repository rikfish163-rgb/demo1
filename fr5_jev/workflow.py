"""Reusable read-only model/LLM workflow. This CLI has no physical execution mode."""
import argparse
import json
from pathlib import Path

from .decision import make_request, validate_answer
from .experiment import decide, save_new
from .memory import retrieve
from .observe import capture
from .policy import assess, thresholds
from .shadow import collect_live_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "observe", "review", "replay"))
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--call-jev", action="store_true", help="Call OpenAI Decisions with gpt-6-luna")
    parser.add_argument("--record", type=Path, help="Existing decision record, for replay only")
    parser.add_argument("--context-file", type=Path, help="LLM annotations for review mode only; data, never commands")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; use a new directory")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "replay":
        if args.record is None or args.call_jev:
            parser.error("Replay needs --record and does not call the model")
        record = json.loads(args.record.read_text())
        detail = record.get("result", record)
        provider = record.get("provider", detail.get("provider", "emrank"))
        request = record.get("request", detail.get("request"))
        reply = record.get("response", detail.get("response"))
        if reply is None:
            if provider == "openai":
                parser.error("OpenAI replay requires the recorded raw response")
            reply = {"answers": {"next_skill": detail["jev_answer"]}}
            if "response_model" in detail:
                reply["model"] = detail["response_model"]
        answer = validate_answer(request, reply, provider=provider)
        dynamic = assess(answer, thresholds(answer["choice"]))
        old_pass = answer["confidence"] >= .8 and answer["probabilities"][answer["choice"]] >= .85
        result = {"mode": "historical_counterfactual_replay", "source": str(args.record),
                  "provider": provider,
                  "old_fixed_gate_passed": old_pass, "dynamic_gate": dynamic,
                  "not_an_independent_accuracy_evaluation": True, "model_called": False,
                  "motion_commands_sent": 0, "gripper_commands_sent": 0}
        save_new(args.output/"replay.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return
    state = collect_live_state()
    state["retrieved_examples"] = retrieve(["fr5", "noncontact", "pregrasp"])
    state["candidates"] = {}  # Read-only workflow cannot offer robot actions.
    if args.mode == "observe":
        state["facts"].update(current_subtask="Refresh synchronized RGB-D and robot feedback",
                                pending_sensing_work=True)
    elif args.mode == "review":
        state["facts"]["current_subtask"] = "Determine whether the next task requires LLM strategy or geometry review"
        if args.context_file:
            state["llm_annotations"] = json.loads(args.context_file.read_text())
    if args.mode == "status" or not args.call_jev:
        save_new(args.output/"state.json", state)
        save_new(args.output/"prepared-request.json", make_request(state, provider="openai"))
        print(json.dumps({"status": "prepared_only", "retrieved_cases": len(state["retrieved_examples"]),
                          "model_called": False, "physical_commands_sent": 0}, ensure_ascii=False))
        return
    record, path = decide(state, "decision", directory=args.output, compact=True)
    outcome = {"status": record["status"], "decision_record": str(path), "gate": record.get("gate"),
               "physical_commands_sent": 0}
    if (record.get("gate", {}).get("route") == "selected"
            and record["gate"]["choice"] == "observe_scene"):
        capture(args.output/"observation", decision_record=str(path))
        outcome["readonly_observation_completed"] = True
    save_new(args.output/"outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False))


if __name__ == "__main__":
    main()
