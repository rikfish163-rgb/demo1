"""Risk gate invariants. No real robot or model calls."""
import unittest
from .policy import assess, thresholds


def answer(choice, p=.98, confidence=.98):
    return {"choice": choice, "confidence": confidence,
            "probabilities": {choice: p, "hold" if choice != "hold" else "observe_scene": 1-p}}


CHECKS = {k: True for k in ["specific_authorization", "onsite_confirmation", "preflight_passed",
                           "scene_reviewed", "bounded_executor_verified"]}


class RiskGateTests(unittest.TestCase):
    def test_user_requested_empty_preparation_gate_is_narrow(self):
        a = answer("prepare_empty_gripper", .80, .73)
        self.assertEqual(assess(a, thresholds(a["choice"]), hard_checks=CHECKS)["route"], "selected")
        for key in CHECKS:
            self.assertEqual(assess(a, thresholds(a["choice"]),
                hard_checks={**CHECKS, key: False})["route"], "blocked")
        for skill in ("align_xy", "descend_pregrasp", "close_gripper", "lift_verify"):
            self.assertEqual(assess(answer(skill, .80, .73), thresholds(skill),
                hard_checks=CHECKS)["route"], "needs_llm")

    def test_low_gate_applies_only_to_reviewed_empty_free_space(self):
        for skill in ("orient_empty_tool", "transfer_empty_tool", "probe_empty_gripper"):
            a = answer(skill, .55, .12)
            self.assertEqual(assess(a, thresholds(skill), hard_checks=CHECKS)["route"], "selected")
            for key in CHECKS:
                self.assertEqual(assess(a, thresholds(skill),
                    hard_checks={**CHECKS, key: False})["route"], "blocked")
            self.assertEqual(assess(a, thresholds(skill, evidence_complete=False),
                                    hard_checks=CHECKS)["route"], "blocked")
        self.assertEqual(assess(answer("align_xy", .55, .12), thresholds("align_xy"),
                                hard_checks=CHECKS)["route"], "needs_llm")

    def test_empty_gripper_preparation_requires_physical_checks(self):
        skill = "prepare_empty_gripper"
        self.assertEqual(thresholds(skill)["risk"], "noncontact_motion")
        self.assertEqual(assess(answer(skill), thresholds(skill))["route"], "blocked")
        self.assertEqual(assess(answer(skill), thresholds("observe_scene"),
                               hard_checks=CHECKS)["route"], "blocked")

    def test_explicit_user_default_is_a_recorded_assumption_only_for_empty_motion(self):
        checks = {**CHECKS, "onsite_confirmation": False, "user_default_normal": True}
        a = answer("transfer_empty_tool", .55, .12)
        result = assess(a, thresholds(a["choice"]), hard_checks=checks)
        self.assertEqual(result["route"], "selected")
        self.assertEqual(result["workspace_basis"], "user_default_normal_not_independently_verified")
        probe = answer("probe_empty_gripper", .55, .12)
        self.assertEqual(assess(probe, thresholds(probe["choice"]), hard_checks=checks)["route"], "selected")
        self.assertEqual(assess(answer("close_gripper"), thresholds("close_gripper"),
                                hard_checks=checks)["route"], "blocked")
        checks["user_default_normal"] = False
        self.assertEqual(assess(a, thresholds(a["choice"]), hard_checks=checks)["route"], "blocked")

    def test_low_risk_observation_can_pass_without_motion_permission(self):
        a = answer("observe_scene", .75, .62)
        self.assertEqual(assess(a, thresholds("observe_scene"))["route"], "selected")

    def test_same_scores_do_not_pass_motion_gate(self):
        self.assertEqual(assess(answer("descend_pregrasp", .75, .62),
            thresholds("descend_pregrasp"), hard_checks=CHECKS)["route"], "needs_llm")

    def test_confidence_never_supplies_missing_authorization(self):
        checks = {**CHECKS, "specific_authorization": False}
        self.assertEqual(assess(answer("descend_pregrasp"), thresholds("descend_pregrasp"),
            hard_checks=checks)["route"], "blocked")

    def test_every_physical_hard_check_is_required(self):
        for key in CHECKS:
            checks = {**CHECKS, key: False}
            with self.subTest(key=key):
                self.assertEqual(assess(answer("align_xy"), thresholds("align_xy"),
                    hard_checks=checks)["route"], "blocked")

    def test_success_history_does_not_relax_contact_or_motion(self):
        for skill in ("orient_empty_tool", "align_xy", "descend_pregrasp", "close_gripper", "lift_verify"):
            self.assertEqual(thresholds(skill)["min_probability"],
                             thresholds(skill, matched_successes=100)["min_probability"])

    def test_repeated_readonly_success_and_failure_adjustments(self):
        self.assertLess(thresholds("observe_scene", matched_successes=5)["min_probability"],
                        thresholds("observe_scene")["min_probability"])
        self.assertGreater(thresholds("observe_scene", recent_failures=1)["min_probability"],
                           thresholds("observe_scene")["min_probability"])

    def test_readonly_policy_cannot_be_supplied_for_motion(self):
        self.assertEqual(assess(answer("align_xy"), thresholds("observe_scene"),
            hard_checks=CHECKS)["route"], "blocked")

    def test_motion_policy_cannot_be_edited_below_declared_floor(self):
        policy = thresholds("descend_pregrasp")
        policy["min_probability"] = .1
        self.assertEqual(assess(answer("descend_pregrasp"), policy,
            hard_checks=CHECKS)["route"], "blocked")

    def test_uncertain_evidence_blocks_motion_even_when_confident(self):
        self.assertEqual(assess(answer("align_xy", .999, .999),
            thresholds("align_xy", evidence_complete=False), hard_checks=CHECKS)["route"], "blocked")

    def test_astra_request_does_not_dispatch_a_skill(self):
        self.assertEqual(assess(answer("request_astra"), thresholds("request_astra"))["route"], "needs_llm")


if __name__ == "__main__":
    unittest.main()
