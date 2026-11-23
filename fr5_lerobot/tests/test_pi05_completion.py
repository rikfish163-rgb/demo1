"""Synthetic gripper completion regression; no devices or processes."""
import unittest
from unittest.mock import patch
from lerobot_robot_fairino.pi05_rollout import PolicyController


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.s=dict(gripper_active=1,gripper_fault=0,gripper_fault_id=0,
                    gripper_motiondone=1,gripper_speed=0,gripper_position=87)
        self.c=object.__new__(PolicyController)

    def done(self,now=101):
        with patch('lerobot_robot_fairino.action_control.time.monotonic',return_value=now):
            return self.c._gripper_complete(self.s,87,100)

    def test_position_at_target_while_busy_is_not_complete(self):
        self.s['gripper_motiondone']=0
        self.assertFalse(self.done())

    def test_precommand_cached_match_is_not_complete(self):
        self.assertFalse(self.done(100.1))
        self.assertTrue(self.done())

    def test_contact_completion_requires_valid_finished_feedback(self):
        self.s.update(gripper_position=90,gripper_motiondone=2)
        self.assertTrue(self.done())
        self.s['gripper_active']=0
        self.assertFalse(self.done())

    def test_lost_contact_or_moving_is_not_complete(self):
        self.s['gripper_motiondone']=3
        self.assertFalse(self.done())
        self.s.update(gripper_motiondone=1,gripper_speed=1)
        self.assertFalse(self.done())


if __name__=='__main__':unittest.main()
