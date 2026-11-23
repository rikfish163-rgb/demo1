import unittest
from .shadow import gripper_position_feedback_valid


class GripperValidityTest(unittest.TestCase):
    def test_busy_and_faults_are_not_position_measurements(self):
        good=dict(gripper_active=1,gripper_fault=0,gripper_fault_id=0,
                  gripper_motiondone=1,gripper_speed=0)
        self.assertTrue(gripper_position_feedback_valid(good))
        for k,v in [('gripper_motiondone',0),('gripper_fault',1),('gripper_fault_id',1),
                    ('gripper_active',0),('gripper_speed',5)]:
            with self.subTest(field=k):
                self.assertFalse(gripper_position_feedback_valid(dict(good,**{k:v})))
        # Contact lost is a valid position report, but not a valid hold signal.
        self.assertTrue(gripper_position_feedback_valid(dict(good,gripper_motiondone=3)))


if __name__=='__main__':unittest.main()
