"""Controller path sampling includes stationary attitude changes."""
import unittest
from unittest.mock import patch

import numpy as np

from fr5_jev import prepare_varied_session as preparation


class RotationController:
    def __init__(self, blocked_yaw=None):
        self.blocked_yaw = blocked_yaw
        self.sampled_yaw = []

    def GetJointSoftLimitDeg(self, flag):
        return [0, *([-180., 180.] * 6)]

    def GetInverseKinRef(self, mode, pose, reference):
        yaw = pose[5]
        self.sampled_yaw.append(yaw)
        if self.blocked_yaw and self.blocked_yaw[0] <= yaw <= self.blocked_yaw[1]:
            return [112]
        return [0, 0., 0., 0., 0., 90., yaw]

    def GetForwardKin(self, joints):
        return [0, 0., 0., 200., 90., 0., joints[5]]


class PathGeometryTests(unittest.TestCase):
    def sample(self, controller):
        route = {'approach_clear': [[0., 0., 200., 90., 0., 0.],
                                    [0., 0., 200., 90., 0., 64.]]}
        chain = np.array([[0., 0., 200.]] * 7)
        with patch.object(preparation, 'controller', return_value=controller), \
                patch.object(preparation, 'model_chain', return_value=chain):
            return preparation.sample_geometry(route, [0., 0., 0., 0., 90., 0.])

    def test_pure_rotation_has_continuous_joint_steps(self):
        rows, summary = self.sample(RotationController())
        self.assertGreater(len(rows), 200)
        self.assertLessEqual(summary['maximum_joint_step_deg'], 0.25)
        np.testing.assert_allclose(rows[-1]['tcp'], [0., 0., 200., 90., 0., 64.])

    def test_endpoint_solutions_do_not_hide_an_invalid_interior_attitude(self):
        controller = RotationController(blocked_yaw=(19., 20.))
        with self.assertRaisesRegex(RuntimeError, 'IK failed'):
            self.sample(controller)
        self.assertTrue(any(19. <= yaw <= 20. for yaw in controller.sampled_yaw))


if __name__ == '__main__':
    unittest.main()
