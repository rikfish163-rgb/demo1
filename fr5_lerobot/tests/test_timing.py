"""Clock-step regressions with synthetic data only; all transports are mocked."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_camera_gemini import GeminiBridgeCamera, GeminiBridgeConfig
from lerobot_camera_gemini.timing import MONOTONIC, camera_clock, command_clock
from lerobot_robot_fairino.continuous_capture import ContinuousCapture
from lerobot_robot_fairino.demo_writer import DemoWriter
from lerobot_robot_fairino.state_stream import FR5StateStream
from fr5_lerobot.tests.test_continuous_capture import FakeRobot
from fr5_lerobot.tests.test_demo_writer import synthetic_sample, synthetic_command
from fr5_lerobot.tests.test_gemini import packet
from fr5_lerobot.tests.test_recording import RecordingTests, synthetic_samples
from lerobot_robot_fairino.verify import verify_dataset
from lerobot_robot_fairino.curate_teacher import curate, SHUTTLE_PHASES
from PIL import Image


def stamped_packet(wall, mono):
    with np.load(BytesIO(packet(timestamp=wall)), allow_pickle=False) as p:
        arrays = {k: p[k] for k in p.files}
    meta = json.loads(arrays['metadata'].item())
    meta['capture']['received_monotonic'] = mono
    arrays['metadata'] = np.array(json.dumps(meta))
    out = BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


class CameraTimingTests(unittest.TestCase):
    def test_wall_backstep_forwardstep_do_not_drop_monotonic_frames(self):
        camera = GeminiBridgeCamera(GeminiBridgeConfig(width=3, height=2))
        self.addCleanup(camera.disconnect)
        camera.connect(warmup=False)
        now = time.monotonic()
        packets = [stamped_packet(wall, now + i * .01 - .05)
                   for i, wall in enumerate((2000., 1999., 2100.))]
        with patch.object(camera._opener, 'open', side_effect=[BytesIO(p) for p in packets]) as opened:
            frames = [camera.read_bundle() for _ in packets]
        self.assertEqual(opened.call_count, 3)
        self.assertEqual([f['metadata']['source_timestamp_unix'] for f in frames], [2000.,1999.,2100.])
        self.assertTrue(all(camera_clock(f['metadata'])[1] == MONOTONIC for f in frames))

    def test_real_staleness_future_and_clock_domain_switch_rejected(self):
        camera = GeminiBridgeCamera(GeminiBridgeConfig(width=3, height=2))
        self.addCleanup(camera.disconnect)
        for mono, error in ((20., TimeoutError), (23., ValueError)):
            with self.subTest(mono=mono), self.assertRaises(error):
                camera.decode_bundle(stamped_packet(1000., mono), received_unix=1000., received_monotonic=22.)
        camera.connect(warmup=False)
        now = time.monotonic()
        with patch.object(camera._opener, 'open', side_effect=[BytesIO(stamped_packet(time.time(), now)), BytesIO(packet())]):
            camera.read_bundle()
            with self.assertRaisesRegex(ValueError, 'clock domain'):
                camera.read_bundle()

    def test_history_matching_uses_monotonic_despite_ambiguous_wall_clock(self):
        stream = FR5StateStream()
        now = time.monotonic()
        stream._socket = object()  # No connect, receive thread or network.
        stream._last_valid_monotonic = now
        stream._history.extend([
            {'timestamp':1000.005,'received_monotonic':now-.2,'frame_counter':1},
            {'timestamp':999.605,'received_monotonic':now-.01,'frame_counter':2}])
        self.assertEqual(stream.nearest(now-.012, clock_basis=MONOTONIC, max_skew_s=.01)['frame_counter'], 2)
        with self.assertRaises(TimeoutError):
            stream.nearest(now-.1, clock_basis=MONOTONIC, max_skew_s=.01)


class DemoTimingTests(unittest.TestCase):
    def test_targets_pair_and_roundtrip_across_clock_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'synthetic_only'
            writer = DemoWriter(root, fps=10, task='SYNTHETIC TIMING ONLY',
                                metadata={'image_width':16,'image_height':12}).start()
            try:
                for i, wall in enumerate((2000.,1999.6,2100.)):
                    sample = synthetic_sample(i)
                    sample['camera']['metadata'].update(source_timestamp_unix=wall, source_timestamp_monotonic=10.+i*.1)
                    sample['robot_state'].update(timestamp=wall+.001, received_monotonic=10.+i*.1+.001)
                    sample['sync_skew_s'] = .001
                    sample['sync_clock'] = MONOTONIC
                    cmd = synthetic_command(sample, i)
                    # Raw wall time appears in the future after the step;
                    # its actual monotonic issue precedes acquisition.
                    cmd.update(issued_unix=2000., issued_monotonic=10.+i*.1-.01)
                    writer.add(sample, cmd)
                report = writer.finish('success')
                self.assertTrue(report['data_valid'])
                self.assertEqual(report['timing_clock'], MONOTONIC)
                ds = LeRobotDataset(repo_id=report['repo_id'], root=root)
                self.assertEqual(ds[1]['observation.source_time_us'][0].item(),1999600000)
                self.assertEqual(ds[1]['observation.source_monotonic_us'][0].item(),10100000)
                self.assertEqual(ds[1]['action.source_monotonic_us'].item(),10090000)
                self.assertEqual(ds[1]['observation.monotonic_timing_valid'].item(),1)
                future = deepcopy(cmd)
                future['issued_monotonic'] += .02
                with self.assertRaisesRegex(ValueError,'Future action'):
                    writer._prepare(sample,future)
                del future['issued_monotonic']
                with self.assertRaises(KeyError):
                    command_clock(future,MONOTONIC)
            finally:
                if not writer._finished:
                    writer.finish('interrupted')


class MonotonicRobot(FakeRobot):
    offset = 0.
    gap = False

    def get_sample(self):
        sample = super().get_sample()
        mono = time.monotonic() + (1. if self.gap else 0.)
        sample['camera']['metadata'].update(source_timestamp_monotonic=mono)
        sample['camera']['metadata']['source_timestamp_unix'] += self.offset
        sample['robot_state'].update(timestamp=sample['camera']['metadata']['source_timestamp_unix']+.001,
                                     received_monotonic=mono+.001)
        sample['sync_clock'] = MONOTONIC
        return sample


class CaptureTimingTests(unittest.TestCase):
    def test_sampling_wait_never_uses_realtime_event_timeout(self):
        from unittest.mock import Mock, patch
        from lerobot_robot_fairino.continuous_capture import _wait_until_monotonic
        stop=Mock()
        stop.is_set.side_effect=[False,False,False,False]
        stop.wait.side_effect=AssertionError('CLOCK_REALTIME wait must not run')
        with patch('lerobot_robot_fairino.continuous_capture.time.monotonic',
                   side_effect=[10.,10.02,10.04,10.05]), patch(
                   'lerobot_robot_fairino.continuous_capture.time.sleep') as sleep:
            _wait_until_monotonic(stop,10.05)
        stop.wait.assert_not_called()
        self.assertEqual(sleep.call_count,3)
        self.assertTrue(all(0 < c.args[0] <= .02 for c in sleep.call_args_list))
        stop.is_set.side_effect=None;stop.is_set.return_value=True
        with patch('lerobot_robot_fairino.continuous_capture.time.sleep') as sleep:
            _wait_until_monotonic(stop,999.)
        sleep.assert_not_called()

    def test_clock_step_survives_but_actual_gap_remains_sticky_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'synthetic_only'
            robot = MonotonicRobot()
            c = ContinuousCapture(robot,root,fps=30,task='SYNTHETIC TIMING ONLY').start()
            def wait_for(predicate):
                deadline = time.monotonic()+1
                while not predicate() and time.monotonic()<deadline:
                    time.sleep(.005)
                self.assertTrue(predicate())
            try:
                phase = c.phase('clock_step_test')
                robot.offset=-50.
                wait_for(lambda: len(c._times)>=3)
                c.assert_healthy()
                robot.offset=50.
                wait_for(lambda: len(c._times)>=5)
                c.assert_healthy()
                robot.gap=True
                wait_for(lambda: c._error is not None)
                with self.assertRaisesRegex(RuntimeError,'frame gap'):
                    c.assert_healthy()
                report=c.finish('success')
                self.assertEqual(report['outcome'],'interrupted')
                self.assertEqual(report['timing_clock'],MONOTONIC)
                rejected=json.loads((root/'capture-rejection.json').read_text())
                self.assertGreater(rejected['last_attempt']['attempted_time']-rejected['last_attempt']['previous_recorded_time'],.5)
                rows=list(map(json.loads,(root/'capture.jsonl').read_text().splitlines()))
                self.assertEqual(rows[-1]['phase_event_id'],phase['event_id'])
            finally:
                if c._report is None:
                    c.finish('interrupted')


class RecordingClockStepTests(RecordingTests):
    def test_observation_verifier_uses_monotonic_but_preserves_wall_evidence(self):
        samples=synthetic_samples()
        mono=time.monotonic()-10
        for i,sample in enumerate(samples):
            sample['camera']['metadata']['source_timestamp_monotonic']=mono+i*.5
            sample['robot_state']['received_monotonic']=mono+i*.5+.01
            sample['sync_clock']=MONOTONIC
        samples[1]['camera']['metadata']['source_timestamp_unix']-=1
        samples[1]['robot_state']['timestamp']-=1
        report=self.run_record(samples)
        self.assertEqual(report['timing_clock'],MONOTONIC)
        self.assertTrue(verify_dataset(self.root)['passed'])


class CurationTimingTests(unittest.TestCase):
    def test_six_phases_future_target_horizon_and_clock_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'synthetic_source'
            (root/'lerobot/meta').mkdir(parents=True)
            (root/'continuous').mkdir()
            import hashlib
            from lerobot_robot_fairino.human_review import record_review
            (root/'scope.json').write_text(json.dumps({'split_role':'training_complete_trial','phase_order':SHUTTLE_PHASES}))
            (root/'review-evidence.jpg').write_bytes(b'SYNTHETIC')
            (root/'summary.json').write_text(json.dumps({'outcome':'pending_review','released':True,'failed':None,
                'evidence':['review-evidence.jpg'],'completed_stages':SHUTTLE_PHASES,
                'continuous_capture':{'capture_complete':True},
                'scope_sha256':hashlib.sha256((root/'scope.json').read_bytes()).hexdigest()}))
            record_review(root,verdict='success',user_statement='SYNTHETIC TEST ONLY')
            journal=[{'sequence':1,'issued_unix':2000.,'issued_monotonic':9.,'kind':'hold','executed':False,'target':[0.]*6+[100.]}]
            rows=[]
            image_rows=[]
            Image.new('RGB',(640,360)).save(root/'continuous/synthetic.png')
            for p,phase in enumerate(SHUTTLE_PHASES):
                mono=10.+p
                wall=2000.-p  # Unix ordering is deliberately reversed.
                previous=journal[-1]['target']
                if phase in ('close','release'):
                    target=previous[:6]+[76. if phase=='close' else 100.]
                    method,kind,args='MoveGripper','gripper',[1,target[6]]
                else:
                    target=[float(p)]*6+[previous[6]]
                    method,kind,args='MoveL','arm',[target[:6]]
                cmd={'sequence':p+2,'issued_unix':wall+.05,'issued_monotonic':mono+.05,
                     'kind':kind,'executed':True,'target':target}
                journal.append(cmd)
                (root/f'command-{p+2:04d}.json').write_text(json.dumps({
                    'command':cmd,'status':'completed','outcome':{'return_code':0},
                    'method':method,'args':args,'phase':phase,
                    'completed_at_unix':wall+.2,'completed_monotonic':mono+.2}))
                for j in range(3):
                    sample=synthetic_sample(len(rows))
                    camera=sample['camera']['metadata']
                    camera.update(source_timestamp_unix=wall+j/6,source_timestamp_monotonic=mono+j/6)
                    row={'frame_index':len(rows),'camera':camera,'robot':sample['robot_state']}
                    rows.append(row)
                    image_rows.append({**row,'rgb_file':'synthetic.png'})
            (root/'target-command-journal.json').write_text(json.dumps(journal))
            (root/'lerobot/meta/capture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            image_log=root/'continuous/capture.jsonl'
            image_log.write_text(''.join(json.dumps(r)+'\n' for r in image_rows))
            output=Path(tmp)/'synthetic_output'
            report=curate(root,output)
            self.assertTrue(report['complete_physical_episode'])  # Synthetic user label only.
            self.assertEqual(report['frames'],18)
            self.assertEqual(report['phase_order'],SHUTTLE_PHASES)
            ds=LeRobotDataset('local/fr5_teacher_pilot',root=output)
            for p in range(6):
                frame=ds[p*3]
                self.assertEqual(frame['observation.state'].shape,(14,))
                self.assertEqual(frame['observation.state'][8+p].item(),1.)
                # Command issued50ms after image is inside+1/6s horizon.
                self.assertEqual(frame['action.command_sequence'].item(),p+2)
            del image_rows[0]['camera']['source_timestamp_monotonic']
            image_log.write_text(''.join(json.dumps(r)+'\n' for r in image_rows))
            with self.assertRaisesRegex(ValueError,'clock domains differ'):
                curate(root,Path(tmp)/'invalid_clock_output')


if __name__=='__main__':
    unittest.main()
