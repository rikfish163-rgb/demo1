"""Causal measured history and model-bound context transport, without devices."""
from collections import deque
from pathlib import Path
import math,subprocess,sys,tempfile,threading,time,unittest
from unittest.mock import Mock,patch
import numpy as np
from fr5_lerobot.tests.test_demo_writer import synthetic_sample
from lerobot_robot_fairino.stream_recording import StreamEvaluationRecording
from lerobot_robot_fairino.stream_policy import StreamPolicy,make_request,validate_reply
from lerobot_robot_fairino.stream_runtime import InferenceWorker
from lerobot_robot_fairino.stream_contract import NAME


def sample(index,stamp):
    value=synthetic_sample(index)
    value['camera']['color_rgb']=np.zeros((360,640,3),dtype=np.uint8)
    value['camera']['metadata']['source_timestamp_monotonic']=stamp
    value['robot_state'].update(received_monotonic=stamp+.004,gripper_position=100,
        gripper_active=1,gripper_fault=0,gripper_fault_id=0,gripper_motiondone=1,gripper_speed=0)
    return value


def recorder(profile):
    value=object.__new__(StreamEvaluationRecording)
    value._sample_lock=threading.Lock();value._sample_ready=threading.Condition(value._sample_lock)
    value._latest=None;value._frames=0;value._state_history=deque(maxlen=128)
    value.observation_context_profile=profile;value.assert_healthy=Mock()
    return value


class RuntimeHistoryTests(unittest.TestCase):
    def test_compressed_transport_reuses_connection_and_preserves_history(self):
        profile = 'state-history-physical-v1'
        digest = 'a' * 64
        ready = {'status': 'ready', 'schema': 'fr5-stream-ready-v1',
                 'checkpoint_sha256': digest, 'policy_contract': NAME,
                 'state_shape': [8], 'action_shape': [50, 7], 'n_action_steps': 10,
                 'observation_context_profile': profile}
        actions = [[i / 100.] * 6 + [.75] for i in range(50)]
        response = {'schema': 'fr5-stream-reply-v1', 'checkpoint_sha256': digest,
                    'policy_contract': NAME, 'observation_context_profile': profile,
                    'actions': actions, 'sampling': {'cuda_rng_state': [1, 2, 3],
                                                       'denoising_steps': 10}}
        server = ('import json,sys\n'
                  f'ready={ready!r}\nresponse={response!r}\n'
                  'sys.stdout.write(json.dumps(ready)+"\\n");sys.stdout.flush()\n'
                  'for line in sys.stdin:\n'
                  ' request=json.loads(line)\n'
                  ' result=dict(response,id=request["id"],echo=request)\n'
                  ' sys.stdout.write(json.dumps(result)+"\\n");sys.stdout.flush()\n')
        calls = []

        def process_factory(command, **kwargs):
            calls.append(command)
            return subprocess.Popen([sys.executable, '-u', '-c', server], **kwargs)

        root = '/home/data4/hts/demo1/fr5-learning'
        policy = StreamPolicy(root + '/checkpoints/model', root + '/src/service',
                              gpu=4, expected_sha=digest, process_factory=process_factory)
        try:
            policy.wait_ready()
            for index in range(2):
                request = {'id': f'frame-{index}', 'task': '抓杯',
                           'source_time': 100. + index, 'source_clock': 'host_monotonic',
                           'state': [0.] * 8, 'rgb_jpeg_base64': 'AA==',
                           'observation_context_profile': profile,
                           'observation_history': [{'age_s': float(i), 'state': [0.] * 8}
                                                   for i in range(1, 7)]}
                result = policy.predict(request)
                self.assertEqual(result['echo'], request)
                self.assertEqual(result['actions'], actions)
                self.assertEqual(result['sampling'], response['sampling'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:3], ['ssh', '-T', '-C'])
        finally:
            policy.close()
            policy.process.stdout.close()

    def test_physical_history_profile_is_bound_to_ready_request_and_reply(self):
        profile = 'state-history-physical-v1'
        rec = recorder(profile)
        rec._sample(sample(0, time.monotonic()))
        request = make_request(rec.observation(), '抓杯并停留5秒', observation_context_profile=profile)
        self.assertEqual(request['observation_history'], [None] * 6)
        policy = object.__new__(StreamPolicy)
        policy.expected_sha = None
        policy.ready = False
        policy.receive = Mock(return_value={'status': 'ready', 'schema': 'fr5-stream-ready-v1',
            'checkpoint_sha256': 'a' * 64, 'policy_contract': NAME, 'state_shape': [8],
            'action_shape': [50, 7], 'n_action_steps': 10, 'observation_context_profile': profile})
        policy.wait_ready()
        self.assertEqual(policy.observation_context_profile, profile)
        reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'],
                 'checkpoint_sha256': 'a' * 64, 'policy_contract': NAME,
                 'actions': [[0.] * 6 + [1.]] * 50, 'observation_context_profile': profile}
        self.assertEqual(len(validate_reply(reply, request, 'a' * 64)), 50)

    def test_history_keeps_measured_invalid_state_and_actual_ages(self):
        rec=recorder('state-history-v1');base=time.monotonic()-9
        for index in range(10):
            value=sample(index,base+index)
            if index==7:value['robot_state']['gripper_motiondone']=0
            rec._sample(value)
        value=rec.observation();request=make_request(value,'抓杯并停留5秒',observation_context_profile='state-history-v1')
        history=request['observation_history']
        self.assertEqual(request['task'],'抓杯并停留5秒')
        self.assertEqual(len(request['state']),8)
        self.assertAlmostEqual(history[0]['age_s'],1.)
        self.assertAlmostEqual(history[0]['state'][0],math.radians(9.))
        self.assertEqual(history[2]['state'][6:],[0.,0.])
        self.assertAlmostEqual(history[-1]['age_s'],8.)
        with rec._sample_lock:
            old=rec._observation_with_history(sample(8,base+8))
        self.assertAlmostEqual(old['observation_history'][0]['state'][0],math.radians(8.))

    def test_new_episode_has_missing_history_and_legacy_input_stays_unchanged(self):
        rec=recorder('state-history-v1');rec._sample(sample(0,time.monotonic()))
        value=rec.observation()
        self.assertEqual(value['observation_history'],[None]*6)
        current=make_request(value,'抓杯',observation_context_profile='state-history-v1')
        legacy=make_request(value,'抓杯')
        self.assertEqual(current['state'],legacy['state'])
        self.assertNotIn('observation_history',legacy)
        self.assertNotIn('observation_context_profile',legacy)
        other=recorder('none');other._sample(sample(0,time.monotonic()))
        self.assertNotIn('observation_history',other.observation())
        self.assertEqual(len(other._state_history),0)

    def test_ready_reply_and_request_use_the_same_context_profile(self):
        policy=object.__new__(StreamPolicy);policy.expected_sha=None;policy.ready=False
        policy.receive=Mock(return_value={'status':'ready','schema':'fr5-stream-ready-v1',
            'checkpoint_sha256':'a'*64,'policy_contract':NAME,'state_shape':[8],
            'action_shape':[50,7],'n_action_steps':10,'observation_context_profile':'state-history-v1'})
        policy.wait_ready();self.assertEqual(policy.observation_context_profile,'state-history-v1')
        request={'id':'current','observation_context_profile':'state-history-v1'}
        reply={'schema':'fr5-stream-reply-v1','id':'current','checkpoint_sha256':'a'*64,
               'policy_contract':NAME,'actions':[[0.]*6+[1.]]*50}
        with self.assertRaises(ValueError):validate_reply(reply,request,'a'*64)
        reply['observation_context_profile']='state-history-v1'
        self.assertEqual(len(validate_reply(reply,request,'a'*64)),50)
        policy.process=Mock()
        with self.assertRaisesRegex(ValueError,'context differs'):policy.predict({'id':'old-without-history'})
        policy.process.stdin.write.assert_not_called()

    def test_worker_passes_the_loaded_history_profile(self):
        request={'id':'current','source_time':100.,'source_clock':'host_monotonic',
                 'state':[0.]*6+[1.,1.],
                 'rgb_jpeg_base64':'AA==','observation_context_profile':'state-history-v1',
                 'observation_history':[None]*6}
        reply={'schema':'fr5-stream-reply-v1','id':'current','checkpoint_sha256':'a'*64,
               'policy_contract':NAME,'actions':[[0.]*6+[1.]]*50,
               'observation_context_profile':'state-history-v1'}
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=10,observation_context_profile='state-history-v1')
        policy.predict.return_value=reply;rec=Mock()
        with tempfile.TemporaryDirectory() as root,patch('lerobot_robot_fairino.stream_runtime.make_request',return_value=request) as make:
            worker=InferenceWorker(policy,rec,'抓杯',Path(root));worker.start()
            try:
                deadline=time.monotonic()+2
                while worker.latest() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(worker.latest())
                self.assertEqual(make.call_args.kwargs,{'observation_context_profile':'state-history-v1'})
                rec.record_prediction.assert_called_once()
            finally:worker.stop();worker.thread.join(2)

if __name__=='__main__':unittest.main()
