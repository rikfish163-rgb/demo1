"""Whole-episode orchestration tests, without a robot, camera or model network."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from .continuous_workflow import (AUTOMATIC, PHASE_SKILLS, VISUAL_FACTS,
                                 ContinuousWorkflow, selected_for_phase, validate_automatic_review)
from .decision import make_request
from .policy import gate_table


class FakeSession:
    def __init__(self, root, provider='emrank'):
        self.root=root;self.scope_sha='synthetic-scope';self.completed_stages=[]
        self.released=False;self.command_sequence=0;self.phase='initial_hold'
        self.uncertain=None;self.fail=None;self.finishes=[]
        self.provider=provider
    def state(self, **kwargs):return {'tcp':[0.]*6}
    def decide(self, phase, filename):
        choice=PHASE_SKILLS[phase]
        state={'schema':'fr5-cup-shadow-v1','source':'synthetic_test',
               'robot_id':'J1','robot_ip':'192.168.58.2','captured_at_unix':time.time(),
               'robot':{'stationary':True,'fault_free':True},'facts':{'requested_phase':phase},
               'candidates':{choice:{'validated':True,'preconditions_met':True,'evidence':'synthetic'}}}
        request=make_request(state,provider=self.provider)
        choices=([entry['value'] for entry in request['questions'][0]['choices']]
                 if self.provider=='openai' else request['questions']['next_skill']['criteria'])
        probabilities={k:.01/(len(choices)-1) for k in choices};probabilities[choice]=.99
        answer={'type':'choice','choice':choice,'confidence':.01 if phase==self.uncertain else .99}
        if self.provider=='openai':
            response={'model':request['model'],'answers':[{**answer,'name':'next_skill',
                'probabilities':[{'value':label,'probability':value}
                                 for label,value in probabilities.items()]}]}
        else:
            model='typesafe/jev-latest' if self.provider=='emrank' else request['model']
            response={'model':model,'answers':{'next_skill':{**answer,'probabilities':probabilities}}}
        record={'provider':self.provider,'scope_sha256':self.scope_sha,
                'state':state,'request':request,'response':response,
                'policies_frozen_before_inference':gate_table(choices)}
        (self.root/filename).write_text(json.dumps(record));return {'latency_ms':1.}
    def review_phase(self, phase, filename):
        self.assert_next(phase);record=json.loads((self.root/filename).read_text())
        if not record.get('current_chat_llm_review'):validate_automatic_review(self,phase,record)
        self.phase=phase
    def assert_next(self,phase):
        if list(PHASE_SKILLS)[len(self.completed_stages)]!=phase:raise ValueError('Phase order')
    def record_prealigned_entry(self,phase):
        self.assert_next(phase);self.completed_stages.append(phase)
    def run_phase(self):
        if self.phase==self.fail:raise RuntimeError('Synthetic telemetry failure')
        self.completed_stages.append(self.phase);self.command_sequence+=1
        if self.phase=='release':self.released=True
    def finish(self,outcome,evidence):
        self.finishes.append(outcome)
        assert self.completed_stages==list(PHASE_SKILLS)
        self.metrics_at_finish=json.loads((self.root/'workflow-metrics.json').read_text())
        return {'outcome':outcome}


def observe(root):
    root.mkdir();(root/'color.jpg').write_bytes(b'SYNTHETIC IMAGE')
    report={'robot_after':{'tcp':[0.]*6}}
    (root/'observation.json').write_text(json.dumps(report));return report


def review(request,session):
    return {'phase':request['phase'],'scope_sha256':request['scope_sha256'],
            'image_sha256':request['image_sha256'],'proceed':True,'reason':'synthetic reviewed facts',
            'resolve_decision_exception':True,**{key:True for key in request['required_facts']}}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.session=FakeSession(self.root);self.requests=[]
        self.net=patch('socket.create_connection',side_effect=AssertionError('No network'))
        self.net.start();self.addCleanup(self.net.stop)
    def workflow(self,callback=review):
        def reviewer(request,session):self.requests.append(request);return callback(request,session)
        return ContinuousWorkflow(self.session,observe=observe,request_review=reviewer,emit=lambda *a,**k:None)
    def test_full_episode_advances_all_phases_and_only_waits_for_visual_facts(self):
        flow=self.workflow();result=flow.run()
        self.assertEqual(self.session.completed_stages,list(PHASE_SKILLS))
        self.assertEqual([r['phase'] for r in self.requests],list(VISUAL_FACTS))
        self.assertEqual(flow.metrics['llm_checkpoints'],6)
        self.assertEqual(flow.metrics['user_action_approval_requests'],0)
        self.assertEqual(result,{'outcome':'pending_review'})
        self.assertEqual(self.session.finishes,['pending_review'])
        self.assertEqual(self.session.metrics_at_finish['outcome'],'pending_user_whole_episode_review')
        with self.assertRaises(FileExistsError):flow.run()
    def test_openai_episode_uses_original_wire_records_and_generic_latency(self):
        self.session=FakeSession(self.root,provider='openai')
        flow=self.workflow();flow.run()
        self.assertEqual(self.session.completed_stages,list(PHASE_SKILLS))
        self.assertEqual([r['phase'] for r in self.requests],list(VISUAL_FACTS))
        record=json.loads((self.root/'workflow_approach_clear.json').read_text())
        self.assertNotIn('state',record['request'])
        self.assertEqual(record['request']['input'][0]['content'][0]['data'],record['state'])
        self.assertEqual(record['response']['model'],'gpt-6-luna')
        for row in flow.metrics['phases']:
            self.assertEqual(row['provider'],'openai')
            self.assertEqual(row['decision_latency_ms'],1.)
            self.assertNotIn('jev_latency_ms',row)
    def test_legacy_record_without_provider_remains_emrank(self):
        self.session.decide('approach_clear','legacy.json')
        record=json.loads((self.root/'legacy.json').read_text());record.pop('provider')
        self.assertTrue(selected_for_phase(record,'approach_clear',self.session.scope_sha))
    def test_phase_selection_validates_record_provider_and_reported_model(self):
        for provider in ('openai','emrank','typesafe'):
            with self.subTest(provider=provider):
                session=FakeSession(self.root,provider=provider)
                session.decide('approach_clear',provider+'.json')
                record=json.loads((self.root/(provider+'.json')).read_text())
                self.assertTrue(selected_for_phase(record,'approach_clear',session.scope_sha))
                wrong=copy.deepcopy(record)
                wrong['response']['model']='jev-1.13.0' if provider=='openai' else 'gpt-6-luna'
                self.assertFalse(selected_for_phase(wrong,'approach_clear',session.scope_sha))
                wrong=copy.deepcopy(record)
                wrong['provider']='emrank' if provider=='openai' else 'openai'
                self.assertFalse(selected_for_phase(wrong,'approach_clear',session.scope_sha))
    def test_openai_phase_selection_binds_the_original_input_state(self):
        self.session=FakeSession(self.root,provider='openai')
        self.session.decide('approach_clear','openai.json')
        record=json.loads((self.root/'openai.json').read_text())
        changed=copy.deepcopy(record)
        changed['request']['input'][0]['content'][0]['data']['facts']['description']='different'
        self.assertFalse(selected_for_phase(changed,'approach_clear',self.session.scope_sha))
        changed=copy.deepcopy(record)
        changed['request']['state']=changed['state']
        changed['request']['input']=[]
        self.assertFalse(selected_for_phase(changed,'approach_clear',self.session.scope_sha))
    def test_uncertain_jev_routes_to_llm_without_retry(self):
        self.session.uncertain='approach_clear';flow=self.workflow();flow.run()
        self.assertEqual(len(self.requests),7)
        self.assertTrue(self.requests[0]['decision_exception'])
    def test_prealigned_episode_closes_without_replaying_an_approach(self):
        self.session.scope={'prealigned_start':True}
        self.workflow().run()
        self.assertEqual(self.session.command_sequence,8)
        self.assertEqual(self.requests[0]['phase'],'close')
        self.assertFalse((self.root/'workflow_approach_clear.json').exists())
        self.assertEqual(self.session.finishes,['pending_review'])
    def test_nominal_supervised_geometry_requires_alignment_before_near_approach(self):
        self.session.scope={'requires_runtime_visual_alignment':True}
        self.workflow().run()
        self.assertEqual(self.requests[0]['phase'],'approach_near')
        self.assertEqual(self.requests[0]['required_facts'],['empty_approach_alignment_verified'])
        self.assertEqual(len(self.requests),7)
    def test_rejected_vision_or_wrong_image_never_dispatches_contact(self):
        for wrong in ('proceed','image_sha256','exterior_fingers_aligned'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                self.session=FakeSession(Path(tmp))
                def bad(r,s):
                    a=review(r,s);a[wrong]=False;return a
                with self.assertRaisesRegex(RuntimeError,'unresolved'):self.workflow(bad).run()
                self.assertEqual(self.session.completed_stages,['approach_clear','approach_near'])
                self.assertFalse(self.session.finishes)
    def test_controller_exception_never_runs_the_next_phase(self):
        self.session.fail='lift_probe'
        with self.assertRaisesRegex(RuntimeError,'telemetry'):self.workflow().run()
        self.assertEqual(self.session.completed_stages,list(PHASE_SKILLS)[:4])
        self.assertFalse(self.session.finishes)
    def test_disk_full_diagnostic_does_not_mask_original_error(self):
        flow=self.workflow()
        with patch.object(self.session, 'decide', side_effect=RuntimeError('original telemetry failure')):
            with patch('fr5_jev.continuous_workflow.save', side_effect=OSError(28, 'disk full')):
                with self.assertRaisesRegex(RuntimeError, 'original telemetry failure'):
                    flow.run()
        self.assertFalse(self.session.finishes)
        self.assertIn('diagnostic_persistence_error', flow.metrics)

    def test_visual_checkpoint_rejects_changed_pose(self):
        def changed(request,session):
            session.state=lambda **kwargs:{'tcp':[1.]+[0.]*5};return review(request,session)
        with self.assertRaisesRegex(RuntimeError,'pose changed'):self.workflow(changed).run()
    def test_automatic_record_does_not_allow_contact_or_wrong_phase(self):
        s=self.session;s.decide('approach_clear','decision.json')
        record=json.loads((self.root/'decision.json').read_text())
        record['workflow_review']={'source':'deterministic_continuous_workflow',
                                   'phase':'approach_clear','proceed':True}
        validate_automatic_review(s,'approach_clear',record)
        for phase in ('close','approach_near'):
            with self.assertRaises(RuntimeError):validate_automatic_review(s,phase,record)
        self.assertFalse(selected_for_phase(record,'approach_clear','different-scope'))
    def test_retreat_auto_continuation_requires_unchanged_visual_release_record(self):
        self.workflow().run();p=self.root/'workflow_retreat_clear.json'
        record=json.loads(p.read_text());validate_automatic_review(self.session,'retreat_clear',record)
        (self.root/'workflow_retreat_probe.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'changed'):
            validate_automatic_review(self.session,'retreat_clear',record)


if __name__=='__main__':unittest.main()
