"""A user decision must bind to the exact complete source episode."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from lerobot_robot_fairino.human_review import record_review,verify_success_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.episode=Path(self.tmp.name)/'episode'
        self.episode.mkdir()
        scope={'phase_order':['approach','grasp','place']}
        raw=json.dumps(scope).encode()
        (self.episode/'scope.json').write_bytes(raw)
        (self.episode/'proof.jpg').write_bytes(b'synthetic image')
        self.summary={'outcome':'pending_review','released':True,'failed':None,
                      'evidence':['proof.jpg'],'completed_stages':scope['phase_order'],
                      'continuous_capture':{'capture_complete':True},
                      'scope_sha256':hashlib.sha256(raw).hexdigest()}
        (self.episode/'summary.json').write_text(json.dumps(self.summary))

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_requires_explicit_user_decision_and_is_bound_to_summary(self):
        with self.assertRaises(FileNotFoundError):verify_success_review(self.episode)
        record_review(self.episode,verdict='success',user_statement='SYNTHETIC: success')
        review,digest=verify_success_review(self.episode)
        self.assertEqual(review['reviewer'],'user')
        self.assertEqual(len(digest),64)
        with self.assertRaises(FileExistsError):
            record_review(self.episode,verdict='failure',user_statement='SYNTHETIC')
        self.summary['released']=False
        (self.episode/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError):verify_success_review(self.episode)

    def test_failure_or_incomplete_episode_never_becomes_success(self):
        record_review(self.episode,verdict='failure',user_statement='SYNTHETIC: failed')
        with self.assertRaises(ValueError):verify_success_review(self.episode)
        (self.episode/'user-review.json').unlink()
        self.summary['completed_stages']=['approach']
        (self.episode/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError):
            record_review(self.episode,verdict='success',user_statement='SYNTHETIC')

    def test_incomplete_marker_blocks_even_if_summary_exists(self):
        (self.episode/'INCOMPLETE').write_text('synthetic incomplete composition')
        with self.assertRaisesRegex(ValueError,'Incomplete composite'):
            record_review(self.episode,verdict='success',user_statement='SYNTHETIC')
        self.assertFalse((self.episode/'user-review.json').exists())

    def test_relocated_absolute_evidence_keeps_summary_and_user_review_hashes(self):
        original=self.episode
        self.summary['evidence']=[str(original/'proof.jpg')]
        self.summary['continuous_capture']['metadata']={
            'external_command_journal':str(original/'target-command-journal.json')}
        (original/'summary.json').write_text(json.dumps(self.summary))
        record_review(original,verdict='success',user_statement='SYNTHETIC: success')
        summary_raw=(original/'summary.json').read_bytes()
        _,review_sha=verify_success_review(original)
        moved=Path(self.tmp.name)/'data4-copy'
        shutil.move(original,moved)
        self.assertEqual((moved/'summary.json').read_bytes(),summary_raw)
        self.assertEqual(verify_success_review(moved)[1],review_sha)
        (moved/'proof.jpg').unlink()
        with self.assertRaisesRegex(ValueError,'evidence is missing'):
            verify_success_review(moved)

    def test_legacy_relocation_requires_exact_original_root(self):
        original=self.episode
        self.summary['evidence']=[str(original/'proof.jpg')]
        (original/'summary.json').write_text(json.dumps(self.summary))
        record_review(original,verdict='success',user_statement='SYNTHETIC')
        moved=Path(self.tmp.name)/'copy'
        shutil.move(original,moved)
        with self.assertRaisesRegex(ValueError,'evidence is missing'):
            verify_success_review(moved)
        self.assertEqual(verify_success_review(moved,original_episode=original)[0]['verdict'],'success')
        with self.assertRaisesRegex(ValueError,'evidence is missing'):
            verify_success_review(moved,original_episode=Path(self.tmp.name)/'wrong')




if __name__=='__main__':unittest.main()
