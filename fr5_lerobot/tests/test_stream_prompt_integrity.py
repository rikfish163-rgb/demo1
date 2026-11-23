"""Prevent silent loss of the task/state suffix and restore processor hooks."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import stream_prompt_integrity as guard


class StateStep:
    _registry_name = 'pi05_prepare_state_tokenizer_processor_step'
    max_state_dim = 32

    def __call__(self, value):
        values = value['observation']['observation.state'].numpy()
        bins = np.digitize(values, bins=np.linspace(-1, 1, 257)[:-1])-1
        value['tasks'] = [f'Task: {task}, State: {" ".join(map(str, row))};\nAction: '
                          for task, row in zip(value['tasks'], bins)]
        return value


class TokenizerStep:
    _registry_name = 'tokenizer_processor'

    def __init__(self):
        self.max_length = 200
        self.input_tokenizer = self.encode
        self.force_truncate = False

    @staticmethod
    def encode(texts, **_):
        return {'input_ids': [[2]+[ord(c)+3 for c in text] for text in texts]}

    def get_task(self, value):
        return value['tasks']

    def get_subtask(self, _):
        return None

    def __call__(self, value):
        count = 200 if self.force_truncate else self.max_length
        ids = self.encode(value['tasks'])['input_ids']
        tokens = torch.zeros((len(ids), count), dtype=torch.int64)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, row in enumerate(ids):
            row = row[:count]
            tokens[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = True
        value['observation'].update({'observation.language.tokens': tokens,
                                    'observation.language.attention_mask': mask})
        return value


class Pipeline:
    def __init__(self):
        self.steps = [StateStep(), TokenizerStep()]
        self.before_step_hooks, self.after_step_hooks = [], []

    def __call__(self, batch):
        value = copy.deepcopy(batch)
        for index, step in enumerate(self.steps):
            for hook in self.before_step_hooks:
                hook(index, value)
            value = step(value)
            for hook in self.after_step_hooks:
                hook(index, value)
        return value


def batch(task='抓取杯子', state=0.):
    return {'observation': {'observation.state': torch.full((1, 32), state)}, 'tasks': [task]}


class PromptIntegrityTests(unittest.TestCase):
    def assert_restored(self, pre):
        self.assertEqual(pre.steps[1].max_length, 200)
        self.assertEqual(pre.before_step_hooks, [])
        self.assertEqual(pre.after_step_hooks, [])

    def test_short_prompt_preserves_saved_processor_output(self):
        pre = Pipeline()
        expected = pre(batch())
        got = guard.preprocess(pre, batch())
        for key, value in expected['observation'].items():
            self.assertTrue(torch.equal(value, got['observation'][key]))
        self.assert_restored(pre)

    def test_long_task_preserves_full_state_and_changes_with_the_state(self):
        pre = Pipeline()
        original = pre(batch('抓取杯子'*40, .4))
        first = guard.preprocess(pre, batch('抓取杯子'*40, .4))
        second = guard.preprocess(pre, batch('抓取杯子'*40, -.4))
        key = 'observation.language.tokens'
        self.assertEqual(original['observation'][key].shape, (1, 200))
        self.assertEqual(first['observation'][key].shape, (1, 512))
        self.assertFalse(torch.equal(first['observation'][key], second['observation'][key]))
        self.assert_restored(pre)

    def test_oversized_prompt_fails_before_silent_truncation(self):
        pre = Pipeline()
        with self.assertRaisesRegex(ValueError, 'maximum is 512'):
            guard.preprocess(pre, batch('任务'*300))
        self.assert_restored(pre)

    def test_tokenizer_that_still_truncates_is_detected(self):
        pre = Pipeline()
        pre.steps[1].force_truncate = True
        with self.assertRaisesRegex(ValueError, 'complete prompt'):
            guard.preprocess(pre, batch('抓取杯子'*40))
        self.assert_restored(pre)

    def test_invalid_state_and_missing_prompt_step_do_not_leave_hooks(self):
        pre = Pipeline()
        with self.assertRaisesRegex(ValueError, 'complete batched'):
            guard.preprocess(pre, batch(state=float('nan')))
        self.assert_restored(pre)
        pre.steps.reverse()
        with self.assertRaisesRegex(ValueError, 'immediately before'):
            guard.preprocess(pre, batch())
        self.assertEqual(pre.before_step_hooks, [])
        self.assertEqual(pre.after_step_hooks, [])


if __name__ == '__main__':
    unittest.main()
