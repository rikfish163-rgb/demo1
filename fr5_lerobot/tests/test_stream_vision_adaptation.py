"""CPU-only checks for explicit SigLIP q/v LoRA and strict checkpoint loading."""
import copy
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import train_stream_pi05 as training
import stream_checkpoint as checkpoints


HAS_PEFT = (importlib.util.find_spec('peft') is not None
            and importlib.util.find_spec('transformers') is not None)


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)

    def forward(self, value):
        return self.q_proj(value) + self.v_proj(value)


class TinyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = TinyAttention()
        self.layers = torch.nn.ModuleList([layer])


class TinyPiPolicy(torch.nn.Module):
    """A small random policy with the installed SigLIP implementation and PI05 names."""
    def __init__(self, config=None):
        super().__init__()
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        self.config = config or SimpleNamespace(freeze_vision_encoder=False, train_expert_only=False)
        self.model = torch.nn.Module()
        self.model.action_in_proj = torch.nn.Linear(8, 8)
        self.model.action_out_proj = torch.nn.Linear(8, 8)
        self.model.paligemma_with_expert = torch.nn.Module()
        core = self.model.paligemma_with_expert
        core.gemma_expert = torch.nn.Module()
        core.gemma_expert.model = TinyBranch()
        core.gemma_expert.lm_head = torch.nn.Linear(8, 8)
        core.paligemma = torch.nn.Module()
        core.paligemma.model = torch.nn.Module()
        core.paligemma.model.language_model = TinyBranch()
        vision_config = SiglipVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                           num_attention_heads=2, image_size=8, patch_size=4,
                                           vision_use_head=False)
        core.paligemma.model.vision_tower = SiglipVisionModel(vision_config)

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state

    def forward(self, images):
        core = self.model.paligemma_with_expert
        image_features = core.paligemma.model.vision_tower(pixel_values=images).last_hidden_state.mean(dim=1)
        language = core.paligemma.model.language_model.layers[0].self_attn(image_features)
        expert = core.gemma_expert.model.layers[0].self_attn(image_features + language)
        return self.model.action_out_proj(self.model.action_in_proj(image_features + language + expert))


class VisionModeContractTests(unittest.TestCase):
    def test_upgrade_is_explicit_and_resume_cannot_change_adaptation(self):
        old, new, lora = (training.FULL_EXPERT_ADAPTATION, training.VISION_EXPERT_ADAPTATION,
                          training.LORA_ADAPTATION)
        resolve = training.resolve_adaptation_transition
        self.assertTrue(resolve(old, new, initialize=True))
        self.assertFalse(resolve(new, new, resume=True))
        self.assertFalse(resolve(lora, old, initialize=True))
        for parent, requested, kwargs in ((old, new, {}), (old, new, {'resume': True}),
                                         (lora, new, {'initialize': True}),
                                         (new, old, {'initialize': True})):
            with self.subTest(parent=parent, requested=requested), self.assertRaises(ValueError):
                resolve(parent, requested, **kwargs)

    def test_cli_keeps_old_default_and_accepts_explicit_vision_mode(self):
        common = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1']
        for extra, expected in (([], training.LORA_ADAPTATION),
                                (['--adaptation', training.VISION_EXPERT_ADAPTATION],
                                 training.VISION_EXPERT_ADAPTATION)):
            with patch.object(sys, 'argv', common + extra), patch.object(training, 'run') as run:
                training.main()
                self.assertEqual(run.call_args.args[0].adaptation, expected)

    def test_vision_inventory_requires_q_and_v_a_and_b(self):
        prefix = 'model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers'
        names = [f'{prefix}.{layer}.self_attn.{projection}_proj.lora_{matrix}.default.weight'
                 for layer in range(2) for projection in ('q', 'v') for matrix in ('A', 'B')]
        self.assertEqual(training.vision_adapter_inventory(names), [0, 1])
        with self.assertRaisesRegex(ValueError, 'q/v A/B'):
            training.vision_adapter_inventory(names[:-1])
        self.assertFalse(training.is_vision_adapter(names[0].replace('q_proj', 'k_proj')))


@unittest.skipUnless(HAS_PEFT, 'PEFT and Transformers are required for the CPU SigLIP tests')
class VisionPeftTests(unittest.TestCase):
    @staticmethod
    def inject(model, mode):
        from peft import LoraConfig, inject_adapter_in_model
        targets = training.VISION_EXPERT_LORA_TARGETS if mode == training.VISION_EXPERT_ADAPTATION else training.LORA_TARGETS
        model = inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, lora_dropout=0., bias='none',
                                                   target_modules=targets), model)
        training.trainable_parameters(model, mode)
        return model

    @staticmethod
    def save_tiny_checkpoint(model, root, mode):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        root.mkdir()
        model.peft_config['default'].save_pretrained(root)
        adapter = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
        save_file(adapter, str(root / 'adapter_model.safetensors'))
        save_file(training.projection_state(model), str(root / 'projections.safetensors'))
        save_file(training.expert_state(model), str(root / 'expert_model.safetensors'))
        checkpoints.write_json(root / 'adaptation.json', training.adaptation_metadata(mode))
        return adapter

    def test_real_siglip_namespace_and_trainable_gradients(self):
        torch.manual_seed(71)
        policy = self.inject(TinyPiPolicy(), training.VISION_EXPERT_ADAPTATION)
        report = training.vision_adapter_report(policy, training.VISION_EXPERT_ADAPTATION)
        self.assertEqual(report['layers'], [0, 1])
        self.assertEqual(report['tensor_count'], 8)
        self.assertEqual(report['trainable_parameters'], 2 * 2 * (2 * 8 * 2))
        params = dict(policy.named_parameters())
        vision = {name: value for name, value in params.items() if training.is_vision_adapter(name)}
        self.assertTrue(all(value.requires_grad for value in vision.values()))
        self.assertTrue(all(torch.count_nonzero(value) == 0 for name, value in vision.items()
                            if '.lora_B.' in name))
        self.assertTrue(all(not value.requires_grad for name, value in params.items()
                            if '.vision_tower.' in name and name not in vision))
        self.assertTrue(all(not value.requires_grad for name, value in params.items()
                            if '.gemma_expert.model.' in name and '.lora_' in name))
        images = torch.randn(2, 3, 8, 8)
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=.1)
        from torch.utils.checkpoint import checkpoint
        # LeRobot checkpoints its image path with non-reentrant autograd, even
        # though camera tensors do not themselves require gradients.
        checkpoint(policy, images, use_reentrant=False, preserve_rng_state=False).square().mean().backward()
        components = training.gradient_summary(policy)
        self.assertTrue({'vision_lora', 'language_lora', 'action_expert', 'action_projections'} <= set(components))
        for projection in ('q_proj', 'v_proj'):
            self.assertTrue(any(value.grad is not None and value.grad.abs().sum() > 0
                                for name, value in vision.items() if projection in name and '.lora_B.' in name))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        policy(images).square().mean().backward()
        self.assertTrue(any(value.grad is not None and value.grad.abs().sum() > 0
                            for name, value in vision.items() if '.lora_A.' in name))

    def test_initialize_upgrade_and_strict_new_checkpoint_reload(self):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file, save_file
        import lerobot.policies.pi05.modeling_pi05 as official
        torch.manual_seed(123)
        base = TinyPiPolicy()
        base_state = copy.deepcopy(base.state_dict())
        parent = self.inject(base, training.FULL_EXPERT_ADAPTATION)
        with torch.no_grad():
            for name, value in parent.named_parameters():
                if '.lora_' in name:
                    value.fill_(.03)
                elif training.is_expert_base(name) or name.startswith(training.PROJECTION_PREFIXES):
                    value.add_(.01)
        parent.eval()
        images = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            parent_output = parent(images).clone()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_file = root / 'base.safetensors'
            save_file(base_state, str(base_file))
            old = root / 'old'
            old_adapter = self.save_tiny_checkpoint(parent, old, training.FULL_EXPERT_ADAPTATION)
            config = SimpleNamespace(freeze_vision_encoder=False, train_expert_only=False)
            with patch.object(official, 'PI05Policy', TinyPiPolicy):
                fresh = training.load_policy(config, base_file, rank=2,
                                             adaptation=training.VISION_EXPERT_ADAPTATION)
                fresh_vision = {name: value for name, value in get_peft_model_state_dict(fresh).items()
                                if training.is_vision_adapter(name)}
                self.assertTrue(fresh_vision)
                self.assertTrue(all(torch.count_nonzero(value) == 0 for name, value in fresh_vision.items()
                                    if '.lora_B.' in name))
                with self.assertRaisesRegex(ValueError, 'explicit initialization'):
                    training.load_policy(config, base_file, checkpoint=old,
                                         adaptation=training.VISION_EXPERT_ADAPTATION)
                child = training.load_policy(config, base_file, checkpoint=old,
                                             adaptation=training.VISION_EXPERT_ADAPTATION,
                                             allow_vision_upgrade=True)
                child.eval()
                with torch.no_grad():
                    torch.testing.assert_close(child(images), parent_output, rtol=0, atol=1e-6)
                child_adapter = get_peft_model_state_dict(child)
                self.assertTrue(all(torch.equal(child_adapter[k], v) for k, v in old_adapter.items()))
                new_keys = set(child_adapter) - set(old_adapter)
                self.assertTrue(new_keys and all(training.is_vision_adapter(name) for name in new_keys))
                self.assertTrue(all(torch.count_nonzero(child_adapter[name]) == 0
                                    for name in new_keys if '.lora_B.' in name))
                new = root / 'new'
                self.save_tiny_checkpoint(child, new, training.VISION_EXPERT_ADAPTATION)
                restored = training.load_policy(config, base_file, checkpoint=new)
                self.assertTrue(all(torch.equal(v, child_adapter[k])
                                    for k, v in get_peft_model_state_dict(restored).items()))
                saved = load_file(str(new / 'adapter_model.safetensors'))
                missing = next(name for name in saved if training.is_vision_adapter(name))
                save_file({name: value for name, value in saved.items() if name != missing},
                          str(new / 'adapter_model.safetensors'))
                with self.assertRaisesRegex(ValueError, 'Adapter tensor keys or shapes differ'):
                    training.load_policy(config, base_file, checkpoint=new)
                save_file({**saved, 'unexpected.lora_A.weight': torch.zeros(1)},
                          str(new / 'adapter_model.safetensors'))
                with self.assertRaisesRegex(ValueError, 'Adapter tensor keys or shapes differ'):
                    training.load_policy(config, base_file, checkpoint=new)
                save_file(saved, str(new / 'adapter_model.safetensors'))
                original_expert = load_file(str(new / 'expert_model.safetensors'))
                omitted_expert = next(iter(original_expert))
                save_file({name: value for name, value in original_expert.items() if name != omitted_expert},
                          str(new / 'expert_model.safetensors'))
                with self.assertRaisesRegex(ValueError, 'action-expert tensors do not match'):
                    training.load_policy(config, base_file, checkpoint=new)
                save_file(original_expert, str(new / 'expert_model.safetensors'))
                nonzero = self.inject(TinyPiPolicy(), training.VISION_EXPERT_ADAPTATION)
                with torch.no_grad():
                    for name, value in nonzero.named_parameters():
                        if training.is_vision_adapter(name) and '.lora_B.' in name:
                            value.fill_(.1)
                with self.assertRaisesRegex(ValueError, 'B tensors must be zero'):
                    training.reload_parameters(nonzero, old, allow_vision_upgrade=True,
                                               target_mode=training.VISION_EXPERT_ADAPTATION)
                old_saved = load_file(str(old / 'adapter_model.safetensors'))
                omitted_old = next(iter(old_saved))
                save_file({name: value for name, value in old_saved.items() if name != omitted_old},
                          str(old / 'adapter_model.safetensors'))
                with self.assertRaisesRegex(ValueError, 'every old adapter'):
                    training.load_policy(config, base_file, checkpoint=old,
                                         adaptation=training.VISION_EXPERT_ADAPTATION,
                                         allow_vision_upgrade=True)

    def test_official_config_enables_vision_only_for_new_mode(self):
        self.assertTrue(training.make_config().freeze_vision_encoder)
        config = training.make_config(10, adaptation=training.VISION_EXPERT_ADAPTATION)
        self.assertFalse(config.freeze_vision_encoder)
        self.assertFalse(config.train_expert_only)
        self.assertEqual(config.n_action_steps, 10)

    def test_new_mode_checkpoint_seals_and_bundled_loader_restores(self):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        import lerobot.policies.pi05.modeling_pi05 as official
        import serve_stream_pi05 as service

        class Config(SimpleNamespace):
            def save_pretrained(self, root):
                checkpoints.write_json(Path(root) / 'config.json', {
                    'type': 'pi05', 'chunk_size': 50, 'n_action_steps': 10,
                    'freeze_vision_encoder': False, 'train_expert_only': False,
                    'use_relative_actions': False,
                    'input_features': {'observation.state': {'shape': [8]}},
                    'output_features': {'action': {'shape': [7]}}})

        class Processor:
            def __init__(self, name):
                self.name = name
            def save_pretrained(self, root):
                checkpoints.write_json(Path(root) / f'policy_{self.name}processor.json', {})
                save_file({'stats': torch.ones(1)},
                          str(Path(root) / f'policy_{self.name}processor_step_0_test.safetensors'))

        torch.manual_seed(311)
        config = Config(freeze_vision_encoder=False, train_expert_only=False)
        policy = TinyPiPolicy(config)
        base_state = copy.deepcopy(policy.state_dict())
        policy = self.inject(policy, training.VISION_EXPERT_ADAPTATION)
        before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(policy).items()}
        expert_before = {k: v.clone() for k, v in training.expert_state(policy).items()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'run'
            run.mkdir()
            base = root / 'base.safetensors'
            save_file(base_state, str(base))
            tokenizer = root / 'revision'
            tokenizer.mkdir()
            (tokenizer / 'tokenizer.json').write_text('{}')
            stats = {name: {'q01': torch.zeros(size), 'q99': torch.ones(size)}
                     for name, size in (('observation.state', 8), ('action', 7))}
            report = {'adaptation_mode': training.VISION_EXPERT_ADAPTATION,
                      'action_representation': training.ABSOLUTE_REPRESENTATION,
                      'image_profile': training.FULL_IMAGE_PROFILE, 'dataset_fingerprint': 'f',
                      'n_action_steps': 10}
            optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad])
            with patch.object(training.legacy, 'EXPECTED_WEIGHTS_SHA256', checkpoints.digest(base)), \
                    patch.object(training.legacy, 'TOKENIZER_REVISION', 'revision'), \
                    patch.object(training.legacy, 'tokenizer_path', return_value=tokenizer), \
                    patch.object(torch.cuda, 'get_rng_state', return_value=torch.zeros(1, dtype=torch.uint8)):
                training.save_checkpoint(run, 1, policy, optimizer, Processor('pre'), Processor('post'),
                                         stats, report, random.Random(1), set(), base, 'f')
            bundle = run / 'step-000001'
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['adaptation_mode'], training.VISION_EXPERT_ADAPTATION)
            self.assertEqual(info['n_action_steps'], 10)
            self.assertIn('expert_model.safetensors', info['manifest']['files'])
            bundled = service.load_bundle_module('test_vision_bundled_training', bundle / 'train_stream_pi05.py')
            with patch.object(official, 'PI05Policy', TinyPiPolicy):
                restored = bundled.load_policy(config, base, checkpoint=bundle)
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in get_peft_model_state_dict(restored).items()))
            self.assertTrue(all(torch.equal(v, expert_before[k]) for k, v in bundled.expert_state(restored).items()))


if __name__ == '__main__':
    unittest.main()
