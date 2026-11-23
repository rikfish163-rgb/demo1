"""CPU-only checks for the shared FR5 image-profile transformation."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import stream_image_profile as profiles


class ImageProfileTests(unittest.TestCase):
    def test_full_default_preserves_legacy_values_in_new_mapping(self):
        image = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
        existing_wrist = torch.ones(1, 3, 7, 9)
        source = {profiles.SOURCE_KEY: image, profiles.WORKSPACE_KEY: existing_wrist,
                  'task': ['legacy input']}
        result = profiles.apply_image_profile(source)
        self.assertIsNot(result, source)
        self.assertEqual(set(result), set(source))
        self.assertIs(result[profiles.SOURCE_KEY], image)
        self.assertIs(result[profiles.WORKSPACE_KEY], existing_wrist)
        self.assertIs(result['task'], source['task'])

    def test_workspace_crop_coordinates_values_and_independent_storage(self):
        image = torch.arange(2 * 3 * 360 * 640, dtype=torch.float32).reshape(2, 3, 360, 640)
        baseline = image.clone()
        source = {profiles.SOURCE_KEY: image, 'task': ['one', 'two']}
        result = profiles.apply_image_profile(source, profiles.WORKSPACE_IMAGE_PROFILE)
        crop = result[profiles.WORKSPACE_KEY]
        self.assertIsNot(result, source)
        self.assertNotIn(profiles.WORKSPACE_KEY, source)
        self.assertIs(result[profiles.SOURCE_KEY], image)
        self.assertEqual(tuple(crop.shape), (2, 3, 260, 400))
        self.assertTrue(torch.equal(crop, image[:, :, 100:360, 240:640]))
        self.assertEqual(crop[1, 2, 0, 0].item(), image[1, 2, 100, 240].item())
        self.assertEqual(crop[0, 1, -1, -1].item(), image[0, 1, 359, 639].item())
        crop[0, 0, 0, 0] = -123.
        self.assertTrue(torch.equal(image, baseline))

    def test_training_and_service_inputs_produce_the_same_views(self):
        # Both paths use the same RGB pixels, arriving as a training tensor or
        # an HWC NumPy image decoded by the serving path.
        hwc = (torch.arange(360 * 640 * 3, dtype=torch.int32) % 256).to(torch.uint8).reshape(360, 640, 3)
        training_image = hwc.permute(2, 0, 1).float().unsqueeze(0) / 255
        serving_image = torch.from_numpy(hwc.numpy().copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        training = profiles.apply_image_profile({profiles.SOURCE_KEY: training_image}, profiles.WORKSPACE_IMAGE_PROFILE)
        serving = profiles.apply_image_profile({profiles.SOURCE_KEY: serving_image}, profiles.WORKSPACE_IMAGE_PROFILE)
        self.assertTrue(torch.equal(training[profiles.SOURCE_KEY], serving[profiles.SOURCE_KEY]))
        self.assertTrue(torch.equal(training[profiles.WORKSPACE_KEY], serving[profiles.WORKSPACE_KEY]))

    def test_workspace_only_preserves_source_pixels_and_excludes_full_view(self):
        image = torch.arange(3*360*640, dtype=torch.float32).reshape(1, 3, 360, 640)
        original = image.clone()
        source = {profiles.SOURCE_KEY: image, 'task': ['grab cup'], 'observation.state': torch.zeros(1, 8)}
        result = profiles.apply_image_profile(source, profiles.WORKSPACE_ONLY_IMAGE_PROFILE)
        self.assertEqual(set(result), set(source))
        self.assertNotIn(profiles.WORKSPACE_KEY, result)
        crop = result[profiles.SOURCE_KEY]
        self.assertEqual(tuple(crop.shape), (1, 3, 260, 400))
        self.assertTrue(torch.equal(crop, image[:, :, 100:360, 240:640]))
        crop.zero_()
        self.assertTrue(torch.equal(image, original))
        with self.assertRaisesRegex(ValueError, r'\[B, 3, 360, 640\]'):
            profiles.apply_image_profile(result, profiles.WORKSPACE_ONLY_IMAGE_PROFILE)
        with self.assertRaisesRegex(ValueError, 'only the original front image'):
            profiles.apply_image_profile({**source, 'observation.images.right_wrist_0_rgb': image},
                                         profiles.WORKSPACE_ONLY_IMAGE_PROFILE)

    def test_workspace_only_metadata_agrees_with_checkpoint_and_serving_path(self):
        import stream_checkpoint
        for name in profiles.IMAGE_PROFILES:
            self.assertEqual(profiles.image_profile_metadata(name), stream_checkpoint.image_profile_metadata(name))
        info = profiles.image_profile_metadata(profiles.WORKSPACE_ONLY_IMAGE_PROFILE)
        self.assertEqual(info['workspace_key'], profiles.SOURCE_KEY)
        self.assertFalse(info['full_view_retained'])
        self.assertEqual(info['output_image_keys'], [profiles.SOURCE_KEY])
        hwc = torch.randint(0, 256, (360, 640, 3), dtype=torch.uint8)
        training = profiles.apply_image_profile({profiles.SOURCE_KEY: hwc.permute(2, 0, 1).float().unsqueeze(0)/255},
                                                 profiles.WORKSPACE_ONLY_IMAGE_PROFILE)
        serving = profiles.apply_image_profile({profiles.SOURCE_KEY: torch.from_numpy(hwc.numpy().copy()).permute(2, 0, 1).float().unsqueeze(0)/255},
                                                profiles.WORKSPACE_ONLY_IMAGE_PROFILE)
        self.assertTrue(torch.equal(training[profiles.SOURCE_KEY], serving[profiles.SOURCE_KEY]))

    def test_cup_centered_keeps_rim_at_same_crop_position_for_batched_observations(self):
        image = torch.arange(2*3*360*640, dtype=torch.float32).reshape(2, 3, 360, 640)
        source = {profiles.SOURCE_KEY: image}
        result = profiles.apply_image_profile(source, profiles.CUP_CENTERED_IMAGE_PROFILE,
                                              initial_cup_uv=[[1238., 688.], [1591.44, 833.86]])
        crop = result[profiles.SOURCE_KEY]
        self.assertEqual(crop.shape, (2, 3, 256, 384))
        torch.testing.assert_close(crop[0, :, 115, 230], image[0, :, 229, 413])
        torch.testing.assert_close(crop[1, :, 115, 230], image[1, :, 278, 530])
        torch.testing.assert_close(crop[1, :, -1, -1], image[1, :, -1, -1])
        self.assertNotIn(profiles.WORKSPACE_KEY, result)
        with self.assertRaisesRegex(ValueError, 'one opening cup position'):
            profiles.apply_image_profile(source, profiles.CUP_CENTERED_IMAGE_PROFILE,
                                         initial_cup_uv=[[1238., 688.]])

    def test_official_image_preprocessing_sees_two_views(self):
        # The normal processor maps front to base_0_rgb before this stage.
        # No policy weights or device are loaded by this CPU-only probe.
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        class Probe:
            config = SimpleNamespace(image_resolution=(224, 224), image_features={
                'observation.images.base_0_rgb': None,
                profiles.WORKSPACE_KEY: None,
                'observation.images.right_wrist_0_rgb': None})

            def parameters(self):
                return iter([torch.empty(1)])

        raw = {profiles.SOURCE_KEY: torch.rand(1, 3, 360, 640)}
        for profile, expected_masks in ((profiles.FULL_IMAGE_PROFILE, [True, False, False]),
                                        (profiles.WORKSPACE_IMAGE_PROFILE, [True, True, False]),
                                        (profiles.WORKSPACE_ONLY_IMAGE_PROFILE, [True, False, False])):
            batch = profiles.apply_image_profile(raw, profile)
            batch['observation.images.base_0_rgb'] = batch.pop(profiles.SOURCE_KEY)
            images, masks = PI05Policy._preprocess_images(Probe(), batch)
            self.assertEqual([tuple(image.shape) for image in images], [(1, 3, 224, 224)] * 3)
            self.assertEqual([bool(mask.item()) for mask in masks], expected_masks)

    def test_workspace_rejects_missing_wrong_shape_and_non_float(self):
        with self.assertRaisesRegex(KeyError, 'Missing source image'):
            profiles.apply_image_profile({}, profiles.WORKSPACE_IMAGE_PROFILE)
        with self.assertRaisesRegex(TypeError, 'floating-point torch tensor'):
            profiles.apply_image_profile({profiles.SOURCE_KEY: torch.zeros(1, 3, 360, 640, dtype=torch.uint8)},
                                         profiles.WORKSPACE_IMAGE_PROFILE)
        with self.assertRaisesRegex(TypeError, 'floating-point torch tensor'):
            profiles.apply_image_profile({profiles.SOURCE_KEY: 'not a tensor'}, profiles.WORKSPACE_IMAGE_PROFILE)
        for shape in ((3, 360, 640), (1, 360, 640, 3), (1, 1, 360, 640), (1, 3, 359, 640), (1, 3, 360, 639)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, r'\[B, 3, 360, 640\]'):
                profiles.apply_image_profile({profiles.SOURCE_KEY: torch.zeros(shape)}, profiles.WORKSPACE_IMAGE_PROFILE)

    def test_workspace_refuses_existing_slot_and_repeated_application(self):
        source = {profiles.SOURCE_KEY: torch.zeros(1, 3, 360, 640)}
        first = profiles.apply_image_profile(source, profiles.WORKSPACE_IMAGE_PROFILE)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            profiles.apply_image_profile(first, profiles.WORKSPACE_IMAGE_PROFILE)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            profiles.apply_image_profile({**source, profiles.WORKSPACE_KEY: None}, profiles.WORKSPACE_IMAGE_PROFILE)
        self.assertNotIn(profiles.WORKSPACE_KEY, source)

    def test_metadata_is_json_and_identifies_derived_single_camera(self):
        info = json.loads(json.dumps(profiles.image_profile_metadata(profiles.WORKSPACE_IMAGE_PROFILE)))
        self.assertEqual(info['source_key'], profiles.SOURCE_KEY)
        self.assertEqual(info['workspace_key'], profiles.WORKSPACE_KEY)
        self.assertEqual(info['workspace_crop_xyxy'], [240, 100, 640, 360])
        self.assertEqual(info['workspace_crop_coordinates'], 'xyxy_half_open')
        self.assertEqual(info['workspace_view_source'], 'crop_of_front_camera')
        self.assertEqual(info['physical_camera_count'], 1)
        self.assertFalse(info['workspace_is_physical_wrist_camera'])
        self.assertIsNone(profiles.image_profile_metadata()['workspace_key'])

    def test_unknown_profile_or_non_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown image profile'):
            profiles.apply_image_profile({}, 'unrecognized')
        with self.assertRaisesRegex(ValueError, 'Unknown image profile'):
            profiles.image_profile_metadata('unrecognized')
        with self.assertRaisesRegex(TypeError, 'mapping'):
            profiles.apply_image_profile(None)


if __name__ == '__main__':
    unittest.main()
