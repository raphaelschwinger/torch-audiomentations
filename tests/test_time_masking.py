import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations.augmentations.time_masking import TimeMasking

class TestTimeMasking(unittest.TestCase):
    def test_time_masking(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = TimeMasking(masked_timesteps=None, p=1.0, output_type='dict')
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        # assert that samples and augmented samples have the same shape
        assert samples.shape == processed_samples.shape

        # assert that the augmented samples are different from the original samples
        assert not np.allclose(samples, processed_samples)





