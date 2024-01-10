import torch
from torch import Tensor
from typing import Literal, Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict


class TimeMasking(BaseWaveformTransform):
    """
    Randomly masks a portion of an audio sample in the time domain.

    Args:
        p (float, optional): Probability of applying the augmentation. Default: 0.5.
        masked_timesteps (int, optional): Number of time steps to mask. If None, defaults to 5% of the sample length.
        mask_value (float | 'uniform' | 'normal', optimal): Value to mask with. Default: 0.0. Can also be set to 'uniform' to mask with a random value.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        masked_timesteps: Optional[int] = None,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.masked_timesteps = masked_timesteps

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if self.masked_timesteps is None:
            self.masked_timesteps = int(0.05 * samples.shape[-1])
        # get the maximum start index for the mask per sample in the batch      
        max_start = samples.shape[-1] - self.masked_timesteps
        # create a uniform distribution to sample the start indices from
        start_index_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                0, dtype=int, device=samples.device
            ),
            high=torch.tensor(
                max_start, dtype=int, device=samples.device
            ),
            validate_args=True,
        )
        # sample the start indices
        self.start_indices = start_index_distribution.sample(sample_shape=(samples.size(0),))
        # create a mask tensor per sample in the batch of shape (batch_size, 1, 1, num_timesteps) with random values
        self.mask = torch.rand(size=(samples.size(0), 1, 1, self.masked_timesteps), device=samples.device)
        

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        # apply the mask to the samples
        samples[self.start_indices: self.start_indices + self.masked_timesteps] = self.mask
       
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
