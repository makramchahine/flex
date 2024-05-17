from typing import List, Optional, Callable

from collections import deque
import numpy as np

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

try:
    import clip
except ImportError:
    print("Fail to import clip")


class RejectionSampler:
    def __init__(self, buffer_size=int(4e5)):
        super(RejectionSampler, self).__init__()
        self.samples = deque(maxlen=buffer_size)

    def add_to_history(self, value: float):
        self.samples.append(value)

    def get_sampling_probability(self,
                                 value: float,
                                 smoothing_factor: Optional[float] = 0.01,
                                 min_p: Optional[float] = 0.25,
                                 n_bins: Optional[float] = 30):

        # Find which latent bin every data sample falls in
        if len(self.samples) == 0:
            return 1.
        density, bins = np.histogram(self.samples, density=True, bins=n_bins)
        bins[0] = -float('inf')
        bins[-1] = float('inf')

        # smooth the density function
        smooth_density = density / density.sum()
        smooth_density = smooth_density + (smoothing_factor + 1e-10)
        smooth_density /= smooth_density.sum()

        # invert the density function and normalize
        p = 1.0 / smooth_density
        p = (p - p.min()) / (p.max() - p.min())
        p = np.clip(p, p + min_p, 1)

        # Find which bin the new value sample belongs to and return that prob
        bin_idx = np.digitize(value, bins)
        prob = p[bin_idx - 1]
    
        return prob
    
    
def transform_rgb(img: np.ndarray,
                  sensor: Camera,
                  train: bool,
                  label: float = None,
                  use_standardize: bool = True,
                  use_clip_preprocess: bool = False,
                  lavis_preprocessor: str = None):
    # need copy here probably since img is not contiguous
    img = img.copy()
    img = TF.to_tensor(img)
    if train:  # perform color jitter
        gamma_range = [0.5, 1.5]
        brightness_range = [0.5, 1.5]
        contrast_range = [0.3, 1.7]
        saturation_range = [0.5, 1.5]

        img = TF.adjust_gamma(img, np.random.uniform(*gamma_range))
        img = TF.adjust_brightness(img, np.random.uniform(*brightness_range))
        img = TF.adjust_contrast(img, np.random.uniform(*contrast_range))
        img = TF.adjust_saturation(img, np.random.uniform(*saturation_range))
    if use_standardize:
        img = standardize(img)
    if use_clip_preprocess:
        # follow clip._transform
        n_px = 224
        img = TF.resize(img, n_px, interpolation=InterpolationMode.BICUBIC, antialias=True) # NOTE: resize needs to come after color jitter otherwise will cause nan
        img = TF.center_crop(img, n_px)

        img = TF.normalize(img, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if lavis_preprocessor:
        img = lavis_preprocessor(img)
    return img if label is None else (img, label)


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev
