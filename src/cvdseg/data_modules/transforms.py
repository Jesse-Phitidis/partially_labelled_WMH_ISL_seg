import torchio as tio
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy

class CopyAnyAffine(tio.Transform):

    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        keys = list(subject.keys())
        reference = subject[keys[0]]
        affine = copy.deepcopy(reference.affine)
        for key in keys[1:]:
            image = subject[key]
            image.load()
            if not np.allclose(affine, image.affine, atol=1e-5):
                raise RuntimeError(
                    f"Not all affines for the subject are close. Found: \n\n{reference.path}:\n{affine}\n{image.path}:\n{image.affine}\n"
                    )
            image.affine = affine
        return subject


def get_min_max(subject):
        min_max = {}
        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
                min_max[key] = (min, max)
        return min_max
    

class Brightness(tio.transforms.Transform):

    def __init__(self, rng=(0.7, 1.3), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        x = random.uniform(*self.rng)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                value.set_data(value.data * x)

        return subject


class Contrast(tio.transforms.Transform):

    def __init__(self, rng=(0.65, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        x = random.uniform(*self.rng)

        min_max = get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                scaled_data = value.data * x
                clamped_data = torch.clamp(scaled_data, min_max[key][0], min_max[key][1])
                value.set_data(clamped_data)

        return subject
    
    
class SimulateLowResolution(tio.transforms.Transform):

    def __init__(self, factor=(1, 2), p_per_mod=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.p_per_mod = p_per_mod

    def apply_transform(self, subject):

        for image in subject.get_images():
            T = tio.transforms.RandomAnisotropy(axes=(0,1,2), downsampling=self.factor, image_interpolation="bspline", p=self.p_per_mod)
            image.set_data(T(image.data))

        return subject
    
    
class Gamma(tio.transforms.Transform):

    def __init__(self, rng=(0.7, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        gamma = random.uniform(*self.rng)
        min_max = get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min, max = min_max[key][0], min_max[key][1]
                normalised_data = (value.data - min) / (max - min)
                if random.random() < 0.15:
                    augmented_data = 1 - (1 - normalised_data) ** gamma
                else:
                    augmented_data = normalised_data ** gamma
                rescaled_data = augmented_data * (max - min) + min
                value.set_data(rescaled_data)

        return subject