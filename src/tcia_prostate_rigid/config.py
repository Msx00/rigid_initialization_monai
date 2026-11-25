"""Immutable settings, shared by Python and CLI interfaces."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RegistrationConfig:
    pca_rotation_gain: float = 0.0
    smoothing: int = 0
    landmarks: int = 4000
    iterations: int = 150
    decimation_reduction: float = 0.2
    size: tuple[int, int, int] = (175, 175, 175)
    spacing_mm: tuple[float, float, float] = (0.8, 0.8, 0.8)
    rotation_warning_degrees: float = 30.0
    tre_warning_mm: float = 10.0
    threads: int = 1
    assume_unknown_nifti_units_are_mm: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'size', tuple(self.size))
        object.__setattr__(self, 'spacing_mm', tuple(self.spacing_mm))
        if not math.isfinite(self.pca_rotation_gain) or not 0 <= self.pca_rotation_gain <= 1:
            raise ValueError('pca_rotation_gain must be between 0 and 1')
        if not math.isfinite(self.decimation_reduction) or not 0 <= self.decimation_reduction < 1:
            raise ValueError('decimation_reduction must be between 0 (inclusive) and 1')
        for key in ['smoothing', 'landmarks', 'iterations', 'threads']:
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key == 'smoothing' else 1):
                raise ValueError(f'Invalid {key}')
        if len(self.size) != 3 or any(type(x) is not int or x < 2 for x in self.size):
            raise ValueError('size must contain three integers >= 2')
        if len(self.spacing_mm) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.spacing_mm):
            raise ValueError('spacing_mm must contain three positive finite values')
        if not math.isfinite(self.rotation_warning_degrees) or not 0 <= self.rotation_warning_degrees <= 180:
            raise ValueError('rotation_warning_degrees must be in [0, 180]')
        if not math.isfinite(self.tre_warning_mm) or self.tre_warning_mm <= 0:
            raise ValueError('tre_warning_mm must be positive and finite')

    def to_dict(self):
        return asdict(self)

