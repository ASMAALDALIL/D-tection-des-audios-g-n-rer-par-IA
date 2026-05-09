
from typing import Optional
def get_augmentation() -> Optional[object]:
    try:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

        pipeline = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        ])
        print("Augmentation audiomentations activée")
        return pipeline

    except ImportError:
        print(" audiomentations non installé")
        return None
