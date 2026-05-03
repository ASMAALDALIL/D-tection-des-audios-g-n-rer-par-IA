from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Mp3Compression


def get_augmentation_pipeline():
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.3),
    ])