# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from maskrcnn_benchmark.data.dataset_mode import DatasetMode


def build_transforms(cfg, mode=DatasetMode.TRAIN):
    # default ColorJitter values
    brightness = 0.0
    contrast = 0.0
    saturation = 0.0
    hue = 0.0
    if mode == DatasetMode.TRAIN:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # noise_level = 0.1
        # noise_prob = 0.5
        noise_level = cfg.INPUT.GAUSSIAN_NOISE_LEVEL_TRAIN
        noise_prob = cfg.INPUT.GAUSSIAN_NOISE_PROB_TRAIN
        flip_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        if cfg.INPUT.COLOR_JITTER_TRAIN:
            brightness = cfg.INPUT.BRIGHTNESS
            contrast = cfg.INPUT.CONTRAST
            saturation = cfg.INPUT.SATURATION
            hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        noise_level = cfg.INPUT.GAUSSIAN_NOISE_LEVEL_TEST
        noise_prob = cfg.INPUT.GAUSSIAN_NOISE_PROB_TEST
        flip_prob = 0
        if cfg.INPUT.COLOR_JITTER_TEST:
            brightness = cfg.INPUT.BRIGHTNESS
            contrast = cfg.INPUT.CONTRAST
            saturation = cfg.INPUT.SATURATION
            hue = cfg.INPUT.HUE

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            T.GaussianNoise(noise_level, noise_prob),
            normalize_transform,
        ]
    )
    return transform
