#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.PATH_CACHE = False
    _C.DATA.SPEED_CORRECTION = False  # For the videos which have wrong speed
    _C.DATA.ONLOAD_RESIZE = True  # Whether to automatically resize with ffmpeg when using torchvision.
    _C.DATA.MODALITY = "RGB"  # input modality, RGB or Depth
    _C.DATA.MAX_DEPTH = 0.659631  # Max disparity value in the dataset
    _C.DATA.MIN_DEPTH = 0.089223  # Min disparity value in the dataset
    _C.DATA.NTU_SPLIT = "xsub"  # cross subject or cross view
    _C.MODEL.OMNIVORE_FROZEN_STAGE = 0  # 0 means only patch embed and classifier are trainable
    pass
