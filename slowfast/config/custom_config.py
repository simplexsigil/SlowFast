#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.PATH_CACHE = False
    _C.DATA.ONLOAD_RESIZE = True  # Whether to automatically resize with ffmpeg when using torchvision.
    _C.DATA.AMARV_PERS = "random"
    _C.TRAIN.TBOARD_VIDEO_EPOCH = 0
