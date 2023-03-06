#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.PATH_CACHE = False
    _C.DATA.ONLOAD_RESIZE = True  # Whether to automatically resize with ffmpeg when using torchvision.
    _C.DATA.AMARV_PERS = "random"
    _C.TRAIN.TBOARD_VIDEO_EPOCH = 0
    _C.DEBUG_ITERS = -1

    _C.TENSORBOARD.CWRECA = CfgNode()
    _C.TENSORBOARD.CWRECA.ENABLE = False
    _C.TENSORBOARD.CWPREC = CfgNode()
    _C.TENSORBOARD.CWPREC.ENABLE = False

    _C.MODEL.RET_FEATS = False
