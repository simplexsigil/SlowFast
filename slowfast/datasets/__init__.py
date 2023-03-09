#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # noqa
from .amarv import Amarv # noqa
from .ssv2 import Ssv2  # noqa
from .amarv_tmp import Amarvtmp  # noqa
from .nturgbd import Nturgbd  # noqa
from .nturgbd_tmp import Nturgbdtmp  # noqa
from .nturgbd_multimodal import Nturgbdmultimodal  # noqa

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
