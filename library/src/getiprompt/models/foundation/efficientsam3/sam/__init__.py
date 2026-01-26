# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# TODO: Consider moving these SAM1-style decoder components to a shared location
# if SAM3 also needs to support SAM1-style interactive prompting in the future.
# These components provide the SAM1-style mask decoder, prompt encoder, and
# two-way transformer that EfficientSAM3 uses for its decoder.

from .mask_decoder import MaskDecoder as MaskDecoder
from .prompt_encoder import PromptEncoder as PromptEncoder
from .transformer import TwoWayTransformer as TwoWayTransformer
