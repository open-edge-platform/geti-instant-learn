# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path


class SAMModelName(Enum):
    """Enum for SAM model types."""

    SAM = "SAM"
    MOBILE_SAM = "MobileSAM"
    EFFICIENT_VIT_SAM = "EfficientViT-SAM"
    SAM_HQ = "SAM-HQ"
    SAM_HQ_TINY = "SAM-HQ-tiny"
    SAM_FAST = "SAM-Fast"
    SAM2_TINY = "SAM2-tiny"
    SAM2_SMALL = "SAM2-small"
    SAM2_BASE = "SAM2-base"
    SAM2_LARGE = "SAM2-large"


class PipelineName(Enum):
    """Enum for pipeline types."""

    GROUNDED_SAM = "GroundedSAM"
    MATCHER = "Matcher"
    PER_SAM = "PerSAM"
    PER_DINO = "PerDino"
    PER_SAM_MAPI = "PerSAMMAPI"
    SOFT_MATCHER = "SoftMatcher"


class DatasetName(Enum):
    """Enum for dataset names."""

    PERSEG = "PerSeg"
    LVIS = "lvis"
    LVIS_VALIDATION = "lvis_validation"


class DINOv3BackboneSize(Enum):
    """Enum for DINOv3 backbone size variants."""

    SMALL = "small"
    SMALL_PLUS = "small-plus"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"


DATA_PATH = Path("~/data").expanduser()
DINOV3_WEIGHTS_PATH = DATA_PATH.joinpath("dinov3_weights")
DINOV3_TXT_HEAD_FILENAME = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

# DINOv3 backbone model mapping
DINOV3_BACKBONE_MAP = {
    DINOv3BackboneSize.SMALL.value: "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    DINOv3BackboneSize.SMALL_PLUS.value: "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    DINOv3BackboneSize.BASE.value: "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    DINOv3BackboneSize.LARGE.value: "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    DINOv3BackboneSize.HUGE.value: "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
}

MODEL_MAP = {
    SAMModelName.SAM: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha_sum": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    },
    SAMModelName.SAM2_TINY: {  # 1024x1024 input resolution
        "registry_name": "vit_t",
        "local_filename": "sam2.1_hiera_tiny.pt",
        "config_filename": "sam2.1_hiera_t.yaml",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sha_sum": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    },
    SAMModelName.SAM2_SMALL: {  # 1024x1024 input resolution
        "registry_name": "vit_s",
        "local_filename": "sam2.1_hiera_small.pt",
        "config_filename": "sam2.1_hiera_s.yaml",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sha_sum": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    },
    SAMModelName.SAM2_BASE: {  # 1024x1024 input resolution
        "registry_name": "vit_b",
        "local_filename": "sam2.1_hiera_base_plus.pt",
        "config_filename": "sam2.1_hiera_b+.yaml",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sha_sum": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
    },
    SAMModelName.SAM2_LARGE: {  # 1024x1024 input resolution
        "registry_name": "vit_l",
        "local_filename": "sam2.1_hiera_large.pt",
        "config_filename": "sam2.1_hiera_l.yaml",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sha_sum": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    },
    SAMModelName.MOBILE_SAM: {  # 1024x1024 input resolution
        "registry_name": "vit_t",
        "local_filename": "mobile_sam.pt",
        "download_url": "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt",
        "sha_sum": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    },
    SAMModelName.EFFICIENT_VIT_SAM: {  # 512x512 input resolution
        "registry_name": "efficientvit-sam-l0",
        "local_filename": "efficientvit_sam_l0.pt",
        "download_url": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt",
        "sha_sum": "c4f994b01a16d48bcf2fbbb089448cfbf58fae5811edfa8113c953b8b8cc64b8",
    },
    SAMModelName.SAM_HQ: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_hq_vit_h.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "sha_sum": "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
    },
    SAMModelName.SAM_HQ_TINY: {  # 1024x1024 input resolution
        "registry_name": "vit_tiny",
        "local_filename": "sam_hq_vit_tiny.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
        "sha_sum": "0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196",
    },
    SAMModelName.SAM_FAST: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha_sum": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    },
}


MAPI_ENCODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_encoder.xml")
MAPI_DECODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_decoder.xml")

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


# ImageNet templates for zero shot classification
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "a photo of {}.",
    "a satellite photo of {}.",
    "a medical photo of {}.",
]
