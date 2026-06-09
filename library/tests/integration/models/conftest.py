"""Shared fixtures for model integration tests."""

import pytest

from instantlearn.components.encoders import base as encoder_base
from instantlearn.utils.constants import Backend


@pytest.fixture(autouse=True)
def use_timm_for_dinov3_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force DINOv3 integration tests to use the TIMM encoder backend.

    Production constructors still default to Hugging Face. This fixture keeps
    the temporary integration-test workaround local to this test package.
    """

    original_load_image_encoder = encoder_base.load_image_encoder

    def _load_image_encoder(*args: object, **kwargs: object):
        model_id = kwargs.get("model_id")
        if model_id is None and args:
            model_id = args[0]

        if isinstance(model_id, str) and model_id.startswith("dinov3"):
            kwargs["backend"] = Backend.TIMM

        return original_load_image_encoder(*args, **kwargs)

    monkeypatch.setattr(encoder_base, "load_image_encoder", _load_image_encoder)