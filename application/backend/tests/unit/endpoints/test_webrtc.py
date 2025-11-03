# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from core.runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from dependencies import get_webrtc_manager
from exceptions.handler import custom_exception_handler
from main import app
from services.schemas.webrtc import Answer, Offer
from webrtc.manager import WebRTCManager

PROJECT_ID = uuid4()


@pytest.fixture
def fxt_client():
    # Register the global exception handler
    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def fxt_webrtc_manager():
    webrtc_manager = MagicMock(spec=WebRTCManager)
    app.dependency_overrides[get_webrtc_manager] = lambda: webrtc_manager
    return webrtc_manager


@pytest.fixture
def fxt_offer() -> Offer:
    return Offer(sdp="test_sdp", type="offer", webrtc_id="test_id")


@pytest.fixture
def fxt_answer() -> Answer:
    return Answer(sdp="test_sdp", type="answer")


class TestWebRTCEndpoints:
    def test_create_webrtc_offer_success(self, fxt_client, fxt_webrtc_manager, fxt_offer, fxt_answer):
        fxt_webrtc_manager.handle_offer.return_value = fxt_answer
        resp = fxt_client.post(f"/api/v1/projects/{PROJECT_ID}/offer", json=fxt_offer.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == fxt_answer.model_dump()
        fxt_webrtc_manager.handle_offer.assert_called_once()

    def test_create_webrtc_offer_incorrect_project(self, fxt_client, fxt_webrtc_manager, fxt_offer):
        fxt_webrtc_manager.handle_offer.side_effect = PipelineProjectMismatchError("fail")
        resp = fxt_client.post(f"/api/v1/projects/{PROJECT_ID}/offer", json=fxt_offer.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "fail" in resp.json()["detail"]
        fxt_webrtc_manager.handle_offer.assert_called_once()

    def test_create_webrtc_offer_not_active_project(self, fxt_client, fxt_webrtc_manager, fxt_offer):
        fxt_webrtc_manager.handle_offer.side_effect = PipelineNotActiveError("fail")
        resp = fxt_client.post(f"/api/v1/projects/{PROJECT_ID}/offer", json=fxt_offer.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "fail" in resp.json()["detail"]
        fxt_webrtc_manager.handle_offer.assert_called_once()

    def test_create_webrtc_offer_invalid_payload(self, fxt_client):
        resp = fxt_client.post(f"/api/v1/projects/{PROJECT_ID}/offer", json={"sdp": 123})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "detail" in resp.json()
