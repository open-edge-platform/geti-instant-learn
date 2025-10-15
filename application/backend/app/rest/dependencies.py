# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import Request

from webrtc.manager import WebRTCManager


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager
