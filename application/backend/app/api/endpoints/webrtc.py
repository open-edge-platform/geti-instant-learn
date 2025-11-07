# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WebRTC API Endpoints"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, status

from api.routers import projects_router
from dependencies import get_webrtc_manager
from domain.services.schemas.webrtc import Answer, Offer
from runtime.webrtc.manager import WebRTCManager

logger = logging.getLogger(__name__)


@projects_router.post(
    path="/{project_id}/offer",
    tags=["WebRTC"],
    response_model=Answer,
    responses={
        status.HTTP_200_OK: {"description": "WebRTC Answer"},
        status.HTTP_400_BAD_REQUEST: {"description": "Pipeline Not Active"},
    },
)
async def create_webrtc_offer(
    project_id: UUID, offer: Offer, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc_manager)]
) -> Answer:
    """Create a WebRTC offer"""
    return await webrtc_manager.handle_offer(project_id=project_id, offer=offer)
