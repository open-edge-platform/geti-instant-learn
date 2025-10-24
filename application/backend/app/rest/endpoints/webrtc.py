# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WebRTC API Endpoints"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, status
from fastapi.exceptions import HTTPException

from core.runtime.errors import PipelineNotActiveError, PipelineProjectMismatchError
from dependencies import get_webrtc_manager as get_webrtc
from routers import projects_router
from services.schemas.webrtc import Answer, Offer
from webrtc.manager import WebRTCManager

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
    project_id: UUID, offer: Offer, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc)]
) -> Answer:
    """Create a WebRTC offer"""
    try:
        return await webrtc_manager.handle_offer(project_id=project_id, offer=offer)
    except (PipelineProjectMismatchError, PipelineNotActiveError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
