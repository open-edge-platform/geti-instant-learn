# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel


class Offer(BaseModel):
    webrtc_id: str
    sdp: str
    type: Literal["offer", "pranswer", "answer", "rollback"] = "offer"


class Answer(BaseModel):
    sdp: str
    type: str  # default is answer
