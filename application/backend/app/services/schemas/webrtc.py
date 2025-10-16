# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class Offer(BaseModel):
    webrtc_id: str
    sdp: str
    type: str


class Answer(BaseModel):
    sdp: str
    type: str
