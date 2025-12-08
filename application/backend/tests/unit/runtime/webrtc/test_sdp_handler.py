# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from aiortc.sdp import SessionDescription

from runtime.webrtc.sdp_handler import SDPHandler


@pytest.fixture
def sdp_handler():
    return SDPHandler()


@pytest.mark.asyncio
async def test_mangle_sdp_no_advertise_ip(sdp_handler):
    sdp = "v=0\r\n"
    mangled = await sdp_handler.mangle_sdp(sdp, None)
    assert mangled == sdp


@pytest.mark.asyncio
async def test_mangle_sdp_invalid_sdp(sdp_handler):
    sdp = "invalid-sdp"
    with patch("runtime.webrtc.sdp_handler.SessionDescription.parse", side_effect=Exception("Parse error")):
        mangled = await sdp_handler.mangle_sdp(sdp, "1.2.3.4")
        assert mangled == sdp


@pytest.mark.asyncio
async def test_mangle_sdp_success(sdp_handler):
    # Create a simple SDP with host candidates
    sdp = (
        "v=0\r\n"
        "o=- 123 456 IN IP4 192.168.1.1\r\n"
        "s=-\r\n"
        "c=IN IP4 192.168.1.1\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
        "c=IN IP4 192.168.1.1\r\n"
        "a=candidate:1 1 UDP 2122260223 192.168.1.1 50000 typ host\r\n"
    )
    advertise_ip = "203.0.113.10"

    mangled = await sdp_handler.mangle_sdp(sdp, advertise_ip)

    parsed = SessionDescription.parse(mangled)
    assert parsed.host == advertise_ip
    assert parsed.media[0].host == advertise_ip
    assert parsed.media[0].ice_candidates[0].ip == advertise_ip


@pytest.mark.asyncio
async def test_mangle_sdp_with_domain(sdp_handler):
    sdp = "v=0\r\n"
    domain = "example.com"
    resolved_ip = "1.2.3.4"

    with (
        patch.object(sdp_handler, "resolve_hostname", return_value=resolved_ip) as mock_resolve,
        patch.object(sdp_handler, "_mangle_sdp_with_ip", return_value="mangled-sdp") as mock_mangle,
    ):
        result = await sdp_handler.mangle_sdp(sdp, domain)

        mock_resolve.assert_called_once_with(domain)
        mock_mangle.assert_called_once_with(sdp, resolved_ip)
        assert result == "mangled-sdp"
