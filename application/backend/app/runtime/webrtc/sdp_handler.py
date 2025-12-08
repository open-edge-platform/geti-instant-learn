# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import socket

from aiortc.sdp import SessionDescription

logger = logging.getLogger(__name__)


class SDPHandler:
    """Handler for SDP manipulation and processing."""

    async def resolve_hostname(self, hostname: str) -> str:
        """
        Resolve hostname to IP address.
        Returns the original string if resolution fails or if it's already an IP.
        """
        loop = asyncio.get_running_loop()
        try:
            # gethostbyname handles both domain names and IP strings (returning IP strings as-is)
            return await loop.run_in_executor(None, socket.gethostbyname, hostname)
        except Exception as exc:
            logger.warning(f"Failed to resolve hostname {hostname}: {exc}")
            return hostname

    async def mangle_sdp(self, sdp: str, advertise_ip: str) -> str:
        """
        Mangle SDP with advertise IP.
        Resolves hostname if necessary.
        """
        if not advertise_ip:
            return sdp

        resolved_ip = await self.resolve_hostname(advertise_ip)
        return self._mangle_sdp_with_ip(sdp, resolved_ip)

    def _mangle_sdp_with_ip(self, sdp: str, ip: str) -> str:
        """
        Replace local IP addresses in SDP candidates and connection lines with the advertise IP.
        Useful for 1:1 NAT scenarios where STUN is not available.
        """
        try:
            parsed_sdp = SessionDescription.parse(sdp)
        except Exception as exc:
            logger.warning(f"Failed to parse SDP for mangling: {exc}. Returning original SDP.")
            return sdp

        # Update session-level connection
        if parsed_sdp.host:
            parsed_sdp.host = ip

        for media in parsed_sdp.media:
            # Update media-level connection
            if media.host:
                media.host = ip

            # Update RTCP connection
            if media.rtcp_host:
                media.rtcp_host = ip

            # Update host candidates
            for candidate in media.ice_candidates:
                if candidate.type == "host":
                    candidate.ip = ip

        return str(parsed_sdp)
