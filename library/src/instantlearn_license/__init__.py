# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""License management for instantlearn.

This package can be imported without triggering the license gate,
allowing backend services to check/accept licenses before using instantlearn.
"""

from instantlearn_license.service import (
    LICENSE_MESSAGE,
    LicenseConfig,
    LicenseNotAcceptedError,
    LicenseService,
)

__all__ = [
    "LICENSE_MESSAGE",
    "LicenseConfig",
    "LicenseNotAcceptedError",
    "LicenseService",
]
