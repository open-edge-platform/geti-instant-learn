# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Instant Learn.

This package requires license acceptance before use. The license can be
accepted by either:
1. Setting environment variable INSTANTLEARN_LICENSE_ACCEPTED=1
2. Running the CLI interactively and accepting when prompted
"""

from instantlearn_license import LicenseService

# allow utils module to be imported without license check
_is_utils_import = getattr(__import__("instantlearn.utils", fromlist=[""]), "_IMPORTING_FROM_UTILS", False)

if not _is_utils_import:
    LicenseService().require_accepted()
