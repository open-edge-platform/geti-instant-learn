# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Instant Learn.

This package requires license acceptance before use. The license can be
accepted by either:
1. Setting environment variable INSTANTLEARN_LICENSE_ACCEPTED=1
2. Running the CLI interactively and accepting when prompted
"""

from instantlearn_license import LicenseService

LicenseService().require_accepted()
