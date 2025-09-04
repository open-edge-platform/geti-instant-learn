#!/bin/bash 

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

PYTHONPATH=. python3.10 -m alembic upgrade head
