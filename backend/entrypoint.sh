#!/bin/bash 

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

PYTHONPATH=. python3.10 -m uvicorn main:app --port 9100 --host 0.0.0.0 --log-config logconfig.ini
