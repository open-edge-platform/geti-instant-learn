# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

print("Loading dotenv file")
from dotenv import load_dotenv
loaded = load_dotenv()
if loaded:
    print("Dotenv loaded")
else:
    print("No dotenv file found")