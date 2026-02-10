# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from dotenv import load_dotenv

dotenv_path = os.path.join(getattr(sys, "_MEIPASS", ""), ".env")
print(f"Loading dotenv file {dotenv_path}")
loaded = load_dotenv(dotenv_path)
if loaded:
    print("Dotenv loaded")
else:
    print("No dotenv file found")

# skip license check by library - license acceptance is handled through UI
os.environ.setdefault("INSTANTLEARN_SKIP_LICENSE_CHECK_ON_IMPORT", "1")
