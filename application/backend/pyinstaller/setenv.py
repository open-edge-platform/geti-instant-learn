# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

bundle_dir = Path(getattr(sys, "_MEIPASS", ""))
dotenv_path = os.path.join(bundle_dir, ".env")
print(f"Loading dotenv file {dotenv_path}")
loaded = load_dotenv(dotenv_path)
if loaded:
    print("Dotenv loaded")
else:
    print("No dotenv file found")

sample_dataset_dir = bundle_dir.parent / "InitialData" / "templates" / "datasets"
if sample_dataset_dir.exists():
    os.environ.setdefault("SAMPLE_DATASET_DIR", str(sample_dataset_dir))
