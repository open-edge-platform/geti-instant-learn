# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import pathlib

from app.main import app

parser = argparse.ArgumentParser(description="Generate OpenAPI schema")
parser.add_argument("output_path", type=pathlib.Path, help="The path where the openapi-spec.json file will be saved.")
args = parser.parse_args()

output_file = pathlib.Path(args.output_path) / "openapi-spec.json"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as file:
    open_api = app.openapi()
    json.dump(obj=open_api, fp=file, indent=2)

print(f"OpenAPI schema generated at {output_file}")
