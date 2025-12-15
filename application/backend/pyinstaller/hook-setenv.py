# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform

system = platform.system()
print("Setup Hook: Detected operating system:", system)

if system == "Windows":
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise OSError("LOCALAPPDATA environment variable is not set.")

    import ctypes

    GetCurrentPackageFamilyName = ctypes.windll.kernel32.GetCurrentPackageFamilyName  # type: ignore[attr-defined]
    GetCurrentPackageFamilyName.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
    GetCurrentPackageFamilyName.restype = ctypes.c_long

    length = ctypes.c_uint(256)
    package_family_name_buffer = ctypes.create_unicode_buffer(256)

    result = GetCurrentPackageFamilyName(ctypes.byref(length), package_family_name_buffer)
    if result == 0:
        package_family_name = package_family_name_buffer.value
        print("Setup Hook: Application runs in a UWP context. Package Family Name:", package_family_name)

        app_data_folder = os.path.join(local_app_data, "Packages", package_family_name, "LocalState")

        print("Setup Hook: Using local state folder:", app_data_folder)
        os.environ["DB_DATA_DIR"] = app_data_folder

        print("Setup Hook: Writing log to:", app_data_folder)
        os.environ["LOGS_DIR"] = app_data_folder
    else:
        print("Setup Hook: Application doesn't run in a UWP context; skipping folder setup.")
