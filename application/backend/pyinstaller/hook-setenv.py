# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil


def _main() -> None:  # noqa: C901
    system = platform.system()
    print("Setup Hook: Detected operating system:", system)

    if system == "Windows":
        local_app_data = os.getenv("LOCALAPPDATA")
        if not local_app_data:
            raise OSError("LOCALAPPDATA environment variable is not set.")

        import ctypes  # noqa: PLC0415

        def _get_current_package_family_name() -> str | None:
            GetCurrentPackageFamilyName = ctypes.windll.kernel32.GetCurrentPackageFamilyName  # type: ignore[attr-defined]
            GetCurrentPackageFamilyName.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
            GetCurrentPackageFamilyName.restype = ctypes.c_long

            length = ctypes.c_uint(256)
            buffer = ctypes.create_unicode_buffer(256)

            result = GetCurrentPackageFamilyName(ctypes.byref(length), buffer)
            if result != 0:
                return None
            return buffer.value

        def _get_current_package_path() -> str | None:
            GetCurrentPackagePath = ctypes.windll.kernel32.GetCurrentPackagePath  # type: ignore[attr-defined]
            GetCurrentPackagePath.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
            GetCurrentPackagePath.restype = ctypes.c_long

            length = ctypes.c_uint(256)
            buffer = ctypes.create_unicode_buffer(256)

            result = GetCurrentPackagePath(ctypes.byref(length), buffer)
            if result != 0:
                return None
            return buffer.value

        def _copy_initial_data(app_data_folder: str) -> None:
            package_path = _get_current_package_path()
            print("Setup Hook: Application package path:", package_path)
            if not package_path:
                return
            initial_data_path = os.path.join(package_path, "InitialData")
            if not os.path.exists(initial_data_path):
                return
            for item in os.listdir(initial_data_path):
                source_path = os.path.join(initial_data_path, item)
                destination_path = os.path.join(app_data_folder, item)
                if os.path.exists(destination_path):
                    continue
                print("Setup Hook: Copying initial data:", item, " Destination:", destination_path)
                try:
                    shutil.copytree(source_path, destination_path)
                except Exception as e:
                    print(f"Setup Hook: Failed to copy '{source_path}' to '{destination_path}': {e}")

        package_family_name = _get_current_package_family_name()
        if not package_family_name:
            print("Setup Hook: Application doesn't run in a UWP context; skipping folder setup.")
            return

        print("Setup Hook: Application runs in a UWP context. Package Family Name:", package_family_name)

        app_data_folder = os.path.join(local_app_data, "Packages", package_family_name, "LocalState")

        print("Setup Hook: Using local state folder:", app_data_folder)
        os.environ["DB_DATA_DIR"] = app_data_folder

        print("Setup Hook: Writing log to:", app_data_folder)
        os.environ["LOGS_DIR"] = app_data_folder

        _copy_initial_data(app_data_folder)


_main()
