import fs from 'node:fs';
import path from 'node:path';

import { Plugin } from '@docusaurus/types';

/**
 * Outputs a shell script which:
 * 1. Downloads the geti installer from `installerUrl`
 * 2. Extracts the tar file
 * 3. CDs into the installer directory
 * 4. Calls the installer's install command
 * */
function getInstallScriptContent(installerUrl: string) {
    return `#!/bin/bash
# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

# Check if running as root
if [ "$(id -u)" != "0" ]; then
  if command -v sudo >/dev/null 2>&1; then
    exec sudo "$0" "$@"
  else
    echo "This script must be run as root. Please either:"
    echo "1. Run this script as root (su root)"
    echo "2. Install sudo and run with sudo"
    exit 1
  fi
fi


# Define variables
PACKAGE_URL="${installerUrl}"
PACKAGE_NAME="platform_installer.tar.gz"
DOWNLOAD_DIR="/tmp/intel_geti"
INSTALLER_NAME="platform_installer"

# Create a temporary directory for downloading and extracting the package
mkdir -p "$DOWNLOAD_DIR"

# Change to the temporary directory
cd "$DOWNLOAD_DIR" || {
    echo "Failed to change directory to $DOWNLOAD_DIR"
    exit 1
}

# Download the package
echo "Downloading installer from $PACKAGE_URL..."
curl -O "$PACKAGE_URL" || {
    echo "Failed to download package"
    exit 1
}

# Extract the package
echo "Extracting installer..."
tar -xf "$PACKAGE_NAME" || {
    echo "Failed to extract package"
    exit 1
}

# Find the extracted directory
EXTRACTED_DIR=$(find . -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$EXTRACTED_DIR" ]; then
    echo "No directory found after extraction"
    exit 1
fi

# Change to the extracted directory
cd "$EXTRACTED_DIR" || {
    echo "Failed to change directory to $EXTRACTED_DIR"
    exit 1
}

# Check if the installer exists and is executable
if [ ! -x "$INSTALLER_NAME" ]; then
    echo "Installer $INSTALLER_NAME not found or not executable"
    exit 1
fi

# Check if the /data folder exists, create it if not, and set permissions
if [ ! -d "/data" ]; then
    echo "Creating /data directory..."
    sudo mkdir /data || {
        echo "Failed to create /data directory"
        exit 1
    }
    sudo chmod 750 /data || {
        echo "Failed to set permissions on /data directory"
        exit 1
    }
    echo "/data directory created and permissions set to 750"
fi

# Run the installer with sudo
echo "Running installer..."
sudo ./"$INSTALLER_NAME" install || {
    echo "Failed to run installer"
    exit 1
}

# Clean up
rm -rf "$DOWNLOAD_DIR"

exit 0;`;
}

interface Options {
    getiInstallerUrl: string;
}

export function InstallGetiScript(_, options: Options): Plugin {
    return {
        name: 'install-geti-script',

        postBuild: async ({ outDir }) => {
            const script = getInstallScriptContent(options.getiInstallerUrl);

            fs.writeFileSync(path.join(outDir, 'install-geti.sh'), script);
        },
    };
}
