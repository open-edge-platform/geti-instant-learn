#!/bin/bash
set -e

# Check if EXTERNAL_IP environment variable is set
if [ -n "$EXTERNAL_IP" ]; then
    echo "Configuring external-ip to $EXTERNAL_IP"
    # Append the external-ip configuration to the config file
    echo "external-ip=$EXTERNAL_IP" >> /etc/turnserver.conf
fi

# Execute the passed command (turnserver)
exec "$@"
