#!/bin/bash
set -e

if [ -n "$EXTERNAL_IP" ]; then
    echo "Configuring external-ip to $EXTERNAL_IP"
    echo "external-ip=$EXTERNAL_IP" >> /etc/turnserver.conf
fi

exec "$@"
