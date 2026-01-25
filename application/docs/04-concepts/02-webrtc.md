# WebRTC Networking

Geti Prompt uses WebRTC for real-time video streaming between the browser and backend. This guide explains how to configure WebRTC for different network environments.

## Overview

WebRTC requires the browser to establish a direct connection to the backend's media server. When the backend runs in Docker or behind network infrastructure (NAT, load balancers, firewalls), the browser cannot connect to internal IPs like `172.17.0.2` or `10.0.0.5`. The backend must advertise a reachable address.

## Configuration Options

Choose the configuration method based on your deployment scenario:

| Scenario | Description | Configuration |
| -------- | ----------- | ------------- |
| [Local development](#local-network) | Docker container with internal IP | `WEBRTC_ADVERTISE_IP="127.0.0.1"` |
| [Cloud (auto-discovery)](#cloud-with-auto-discovery) | Dynamic public IP (auto-scaling, ephemeral instances) | `ICE_SERVERS='[{"urls": "stun:stun.l.google.com:19302"}]'` |
| [Cloud (static IP/DNS)](#cloud-with-manual-ip) | Known public IP or DNS behind load balancer | `WEBRTC_ADVERTISE_IP="203.0.113.10"` |
| [Restrictive firewall](#restrictive-firewall-turn) | UDP traffic blocked by firewall | `ICE_SERVERS='[{"urls": "turn:...", ...}]'` |

## Port Requirements

Direct media streaming requires UDP ports to be accessible from external clients. Map these ports from the container to the host:

| Ports         | Protocol | Purpose      |
| ------------- | -------- | ------------ |
| `50000-50050` | UDP      | WebRTC media |

The container's `net.ipv4.ip_local_port_range` is set to `50000-50050` to constrain ephemeral ports, making firewall rules predictable.

## Deployment Examples

### Local Network

**Scenario:** Docker gives the backend an internal IP (e.g., `172.17.0.2`) that the browser cannot reach.

**Solution:** Tell the backend to advertise your host machine's IP instead.

```bash
# Using just (use 127.0.0.1 for localhost, or your LAN IP for other devices)
just webrtc-advertise-ip="127.0.0.1" run-image

# Using Docker
docker run --rm \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="127.0.0.1" \
    <image_name>
```

### Cloud with Auto-discovery

**Scenario:** Your server's public IP may change (auto-scaling, ephemeral instances), so you can't hardcode it.

**Solution:** The backend queries a public STUN server to discover its own public IP automatically. A STUN server simply tells the backend "your public IP is X.X.X.X" — it doesn't relay any traffic. Configure via `ICE_SERVERS` environment variable.

```bash
# Using just
just enable-stun=true run-image

# Using Docker
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 50050" \
    -p 50000-50050:50000-50050/udp \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "stun:stun.l.google.com:19302"}]' \
    <image_name>
```

### Cloud with Manual IP

**Scenario:** The backend is behind a load balancer, reverse proxy, or has a static IP/DNS. STUN won't help because it would return the private IP.

**Solution:** Manually configure the public IP or DNS name that clients should use.

```bash
# Using just
just webrtc-advertise-ip="203.0.113.10" run-image

# Using Docker
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-50050:50000-50050/udp \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="203.0.113.10" \
    <image_name>
```

### Restrictive Firewall (TURN)

**Scenario:** Corporate firewalls block UDP traffic or non-standard ports. WebRTC media cannot get through.

**Solution:** Route all traffic through a TURN relay server on TCP port 443 (usually allowed). A TURN server forwards all media between browser and backend — use it only when direct connections fail.

#### Step 1: Start the TURN server

```bash
# Using just
just run-coturn

# Using Docker
docker run --rm -d \
    --network=host \
    --name coturn-server \
    quay.io/coturn/coturn -n \
    --listening-port=443 \
    --external-ip=$(curl -s ifconfig.me) \
    --user=user:password \
    --realm=my-realm \
    --no-udp
```

The coturn server is configured via command-line arguments:

* `--listening-port=443`: Listens on the standard HTTPS port. You can change this via the `coturn-port` variable if port 443 is already in use on your host.
* `--no-udp`: Disables UDP listeners entirely to force TCP usage.
* `--user=user:password`: **For testing purposes only.**
  * These static credentials are provided for ease of development.
  * **Production Warning:** In a production environment, you should **never** use static credentials. Instead, use the **TURN REST API** (Time-Limited Credentials) mechanism. This involves sharing a secret key between the backend and the TURN server to generate temporary, expiring passwords for each client session. Coturn supports this via the `use-auth-secret` configuration.

#### Step 2: Start the application

```bash
# Using just
just enable-coturn=true run-image

# Using Docker (replace <external_ip> with your server's public IP)
docker run --rm \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "turn:<external_ip>:443?transport=tcp", "username": "user", "credential": "password"}]' \
    <image_name>
```
