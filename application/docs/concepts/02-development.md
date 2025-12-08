# Architecture

The backend uses a 3-layer architecture with strict unidirectional dependencies. This keeps the codebase modular, testable, and easy to reason about.

## Layers

| Layer | Responsibility | Can Import | Cannot Import |
|-------|----------------|------------|---------------|
| **API** | Provide the interface for interacting with the zero-shot learning framework | Runtime, Domain | — |
| **Runtime** | Execute zero-shot inference pipelines, manage video streams, broadcast predictions | Domain | API |
| **Domain** | Persist projects, prompts, labels, and model configurations | None | API, Runtime |

> **Rule:** Dependencies flow top-to-bottom only. Never import from a layer above.

## Testing

The layered architecture simplifies testing by providing clear boundaries for mocking.

| Type | Scope | Approach |
|------|-------|----------|
| Unit | Single layer | Mock dependencies from the layer below |
| Integration | Data layer | Test repositories against a real database |

## WebRTC Networking & Topologies

Geti Prompt uses WebRTC for low-latency video streaming. We support three main network topologies:

### 1. Local Network
*   **Scenario:** Running on `localhost` or devices on the same LAN.
*   **Mechanism:** Docker Bridge network with Port Mapping.
*   **Configuration:** Must advertise the Host IP (e.g., `127.0.0.1`) to route traffic from Host to Container.

### 2. Cloud / NAT / Load Balancer
*   **Scenario:**
    *   Public network (Public IP)
    *   Private network (behind a Reverse Proxy / Load Balancer)
*   **Mechanism:**
    *   **Automatic (STUN):** Use a public STUN server to discover the public IP. Preferred for dynamic IPs.
    *   **Manual (Advertised IP):** Explicitly set the Public IP or Domain. Preferred for Load Balancers or static IPs.
*   **Requirement:** UDP ports must be open in the firewall/security group. See [Port Range Requirement](#port-range-requirement) below for instructions on constraining the UDP port range.

### 3. Restrictive Firewalls
*   **Scenario:** Strict firewalls blocking UDP or non-standard ports.
*   **Mechanism:** Traffic is relayed through a TURN server (Coturn) over an allowed TCP port (typically 443).
*   **Requirement:** A running instance of Coturn.


---

## Architecture: Server-Side ICE Gathering

Geti Prompt uses a **Server-Side Gathering** approach.

1.  **Backend Config:** The backend is configured with `ICE_SERVERS` (STUN/TURN) and/or `WEBRTC_ADVERTISE_IP`.
2.  **Gathering:** When a connection starts, the backend uses these settings to gather its own ICE candidates (Host, Srflx, Relay).
3.  **Offer:** The backend sends these candidates to the client in the SDP Answer.
4.  **Client Role:** The client is "passive". It does not need to know about TURN servers or gather its own complex candidates; it simply connects to the candidates provided by the backend.

This simplifies the frontend logic and keeps network configuration centralized in the backend.

---

## Configuration Guide

> **Important:** The following configuration options (`enable-stun`, `webrtc-advertise-ip`, `enable-coturn`) are mutually exclusive. Please use only one at a time.

### 1. Local Network
Use this for local development on `localhost` or within the same LAN.
**Note:** Because Docker runs in an isolated network, you must explicitly advertise the host's IP so the browser can reach the container.

**For Localhost (Same Machine):**
```bash
just webrtc-advertise-ip="127.0.0.1" run-image
```

**For LAN (Different Devices):**
```bash
# Replace with your machine's LAN IP (e.g., 192.168.1.50)
just webrtc-advertise-ip="YOUR_LAN_IP" run-image
```

**Using Docker:**
```bash
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-51000:50000-51000/udp \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="127.0.0.1" \
    <image_name>
```

### 2. Public Cloud (STUN)
Best for dynamic IPs. The backend asks a public STUN server (Google) for its IP.

**Using Just:**
```bash
just enable-stun=true run-image
```

**Using Docker:**
```bash
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-51000:50000-51000/udp \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "stun:stun.l.google.com:19302"}]' \
    <image_name>
```

### 3. Public Cloud (Manual IP)
Best for Load Balancers or static IPs. Explicitly set the Public IP or Domain.

**Using Just:**
```bash
just webrtc-advertise-ip="203.0.113.10" run-image
```

**Using Docker:**
```bash
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-51000:50000-51000/udp \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="203.0.113.10" \
    <image_name>
```

### 4. Restrictive Firewalls (TURN)
Use this if UDP is blocked. Traffic is relayed through a local TURN server on port 443.

**Using Just:**
```bash
# Start TURN server
just run-coturn
# Run App
just enable-coturn=true run-image
```

**Using Docker:**
```bash
# Start TURN server (simplified)
docker run --rm -d --network=host --name coturn-server quay.io/coturn/coturn -n --listening-port=443 --external-ip=$(curl -s ifconfig.me) --user=user:password --realm=my-realm --no-udp

# Run App
docker run --rm \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "turn:<external_ip>:443?transport=tcp", "username": "user", "credential": "password"}]' \
    <image_name>
```

### Port Range Requirement
For scenarios 2 and 3 (Public Cloud), you must allow UDP traffic on the configured port range (default: 50000-51000).
```bash
# The justfile automatically handles this with:
# --sysctl net.ipv4.ip_local_port_range="50000 51000"
# --publish 50000-51000:50000-51000/udp
```
