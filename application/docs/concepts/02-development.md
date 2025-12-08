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

## WebRTC Networking

Geti Prompt uses WebRTC for low-latency video streaming. The backend handles all ICE candidate gathering, so client configuration is minimal.

### How It Works

1. The backend gathers ICE candidates using configured STUN/TURN servers or an advertised IP.
2. Candidates are sent to the client in the SDP Answer.
3. The client connects using the provided candidates—no additional setup required.

### Network Topologies

| Topology | Use Case | Method |
|----------|----------|--------|
| [Local](#local-network) | Development on localhost or LAN | Advertised IP |
| [Cloud (STUN)](#cloud-with-stun) | Dynamic public IPs | STUN server |
| [Cloud (Static IP)](#cloud-with-static-ip) | Load balancers, static IPs | Advertised IP |
| [Restrictive Firewall](#restrictive-firewall-turn) | UDP blocked | TURN relay |

> **Note:** These options are mutually exclusive. Use only one configuration method at a time.

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `WEBRTC_ADVERTISE_IP` | IP address to advertise to clients |
| `ICE_SERVERS` | JSON array of STUN/TURN server configurations |

### Port Requirements

Cloud deployments require UDP ports for WebRTC media traffic:

| Ports | Protocol | Purpose |
|-------|----------|---------|
| `9100` | TCP | HTTP API |
| `50000-51000` | UDP | WebRTC media |

---

## Deployment Examples

### Local Network

For development on localhost or within a LAN. Docker runs in an isolated network, so you must advertise the host IP.

<details>
<summary><strong>Localhost</strong></summary>

```bash
# Using just
just webrtc-advertise-ip="127.0.0.1" run-image

# Using Docker
docker run --rm \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="127.0.0.1" \
    <image_name>
```

</details>

<details>
<summary><strong>LAN (other devices)</strong></summary>

```bash
# Using just (replace with your LAN IP)
just webrtc-advertise-ip="192.168.1.50" run-image

# Using Docker
docker run --rm \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="192.168.1.50" \
    <image_name>
```

</details>

### Cloud with STUN

For dynamic public IPs. The backend discovers its IP via a public STUN server.

```bash
# Using just
just enable-stun=true run-image

# Using Docker
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-51000:50000-51000/udp \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "stun:stun.l.google.com:19302"}]' \
    <image_name>
```

### Cloud with Static IP

For load balancers or static public IPs.

```bash
# Using just
just webrtc-advertise-ip="203.0.113.10" run-image

# Using Docker
docker run --rm \
    --sysctl net.ipv4.ip_local_port_range="50000 51000" \
    -p 50000-51000:50000-51000/udp \
    -p 9100:9100 \
    -e WEBRTC_ADVERTISE_IP="203.0.113.10" \
    <image_name>
```

### Restrictive Firewall (TURN)

For networks blocking UDP. Traffic relays through a TURN server on TCP port 443.

**Step 1: Start the TURN server**

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

**Step 2: Start the application**

```bash
# Using just
just enable-coturn=true run-image

# Using Docker (replace <external_ip> with your server's public IP)
docker run --rm \
    -p 9100:9100 \
    -e ICE_SERVERS='[{"urls": "turn:<external_ip>:443?transport=tcp", "username": "user", "credential": "password"}]' \
    <image_name>
```
