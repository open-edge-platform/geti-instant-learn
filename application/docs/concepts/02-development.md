# Architecture

The backend uses a 3-layer architecture with strict unidirectional dependencies. This keeps the codebase modular, testable, and easy to reason about.

## Layers

| Layer | Responsibility | Can Import | Cannot Import |
| ----- | -------------- | ---------- | ------------- |
| **API** | Provide the interface for interacting with the zero-shot learning framework | Runtime, Domain | — |
| **Runtime** | Execute zero-shot inference pipelines, manage video streams, broadcast predictions | Domain | API |
| **Domain** | Persist projects, prompts, labels, and model configurations | None | API, Runtime |

> **Rule:** Dependencies flow top-to-bottom only. Never import from a layer above.

## Testing

The layered architecture simplifies testing by providing clear boundaries for mocking.

| Type        | Scope        | Approach                                   |
| ----------- | ------------ | ------------------------------------------ |
| Unit        | Single layer | Mock dependencies from the layer below     |
| Integration | Data layer   | Test repositories against a real database  |

## WebRTC Networking

Geti Prompt uses WebRTC for real-time video streaming between the browser (UI) and the backend. For WebRTC to work, the browser needs to know how to reach the backend's media server.

### The Challenge

The backend runs inside Docker or behind network infrastructure (NAT, load balancers, firewalls). The browser cannot directly connect to internal IPs like `172.17.0.2` or `10.0.0.5`. We need to tell the browser which address to use.

### The Solution

Configure the backend to advertise a reachable address. Choose the method based on your deployment:

| Scenario | Challenge | Solution | Configuration |
| -------- | --------- | -------- | ------------- |
| [Local development](#local-network) | Docker container has internal IP | Tell backend to advertise `127.0.0.1` or LAN IP | `WEBRTC_ADVERTISE_IP="127.0.0.1"` |
| [Cloud (public IP unknown)](#cloud-with-auto-discovery) | Server IP changes dynamically | Backend discovers its public IP via STUN | `ICE_SERVERS='[{"urls": "stun:stun.l.google.com:19302"}]'` |
| [Cloud (public IP known)](#cloud-with-manual-ip) | Behind load balancer or have static IP/DNS | Tell backend to advertise the public IP or DNS | `WEBRTC_ADVERTISE_IP="203.0.113.10"` |
| [Restrictive network](#restrictive-firewall-turn) | Firewall blocks UDP traffic | Relay all traffic through TURN server on TCP 443 | `ICE_SERVERS='[{"urls": "turn:...", ...}]'` |

> **Note:** Use only one configuration method at a time.

---

### Port Requirements

When running without a relay server (TURN), the container's UDP ports must be accessible from the outside for direct media streaming. Whether running locally or in the cloud, you must map these ports from the container to the host to allow media traffic to flow.

| Ports         | Protocol | Purpose      |
| ------------- | -------- | ------------ |
| `50000-50050` | UDP      | WebRTC media |

We constrain the UDP port range by setting `net.ipv4.ip_local_port_range` in the container's Linux network namespace. This limits ephemeral ports to `50000-50050`, making firewall rules predictable.

---

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

**Solution:** Route all traffic through a TURN relay server on TCP port 443 (usually allowed).

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
