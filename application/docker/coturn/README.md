## WebRTC Connectivity & Coturn Setup

### Understanding WebRTC Connectivity (ICE, STUN, TURN)

WebRTC is designed to establish a direct Peer-to-Peer (P2P) connection between two devices (peers) to stream video and data with minimal latency. However, devices are rarely connected directly to the open internet; they sit behind firewalls and NAT routers.

To overcome this, WebRTC uses a protocol called **ICE** (Interactive Connectivity Establishment). ICE tries to find the best path to connect peers in the following order:

1.  **Host Candidates (Local Network):**
    *   The peers try to connect directly using their local LAN IP addresses.
    *   *Works if:* Both devices are on the same network.

2.  **STUN Candidates (Public IP):**
    *   **STUN** (Session Traversal Utilities for NAT) is a lightweight protocol that tells a device "What is my public IP address?".
    *   *Works if:* You are on a standard home network or a permissive public network.

3.  **TURN Candidates (Relay):**
    *   **TURN** (Traversal Using Relays around NAT) is the fallback of last resort.
    *   If a direct connection is impossible (e.g., strict corporate firewalls, Symmetric NAT, or blocked UDP), the TURN server acts as a middleman.
    *   Peer A sends data to the TURN server -> TURN server relays it to Peer B.
    *   *Works if:* Almost always, as long as the TURN server itself is reachable.

### When Do You Need a TURN Server?

You absolutely need a TURN server in **Restrictive Network Environments**, such as:
*   **Corporate Offices:** Where firewalls block all non-standard ports and UDP traffic.
*   **3G/4G/5G Mobile Networks:** Which often use Carrier-Grade NAT (CGNAT) that blocks P2P.
*   **VPNs:** Which might interfere with local routing.

If your video stream fails to load in these environments, it is likely because the direct P2P connection is blocked, and you have no TURN server configured to relay the traffic.

### What is Coturn?

**Coturn** is a mature, [open-source implementation](https://github.com/coturn/coturn) of a TURN and STUN server. It is widely used in the industry to power WebRTC infrastructure.

In this project, we provide a Dockerized version of Coturn configured specifically to bypass strict firewalls by masquerading as standard web traffic.

### How to Use with Docker & Just

We have automated the entire setup using `just` (a command runner) and Docker.

#### 1. Start the TURN Server
Run the following command to build the Coturn Docker image and start the server:

```bash
just run-coturn
```

**What this command does:**
1.  **Builds:** Creates a lightweight Docker image installing `coturn` on Debian.
2.  **Detects IP:** Automatically finds your host machine's IP address.
3.  **Runs:** Starts the container with **Host Networking** (`--network=host`).
    *   It binds to port **443** (TCP). This is crucial because port 443 is the standard HTTPS port, which is almost never blocked by firewalls.
    *   *Note:* The port is configurable. If you need to change it, update `coturn-port` in the `Justfile` AND `listening-port` in `application/docker/coturn/turnserver.conf`.
    *   It configures the server to accept **TCP** connections (since UDP is often blocked).

#### 2. Run the Application
Once the TURN server is running, start the main application stack with the `enable-coturn` flag:

```bash
just enable-coturn=true run-image
```

**What this command does:**
1.  **Configures:** It constructs a JSON configuration string pointing to your local Coturn instance (e.g., `turn:YOUR_IP:443?transport=tcp`).
2.  **Injects:** It passes this string as the `ICE_SERVERS` environment variable to the backend container.
3.  **Serves:** The backend provides this configuration to the frontend UI. The UI then uses it to establish the WebRTC connection via your local relay.

*Note: If you run `just run-image` without the flag, the application will start without any TURN server configuration.*

#### 3. Stop the Server
When you are finished testing, stop the server to free up port 443:

```bash
just stop-coturn
```

### Configuration Details (`turnserver.conf`)

The server is pre-configured with settings in `application/docker/coturn/turnserver.conf`:

*   `listening-port=443`: Listens on the standard HTTPS port. You can change this if port 443 is already in use on your host.
*   `no-udp`: Disables UDP listeners entirely to force TCP usage.
*   `no-tcp-relay=0`: Explicitly enables TCP relaying (required for this setup).
*   `user=user:password`: **For testing purposes only.**
    *   These static credentials are provided for ease of development.
    *   **Production Warning:** In a production environment, you should **never** use static credentials. Instead, use the **TURN REST API** (Time-Limited Credentials) mechanism. This involves sharing a secret key between the backend and the TURN server to generate temporary, expiring passwords for each client session. Coturn supports this via the `use-auth-secret` configuration.
