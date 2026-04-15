# Video Streaming

Geti Instant Learn uses MJPEG over HTTP for real-time video streaming of inference results from the backend to the browser.

## Overview

The backend streams annotated inference frames as a continuous MJPEG stream over a standard HTTP connection. The browser displays the stream using a simple `<img>` element. No special network configuration, UDP ports, or relay servers are required — the stream uses the same HTTP port as the REST API (default: 9100).

## How It Works

1. The browser requests `GET /api/v1/projects/{project_id}/stream`
2. The backend responds with `Content-Type: multipart/x-mixed-replace; boundary=frame`
3. Each frame is an independent JPEG image sent as a multipart chunk
4. The browser's `<img>` element natively renders each frame as it arrives

## Configuration

| Environment Variable | Default | Description |
| -------------------- | ------- | ----------- |
| `MJPEG_QUALITY`      | `80`    | JPEG compression quality (1-100). Lower = smaller frames, more artifacts |
| `MJPEG_MAX_FPS`      | `30`    | Maximum frames per second. Actual FPS is limited by inference speed |

Quality and FPS can also be overridden per-connection via query parameters:

```
GET /api/v1/projects/{project_id}/stream?quality=60&fps=15
```

## Deployment

MJPEG streaming works over plain HTTP with no special requirements:

```bash
# Local development
docker run --rm \
    -p 9100:9100 \
    <image_name>

# Behind a reverse proxy — no special proxy configuration needed
# The stream uses standard HTTP chunked transfer on port 9100
```

### Network Requirements

| Port   | Protocol | Purpose                          |
| ------ | -------- | -------------------------------- |
| `9100` | TCP      | HTTP API + MJPEG video streaming |

No UDP ports, STUN servers, or TURN relays are needed.

## Bandwidth Estimates

| Resolution | Quality | FPS | Bandwidth (approx.) |
| ---------- | ------- | --- | ------------------- |
| 640×480    | 80      | 30  | ~8 Mbps             |
| 1280×720   | 80      | 30  | ~20 Mbps            |
| 1280×720   | 60      | 15  | ~7 Mbps             |
| 1920×1080  | 80      | 30  | ~40 Mbps            |

For 1-10 simultaneous connections on a LAN, bandwidth is not a concern. For constrained networks, lower the quality and FPS via query parameters or environment variables.
