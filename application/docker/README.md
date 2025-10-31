# Docker Container with Geti Prompt
This directory contains the Dockerfile and related configuration to build a Docker container for running Geti Prompt.

## Prerequisites
- Docker installed on your machine. You can download it from [Docker's official website](https://www.docker.com/get-started).
- Just installed on your machine. You can download it from [Just's official website](https://github.com/casey/just).

## Building the Docker Image
To build the Docker image for Geti Prompt, navigate to application directory in your terminal and run the following command:

```bash
just build-image
```

## Running the Docker Container

```
just [OPTIONS] run-image
```

## Options

| Option | Default | Description |
|:--------|:---------|:-------------|
| `host-port` | `9100` | Host port mapping |
| `container-port` | `9100` | Internal container port |
| `docker-volume` | *(none)* | Directory for persistent data (mounted to `WORKDIR_PATH/data`) |
| `webcam-device` | `/dev/video0` | Host webcam device path (if available) |

To run the Docker container with default parameters, use the following command. 
The Docker image will be built automatically if it doesn't exist yet:

```bash
just run-image
```

Then navigate to `http://localhost:9100` in your web browser to access Geti Prompt.
