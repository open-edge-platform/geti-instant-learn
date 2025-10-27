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
To run the Docker container with default parameters, use the following command. 
The Docker image will be built automatically if it doesn't exist yet:

```bash
just run-image
```

Then navigate to `http://localhost:9100` in your web browser to access Geti Prompt.

## Parametrizing the Docker Container
User can customize the behavior of the Docker container using the runtime variables defined in the `Justfile` in application directory.

### Port mapping
By default, Geti Prompt runs on port `9100` inside the container and this port is mapped to port `9100` on the host machine.
User can change the port mapping by setting proper variable `host-port` or `container-port`:

This command will use host port `9200` to access Geti Prompt:
```bash
just host-port=9200 run-image
```

or this command that will run Geti Prompt inside the container on port `9201`:
```bash
just container-port=9201 run-image
```

### Docker volume for persistent storage
Variable `docker-volume` is used to tell Docker name of directory used for persistent storage. Default is empty string (no volume).
If specified, the volume will be mounted to `WORKDIR_PATH/data` inside the container. 
User can define the directory as follows:

```bash
just docker-volume=/tmp/some_dir run-image
```

then data will be stored in `/tmp/some_dir` on host machine.

### Webcam access
Variable `webcam-device` is used to tell location of webcam device on host machine. Default is `/dev/video0`.
The device will be passed to the container to enable webcam access only when such device is available on host machine.

```bash
just webcam-device=/dev/video1 run-image
```
