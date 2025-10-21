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
To run the Docker container, use the following command:

```bash
just run-image
```

Then navigate to `http://localhost:9100` in your web browser to access Geti Prompt.

## Parametrizing the Docker Container
You can customize the behavior of the Docker container using the following environment variables:

### Port mapping
The `HOST_PORT` environment variable specifies the port on the host machine to be used. Defaults to `CONTAINER_PORT`.  
The `CONTAINER_PORT` environment variable defines the port inside the container where the application runs. Default is `9100`.

### Docker volume for persistent storage
Environment variable `DOCKER_VOLUME` is used to tell Docker volume name for persistent storage. Default is empty string (no volume).
If specified, the volume will be mounted to `WORKDIR_PATH/data` inside the container. 

### Webcam access
Environment variable `WEBCAM_DEVICE` is used to tell location of webcam device on host machine. Default is `/dev/video0`.
If specified, the device will be passed to the container to enable webcam access only when such device is available on host machine.
