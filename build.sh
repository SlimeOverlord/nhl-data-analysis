#!/bin/bash
# Build the Docker image for the serving application
docker build -t ift6758-serving -f Dockerfile.serving .
