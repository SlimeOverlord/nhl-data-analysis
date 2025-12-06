#!/bin/bash
# Run the Docker container for the serving application
docker run -p 5050:5000 -e WANDB_API_KEY=${WANDB_API_KEY} ift6758-serving
