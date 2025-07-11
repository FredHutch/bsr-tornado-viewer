#!/bin/bash
set -e

docker build --platform linux/amd64 . -t tornado
docker run --platform linux/amd64 -p 8000:8000 tornado