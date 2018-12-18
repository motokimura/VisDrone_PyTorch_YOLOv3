#! /bin/bash

# Set image name
IMAGE="yolov3-visdrone:latest"
if [ $# -eq 1 ]; then
    IMAGE=$1
fi

# Set project root dicrectory to map to docker 
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Run container
CONTAINER="yolov3-visdrone"

nvidia-docker run -it --rm \
	-p 8888:8888 -p 6006:6006 \
	-v ${PROJ_DIR}:/work \
	--name ${CONTAINER} \
	${IMAGE}