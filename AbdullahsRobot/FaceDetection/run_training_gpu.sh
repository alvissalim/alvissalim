#!/bin/bash
script_directory=$(dirname "$(realpath $0)")

# Make sure that 
docker run --rm --shm-size=8096m --name pytorch -v ${script_directory}:/workspace/face_detector/codes -v ${FACE_DATASET_PATH? Need to set FACE_DATASET_PATH}:/workspace/face_detector/dataset --gpus all -it pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

