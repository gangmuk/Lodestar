#!/bin/bash

python generate-prefix-workload.py Sharing10%/config.json &
python generate-prefix-workload.py Sharing30%/config.json &
python generate-prefix-workload.py Sharing50%/config.json &
python generate-prefix-workload.py Sharing70%/config.json &
python generate-prefix-workload.py MixedSharingRatio10_30_50_70%