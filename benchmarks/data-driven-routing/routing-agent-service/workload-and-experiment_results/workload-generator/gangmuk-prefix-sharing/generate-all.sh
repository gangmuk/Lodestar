#!/bin/bash

python generate-prefix-workload.py config_sharing10%/config.json &
python generate-prefix-workload.py config_sharing30%/config.json &
python generate-prefix-workload.py config_sharing50%/config.json &
python generate-prefix-workload.py config_sharing70%/config.json &
python generate-prefix-workload.py text_to_sql/config.json