#!/bin/bash

one_file="one_file.py"

cat simpler_contextual_bandit.py > "$one_file"
cat feature_normalization.py >> "$one_file"
cat logger.py >> "$one_file"
cat offline_routing_agent.py >> "$one_file"
cat preprocess.py >> "$one_file"
cat encoding.py >> "$one_file"
cat model_and_data_analysis_helper.py >> "$one_file"

echo "Combined files into $one_file"