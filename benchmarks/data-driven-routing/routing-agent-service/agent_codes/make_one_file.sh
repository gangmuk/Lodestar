#!/bin/bash

one_file="one_file.py"

cat simpler_contextual_bandit.py > "$one_file"
cat logger.py >> "$one_file"
cat offline_routing_agent.py >> "$one_file"
cat preprocess.py >> "$one_file"
cat data_normalizer.py >> "$one_file"
cat data_processor.py >> "$one_file"
cat encoding.py >> "$one_file"

echo "Combined files into $one_file"