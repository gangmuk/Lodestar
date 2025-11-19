#!/bin/bash

# processed_csv="/home/ec2-user/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/agent_codes/data/processed_data.csv"
processed_csv=$1

python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function linear_simple --ttft-slo 1000 --avg-tpot-slo 100 --ttft-reward-weight 1 --save-sampled-dataset