#!/bin/bash

file_list=(
    "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/for_paper/SharingRatio9%/latency_predictor_ttft-L20-rps7.5-trained_on_L20-7_merged-data-iter4-20251026_072048/filtered-aibrix-gateway-plugins.log.csv"
    
    "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/for_paper/SharingRatio28%/latency_predictor_ttft-L20-rps8-trained_on_L20-7_merged-data-iter6-20251027_061247/filtered-aibrix-gateway-plugins.log.csv"

    "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/for_paper/SharingRatio47%/latency_predictor_ttft-L20-rps8-trained_on_L20-7_merged-data-iter3-20251027_052806/filtered-aibrix-gateway-plugins.log.csv"

    "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/for_paper/SharingRatio71%/latency_predictor_ttft-L20-rps8-trained_on_L20-7_merged-data-iter4-20251027_073910/filtered-aibrix-gateway-plugins.log.csv"

    "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/for_paper/MixedSharingRatio/latency_predictor_ttft-L20-rps10-trained_on_L20-7_merged-data-iter2-20251027_204411/filtered-aibrix-gateway-plugins.log.csv"
)

for filtered_file_path in "${file_list[@]}"; do
    python plot_actual_vs_predicted_by_iteration.py ${filtered_file_path} & 
    python plot_prediction_accuracy_by_iteration.py ${filtered_file_path} &
    python plot_ttft_trends_by_iteration.py ${filtered_file_path} &
done
