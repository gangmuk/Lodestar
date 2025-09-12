#!/bin/bash

set -e

parent_dir_list=(
    "p4096_s1024_rps20"
    "SharingRatio71%"
    "SharingRatio47%"
    "SharingRatio28%"
    "SharingRatio9%"
)

routing_policy_dir_list=(
    "prefix"
    "rl"
    "random"
)

data_file_list=(
    "data.csv"
    # "data_replaced.csv"
)

for parent_dir in "${parent_dir_list[@]}"; do
    for routing_policy_dir in "${routing_policy_dir_list[@]}"; do
        for data_file in "${data_file_list[@]}"; do
            data_file_full_path="../training_data/${parent_dir}/${routing_policy_dir}/${data_file}"
            if [ -f "${data_file_full_path}" ]; then
                echo "Data file already exists: ${data_file_full_path}. exiting..."
                exit 1
            fi
            cat ${data_file_full_path} >> ../training_data/${parent_dir}/all/${data_file}.csv
            echo "Appending ${data_file_full_path} to ../training_data/${parent_dir}/all/${data_file}.csv"
        done
    done
done

echo "Done"