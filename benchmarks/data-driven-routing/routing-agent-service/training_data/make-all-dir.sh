#!/bin/bash

set -e

parent_dir="p4096_s1024_rps20"
# parent_dir=$1

if [ -z "${parent_dir}" ]; then
    echo "Usage: ${0} <parent_dir>"
    exit 1
fi

if [ ! -d "${parent_dir}" ]; then
    echo "Parent directory ${parent_dir} does not exist."
    exit 1
fi

if [ ! -d "${parent_dir}/all" ]; then
    mkdir ${parent_dir}/all
    echo "Created all directory ${parent_dir}/all"
fi

touch ${parent_dir}/all/data_replaced.csv
cat ${parent_dir}/rl/data_replaced.csv >> ${parent_dir}/all/data_replaced.csv
cat ${parent_dir}/prefix/data_replaced.csv >> ${parent_dir}/all/data_replaced.csv
cat ${parent_dir}/random/data_replaced.csv >> ${parent_dir}/all/data_replaced.csv

echo "Combined all data into ${parent_dir}/all/data_replaced.csv"