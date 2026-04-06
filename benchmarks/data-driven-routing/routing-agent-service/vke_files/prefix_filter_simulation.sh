#!/bin/bash

experiment_dir=$1

python prefix_filter_simulator_v2.py ${experiment_dir} --algorithms "static_K2,adaptive_dom100,adaptive_dom300,adaptive_dom500,adaptive_dom1000"