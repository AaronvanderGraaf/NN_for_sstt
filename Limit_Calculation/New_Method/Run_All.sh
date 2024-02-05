#!/bin/bash

#conda activate root-env
#
#python Convert_Root_to_Csv.py data/NLLscan_cuu_cqu1_curve.root
#python Convert_Root_to_Csv.py data/NLLscan_cuu_cqu8_curve.root
#python Convert_Root_to_Csv.py data/NLLscan_cqu1_cqu8_curve.root
#
#conda deactivate
#python NLL_Scans_to_CLs.py data_fixed_sys/NLLscan_cuu.yaml
#python NLL_Scans_to_CLs.py data_fixed_sys/NLLscan_cqu1.yaml
#python NLL_Scans_to_CLs.py data_fixed_sys/NLLscan_cqu8.yaml

python NLL_2D_Scans_to_CLs.py data/data_cuu_cqu1.csv
python NLL_2D_Scans_to_CLs.py data/data_cuu_cqu8.csv
python NLL_2D_Scans_to_CLs.py data/data_cqu1_cqu8.csv

#rm data/*.csv