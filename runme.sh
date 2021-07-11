#! /bin/bash

python main.py --input-shape "(80,80)"

python main.py --input-shape "(80,80)" --nb-layers 4

python main.py --input-shape "(80,80)" --filter-size 8

python main.py --input-shape "(80,80)" --filter-size 64 --learning-rate 0.01

