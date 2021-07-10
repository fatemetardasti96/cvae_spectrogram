#! /bin/bash

python main.py --input-shape "(80,80)"

python main.py --input-shape "(80,80)" --filter-size 64

python main.py --input-shape "(80,80)" --filter-size 128

python main.py --input-shape "(80,80)" --filter-size 64 --kernel-size 5

python main.py --input-shape "(80,80)" --filter-size 64 --nb-layer 4

python main.py --input-shape "(80,80)" --filter-size 200 --nb-layer 4

