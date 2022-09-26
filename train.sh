#!/bin/bash

python scripts/main.py -wf data/aug_22/workflow-connections_Aug_22.csv -tu data/aug_22/tool_popularity_Aug_22.csv -om data/aug_22/tool_recommendation_model.hdf5 -cd '2021-12-31' -pl 25 -ti 20 -nhd 4 -ed 128 -fd 128 -dt 0.5 -lr 0.001 -ts 0.2 -trbs 512 -tebs 512 -trlg 10 -telg 5 -ud true --is_transformer true
