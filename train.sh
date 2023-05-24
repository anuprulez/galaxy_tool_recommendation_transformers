#!/bin/bash

python scripts/main.py -wf data/aug_22/wf-subset.csv -tu data/aug_22/tool_popularity_Aug_22.csv -om data/aug_22/tool_recommendation_model.hdf5 -cd '2017-12-31' -pl 25 -ti 30 -nhd 4 -ed 128 -fd 128 -dt 0.5 -lr 0.001 -ts 0.2 -trbs 512 -tebs 512 -trlg 10 -telg 10 -ud true --model_type rnn --restart_step 0
