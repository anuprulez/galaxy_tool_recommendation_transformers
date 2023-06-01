# Tool recommender system in Galaxy using Transformers

## General information

Project name: Galaxy tool recommendation using Transformers

Project home page: https://github.com/anuprulez/galaxy_tool_recommendation_transformers

(Example) Data: https://github.com/anuprulez/galaxy_tool_recommendation_transformers/tree/master/data/test_data

Complete data: https://doi.org/10.5281/zenodo.7825973

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation_transformers/tree/master/scripts

IPython notebook for model comparison: https://github.com/anuprulez/galaxy_tool_recommendation_transformers/blob/master/notebooks/evaluate_model.ipynb

Training script: https://github.com/anuprulez/galaxy_tool_recommendation_transformers/blob/master/train.sh

License: MIT License

## (To reproduce this work) How to create a sample tool recommendation model:

**Note**: To reproduce this work after training on complete model, it is required to have a decent compute resource (with at least 10 GB RAM) and it takes > 24 hrs to create a trained model on complete set of workflows (~ 60,000). However, the following steps can be used to create a sample tool recommendation model on a subset of workflows:

1. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_transformers`

2. Execute `sh train.sh` (https://github.com/anuprulez/galaxy_tool_recommendation_transformers/blob/master/train.sh). It runs on a subset of workflows.

3. After successful finish (~2-3 minutes), a trained model is created at `log/saved_model/<<last training iteration>>/tf_model_h5/<<model.h5>>`.
4. For running on complete data: All datasets are shared at: https://doi.org/10.5281/zenodo.7825973. Download these two tabular files and add their paths in the `train.sh` file and execute.

## Plots:

## Precision@k for Transformer, RNN, CNN and DNN

![Precision@k](https://raw.githubusercontent.com/anuprulez/galaxy_tool_recommendation_transformers/master/plots/prec_k_transformer_rnn_cnn_dnn.png "Precision@k")

## Precision@k for Transformer, RNN, CNN and DNN for infrequent tools

![Precision@k](https://raw.githubusercontent.com/anuprulez/galaxy_tool_recommendation_transformers/master/plots/prec_low_prec.png "Precision@k")

Attention scores:

![Differential expression analysis workflow](https://raw.githubusercontent.com/anuprulez/galaxy_tool_recommendation_transformers/master/plots/attention_featurecounts_1_run_2_step_40000.png "Differential expression analysis workflow")

## Data description:

Execute data extraction script `extract_data.sh` to extract two tabular files - `tool_popularity_Aug_22.csv` and `wf-subset.csv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file contains information about the usage of tools per month. The second file contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data/aug_22/` folder and can be used to run deep learning training by following steps.

### Description of all parameters mentioned in the training script:

`python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final H5 model file> -cd <cutoff date to exclude old workflows> -pl <maximum length of tool path> -ti <number of training iterations> -nhd <number of attention heads> -ed <embedding dimensions> -fd <feed forward dimensions> -dt <dropout> -lr <learning rate> -ts <test data percentage> -trbs <training batch size> -tebs <test batch size> -trlg <train logging step> -telg <test logging step> -ud <use preprocessed data> --is_transformer <to use transformer or RNN> --model_type <use one of transformer, rnn, cnn or dnn> --restart_step <use step of last training>`

### (To reproduce this work on complete set of workflows) Example command:

   `python scripts/main.py -wf data/aug_22/wf-subset.csv -tu data/aug_22/tool_popularity_Aug_22.csv -om data/aug_22/tool_recommendation_model.hdf5 -cd '2017-12-31' -pl 25 -ti 200 -nhd 4 -ed 128 -fd 128 -dt 0.2 -lr 0.001 -ts 0.2 -trbs 512 -tebs 512 -trlg 10 -telg 10 -ud false --model_type transformer --restart_step 0`

## (For Galaxy admins) The following steps are only necessary for deploying on any Galaxy server:

1. (Already done!) The latest model is uploaded at: https://github.com/galaxyproject/galaxy-test-data/blob/master/tool_recommendation_model_v_0.2.hdf5. Change this path only if there is a different model.

2. In the `galaxy.yml.sample` config file, make the following changes:
    - Enable and then set the property `enable_tool_recommendations` to `true`.

3. In order to allow Galaxy admins to add/remove tools from the list of recommendations, the following steps can be used:
    - A Galaxy config file has been provided (https://github.com/galaxyproject/galaxy/blob/dev/config/tool_recommendations_overwrite.yml.sample) to offer following features and instructions to use these features are given in the file itself:
        - Enable `admin_tool_recommendations_path` in Galaxy's config file at `config/galaxy.yml.sample`.
        - Add tool(s) and mark them "deprecated".
        - Add new tool(s) to the list of recommendations.
        - Overwrite all recommendations (predicted by trained model). (Enable `overwrite_model_recommendations` and set to `true` in Galaxy's config file at `config/galaxy.yml.sample`).

## For Galaxy end-users:

Open the workflow editor and choose any tool from the toolbox. Then, hover on the `right-arrow` icon in top-right of the tool to see the recommended tools in a pop-over. Moreover, execute a tool and see recommended tools for further analysis in a tree visualisation.

## For contributors:

Information about contributors and how to contribute is present in `CONTRIBUTING.md` file.
