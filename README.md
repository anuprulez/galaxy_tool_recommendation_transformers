# Tool recommender system in Galaxy using Transformers

## General information

Project name: Galaxy tool recommendation using Transformers

Project home page: <<>>

Data: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/data

Operating system(s): Linux

Programming language: Python

Scripts: https://github.com/anuprulez/galaxy_tool_recommendation/tree/master/scripts

iPython notebook: <<>>

Other requirements: <<>>

Nvidia-GPU: conda install -c anaconda tensorflow-gpu  v:v2.4.1
https://anaconda.org/anaconda/tensorflow-gpu

### GPU Conda env
conda create --name tool_prediction_rnn_4 python=3.9
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install --upgrade pip
pip install tensorflow-gpu

pip install pandas tqdm scikit-learn

-------------

Training script: https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/train.sh

License: MIT License

RRID: <<to add>>

bioToolsID: <<to add>>

## History of the work

Initial work to create tool recommendation model is stored at https://github.com/anuprulez/similar_galaxy_workflow. This repository storing the history of work until October, 2019 will not be used in future. The current repository (https://github.com/anuprulez/galaxy_tool_recommendation) will be used for current and future developments.

## Description of other branches in this repository
- Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network) (branch: [master](https://github.com/anuprulez/galaxy_tool_recommendation))
- Tool recommender system in Galaxy using deep learning (Dense neural network) (branch: [dnn_wc](https://github.com/anuprulez/galaxy_tool_recommendation/tree/dnn_wc))
- Tool recommender system in Galaxy using deep learning (Convolutional neural network) (branch: [cnn_wc](https://github.com/anuprulez/galaxy_tool_recommendation/tree/cnn_wc))
- Tool Recommender in Galaxy using stored tool sequences (branch: [statistical_model](https://github.com/anuprulez/galaxy_tool_recommendation/tree/statistical_model))
- Tool recommender system in Galaxy using ExtraTrees classifier (branch: [sklearn_rf](https://github.com/anuprulez/galaxy_tool_recommendation/tree/sklearn_rf))
- Tool recommender system in Galaxy using deep learning (Gated recurrent units neural network with no regularisation) (branch: [no_regularisation](https://github.com/anuprulez/galaxy_tool_recommendation/tree/no_regularisation))

## (To reproduce this work) How to create a sample tool recommendation model:

**Note**: To reproduce this work after training on complete model, it is required to have a decent compute resource (with at least 10 GB RAM) and it takes > 24 hrs to create a trained model on complete set of workflows (~ 18,000). However, the following steps can be used to create a sample tool recommendation model on a subset of workflows:

1. Install the dependencies by executing the following lines:
    *    `conda env create -f environment.yml`
    *    `conda activate tool_prediction_gru_wc`

2. Execute `sh train.sh` (https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/train.sh). It runs on a subset of workflows. Use file `data/worflow-connection-20-04.tsv` in the training script to train on complete set of workflows (It takes a long time to finish).

3. After successful finish (~2-3 minutes), a trained model is created at `data/<<file name>>.hdf5`.

4. Put this trained model file at `ipython_script/data/<<file name>>.hdf5` and execute the ipython notebook.

5. A model trained on all workflows is present at `ipython_script/data/tool_recommendation_model_20_05.hdf5` which can be used to predict tools using the IPython notebook `ipython_script/tool_recommendation_gru_wc.ipynb`

## Data description:

Execute data extraction script `extract_data.sh` to extract two tabular files - `tool-popularity-20-04.tsv` and `worflow-connection-20-04.tsv`. This script should be executed on a Galaxy instance's database (ideally should be executed by a Galaxy admin). There are two methods in the script one each to generate two tabular files. The first file (`tool-popularity-20-04.tsv`) contains information about the usage of tools per month. The second file (`worflow-connection-20-04.tsv`) contains workflows present as the connections of tools. Save these tabular files. These tabular files are present under `/data` folder and can be used to run deep learning training by following steps.

### Description of all parameters mentioned in the training script:

`python <main python script> -wf <path to workflow file> -tu <path to tool usage file> -om <path to the final model file> -cd <cutoff date> -pl <maximum length of tool path> -ep <number of training iterations> -oe <number of iterations to optimise hyperparamters> -me <maximum number of evaluation to optimise hyperparameters> -ts <fraction of test data> -bs <range of batch sizes> -ut <range of hidden units> -es <range of embedding sizes> -dt <range of dropout> -sd <range of spatial dropout> -rd <range of recurrent dropout> -lr <range of learning rates> -cpus <number of CPUs>`

   - `<main python script>`: This script is the entry point of the entire analysis. It is present at `scripts/main.py`.
   
   - `<path to workflow file>`: It is a path to a tabular file containing Galaxy workflows. E.g. `data/worflow-connection-20-04.tsv`.
   
   - `<path to tool popularity file>`: It is a path to a tabular file containing usage frequencies of Galaxy tools. E.g. `data/tool-popularity-20-04.tsv`.
   
   - `<path to trained model file>`: It is a path of the final trained model (`h5` file). E.g. `data/<<file name>>.hdf5`.
   
   - `<cutoff date>`: It is used to set the earliest date from which the usage frequencies of tools should be considered. The format of the date is YYYY-MM-DD. This date should be in the past. E.g. `2017-12-01`.

   - `<maximum length of tool path>`: This takes an integer and specifies the maximum size of a tool sequence extracted from any workflow. Any tool sequence of length larger than this number is not included in the dataset for training. E.g. `25`.


### (To reproduce this work on complete set of workflows) Example command:

   `python scripts/main.py -wf data/worflow-connection-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 10 -oe 5 -me 20 -ts 0.2 -bs '32,256' -ut '32,256' -es '32,256' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001,0.1' -cpus 4`

Once the script finishes, `H5` model file is created at the given location (`path to trained model file`).

## (For Galaxy admins) The following steps are only necessary for deploying on any Galaxy server:

1. (Already done!) The latest model is uploaded at: https://github.com/galaxyproject/galaxy-test-data/blob/master/tool_recommendation_model.hdf5. Change this path only if there is a different model.

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

# Readings
# https://keras.io/examples/nlp/text_classification_with_transformer/
# https://hannibunny.github.io/mlbook/transformer/attention.html
# https://arxiv.org/pdf/1905.09418.pdf
# https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15794705.pdf
# https://royalsocietypublishing.org/doi/10.1098/rsos.191517
# https://arxiv.org/pdf/1909.06639.pdf


#Train data:  (414952, 25)
#Test data:  (103739, 25)
