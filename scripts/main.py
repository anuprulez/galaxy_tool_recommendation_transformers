"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import argparse
import os
import sys
import time


# comment this if running on GPU
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.getcwd())

from scripts import extract_workflow_connections
from scripts import prepare_data
from scripts import utils
import transformer_encoder
import create_rnn


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-wf", "--workflow_file", required=True, help="workflows tabular file")
    arg_parser.add_argument("-tu", "--tool_usage_file", required=True, help="tool usage file")
    arg_parser.add_argument("-om", "--output_model", required=True, help="trained model file")
    # data parameters
    arg_parser.add_argument("-cd", "--cutoff_date", required=True, help="earliest date for taking tool usage")
    arg_parser.add_argument("-pl", "--maximum_path_length", required=True, help="maximum length of tool path")
    arg_parser.add_argument("-ep", "--n_epochs", required=True, help="number of iterations to run to create model")
    arg_parser.add_argument("-oe", "--optimize_n_epochs", required=True, help="number of iterations to run to find best model parameters")
    arg_parser.add_argument("-me", "--max_evals", required=True, help="maximum number of configuration evaluations")
    arg_parser.add_argument("-ts", "--test_share", required=True, help="share of data to be used for testing")
    # neural network parameters
    arg_parser.add_argument("-bs", "--batch_size", required=True, help="size of the tranining batch i.e. the number of samples per batch")
    arg_parser.add_argument("-ut", "--units", required=True, help="number of hidden recurrent units")
    arg_parser.add_argument("-es", "--embedding_size", required=True, help="size of the fixed vector learned for each tool")
    arg_parser.add_argument("-dt", "--dropout", required=True, help="percentage of neurons to be dropped")
    arg_parser.add_argument("-sd", "--spatial_dropout", required=True, help="1d dropout used for embedding layer")
    arg_parser.add_argument("-rd", "--recurrent_dropout", required=True, help="dropout for the recurrent layers")
    arg_parser.add_argument("-lr", "--learning_rate", required=True, help="learning rate")
    arg_parser.add_argument("-ud", "--use_data", required=True, help="Use preprocessed data")
    arg_parser.add_argument("-cpus", "--num_cpus", required=True, help="number of cpus for parallelism")

    # get argument values
    args = vars(arg_parser.parse_args())
    tool_usage_path = args["tool_usage_file"]
    workflows_path = args["workflow_file"]
    cutoff_date = args["cutoff_date"]
    maximum_path_length = int(args["maximum_path_length"])
    trained_model_path = args["output_model"]
    n_epochs = int(args["n_epochs"])
    optimize_n_epochs = int(args["optimize_n_epochs"])
    max_evals = int(args["max_evals"])
    test_share = float(args["test_share"])
    batch_size = args["batch_size"]
    units = args["units"]
    embedding_size = args["embedding_size"]
    dropout = args["dropout"]
    spatial_dropout = args["spatial_dropout"]
    recurrent_dropout = args["recurrent_dropout"]
    learning_rate = args["learning_rate"]
    use_data = args["use_data"]
    num_cpus = int(args["num_cpus"])

    config = {
        'cutoff_date': cutoff_date,
        'maximum_path_length': maximum_path_length,
        'n_epochs': n_epochs,
        'optimize_n_epochs': optimize_n_epochs,
        'max_evals': max_evals,
        'test_share': test_share,
        'batch_size': batch_size,
        'units': units,
        'embedding_size': embedding_size,
        'dropout': dropout,
        'spatial_dropout': spatial_dropout,
        'recurrent_dropout': recurrent_dropout,
        'learning_rate': learning_rate
    }

    if use_data == "true":
        print("Loading preprocessed datasets...")
        base_path = "log/"
        train_data, train_labels = utils.read_train_test(base_path + "saved_data/train.h5")
        test_data, test_labels = utils.read_train_test(base_path + "saved_data/test.h5")
        r_dict = utils.read_file(base_path + "data/rev_dict.txt")
        f_dict = utils.read_file(base_path + "data/f_dict.txt")
        c_wts = utils.read_file(base_path + "data/class_weights.txt")
        tr_tool_freq = utils.read_file(base_path + "data/train_tool_freq.txt")
        print("True size: ", train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
        print(len(r_dict), len(f_dict))

        print("Extracted size: ", train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
        # transformer_encoder.create_enc_transformer(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_tool_freq)
        create_rnn.create_rnn_architecture(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_tool_freq)

    else:
        print("Preprocessing workflows...")
        # Extract and process workflows
        connections = extract_workflow_connections.ExtractWorkflowConnections()
        # Process raw workflow file
        wf_dataframe, usage_df = connections.process_raw_files(workflows_path, tool_usage_path, config)
        workflow_paths, standard_connections = connections.read_tabular_file(wf_dataframe, config)

        # Process the paths from workflows
        print("Dividing data...")
        data = prepare_data.PrepareData(maximum_path_length, test_share)
        train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_tool_freq = data.get_data_labels_matrices(workflow_paths, usage_df, cutoff_date, standard_connections)

        print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
        # transformer_encoder.create_enc_transformer(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_tool_freq)
        create_rnn.create_rnn_architecture(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, tr_tool_freq)

    end_time = time.time()
    print()
    print("Program finished in %s seconds" % str(end_time - start_time))
