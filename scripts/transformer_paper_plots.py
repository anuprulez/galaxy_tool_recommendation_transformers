import os
import time
import subprocess
import h5py
import sys
import random
import numpy as np
import json
import warnings
import operator
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer

from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

import transformer_network


warnings.filterwarnings("ignore")

font = {'family': 'serif', 'size': 18}
fig_size = (12, 12)

plt.rc('font', **font)
size_title = 28
size_label = 24

embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
dropout = 0.1
seq_len = 25


base_path = "/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/tool_prediction_datasets/computed_results/aug_22 data/sample_runs/"
n_runs = 3


def read_file(file_path):
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def collect_loss_prec_data(m_type):
    model_type = m_type[0]
    m_path = base_path + model_type + "/run"
    runs_indices = list()
    runs_te_loss = list()
    model_numbers = [1, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    fig = plt.figure(figsize=fig_size)
    ## Transformer: For test loss
    for i in range(n_runs):
       x_path = "{}{}/".format(m_path, str(i+1))
       epo_te_batch_loss = read_file(x_path + "data/epo_te_batch_loss.txt").split(",")
       epo_te_batch_loss = np.array([np.round(float(item), 4) for item in epo_te_batch_loss])
       epo_te_batch_loss = epo_te_batch_loss[model_numbers]

       run_range = np.array(model_numbers) * 10
       runs_indices.extend(run_range)
       print(run_range)
       runs_te_loss.extend(epo_te_batch_loss)

    df_runs_te_loss = pd.DataFrame(zip(runs_indices, runs_te_loss), columns=["indices", "loss"])
    sns.lineplot(data=df_runs_te_loss, x="indices", y="loss")
    plt.grid(True)
    plt.xlabel("Training iteration")
    plt.ylabel("Test loss")
    plt.title("Test: binary crossentropy loss")
    plt.savefig("plots/transformer_runs_te_loss.pdf", dpi=150)
    
    ## Transformer: For test precision
    fig = plt.figure(figsize=fig_size)
    transformer_runs_te_prec = list()
    transformer_runs_te_prec_low = list()
    #model_numbers = [100, 200, 500, 1000, 2000, 3000, 3500]
    for i in range(n_runs):
       x_path = "{}{}/".format(m_path, str(i+1))
       epo_te_batch_prec = read_file(x_path + "data/epo_te_precision.txt").split(",")
       epo_te_batch_prec = np.array([np.round(float(item), 4) for item in epo_te_batch_prec])
       epo_te_batch_prec = epo_te_batch_prec[model_numbers]
       transformer_runs_te_prec.extend(epo_te_batch_prec)
       #epo_low_te_precision = read_file(x_path + "data/epo_low_te_precision.txt").split(",")
       #epo_low_te_precision = [np.round(float(item), 4) for item in epo_low_te_precision]
       #transformer_run3500s_te_prec_low.extend(epo_low_te_precision)

    ## RNN: For test precision
    rnn_runs_te_prec = list()
    rnn_runs_te_prec_low = list()
    m_path = base_path + m_type[1] + "/run"
    for i in range(n_runs):
       x_path = "{}{}/".format(m_path, str(i+1))
       rnn_te_batch_prec = read_file(x_path + "data/epo_te_precision.txt").split(",")
       rnn_te_batch_prec = np.array([np.round(float(item), 4) for item in rnn_te_batch_prec])
       rnn_te_batch_prec = rnn_te_batch_prec[model_numbers]
       rnn_runs_te_prec.extend(rnn_te_batch_prec)

       #rnn_low_te_precision = read_file(x_path + "data/epo_low_te_precision.txt").split(",")
       #rnn_low_te_precision = [np.round(float(item), 4) for item in rnn_low_te_precision]
       #rnn_runs_te_prec_low.extend(rnn_low_te_precision)

    #df_tr_rnn_runs_te_prec = pd.DataFrame(zip(runs_indices, transformer_runs_te_prec, transformer_runs_te_prec_low, rnn_runs_te_prec, rnn_runs_te_prec_low), columns=["indices", "tran_prec", "tran_low_prec", "rnn_prec", "rnn_low_prec"])

    df_tr_rnn_runs_te_prec = pd.DataFrame(zip(runs_indices, transformer_runs_te_prec, rnn_runs_te_prec), columns=["indices", "tran_prec", "rnn_prec"])
    print(df_tr_rnn_runs_te_prec)

    print(df_tr_rnn_runs_te_prec)
    sns.lineplot(data=df_tr_rnn_runs_te_prec, x="indices", y="tran_prec", label="Transformer: test tools", linestyle="-", color="green")
    #sns.lineplot(data=df_tr_rnn_runs_te_prec, x="indices", y="tran_low_prec", label="Transformer: lowest 25% test tools", color="green", linestyle=":")
    sns.lineplot(data=df_tr_rnn_runs_te_prec, x="indices", y="rnn_prec", label="RNN (GRU): test tools", color="red", linestyle="-")
    #sns.lineplot(data=df_tr_rnn_runs_te_prec, x="indices", y="rnn_low_prec", label="RNN (GRU): lowest 25% test tools", color="red", linestyle=":")
    plt.grid(True)
    plt.xlabel("Training iteration")
    plt.ylabel("Precision@k")
    plt.title("Test: precision@k")
    plt.savefig("plots/transformer_rnn_runs_te_prec.pdf", dpi=150)


##################### Model vs load time ###############################

def read_h5_model(run, m_type, m_num):
    #path_test_data = base_path + m_type + "/run" + str(run) + "/saved_data/test.h5"
    #print(path_test_data)

    '''test_file = h5py.File(path_test_data, 'r')
    test_input = np.array(test_file["input"])
    test_target = np.array(test_file["target"])'''

    model_path_h5 = base_path + m_type + "/run" + str(run) + "/saved_model/" + str(m_num) + "/tf_model_h5/"
    print(model_path_h5)
    h5_path = model_path_h5 + "model.h5"
    model_h5 = h5py.File(h5_path, 'r')
    #seq_len = 25
    r_dict = json.loads(model_h5["reverse_dict"][()].decode("utf-8"))
    #print(r_dict)
    m_load_s_time = time.time()
    if m_type == "transformer":
        tf_loaded_model = create_transformer_model(seq_len, len(r_dict) + 1)
    else:
        tf_loaded_model = create_rnn_model(seq_len, len(r_dict) + 1)
    tf_loaded_model.load_weights(h5_path)
    m_load_e_time = time.time()
    model_loading_time = m_load_e_time - m_load_s_time

    f_dict = dict((v, k) for k, v in r_dict.items())
    c_weights = json.loads(model_h5["class_weights"][()].decode("utf-8"))
    c_tools = json.loads(model_h5["compatible_tools"][()].decode("utf-8"))
    s_conn = json.loads(model_h5["standard_connections"][()].decode("utf-8"))

    model_h5.close()
    return tf_loaded_model, f_dict, r_dict, c_weights, c_tools, s_conn, model_loading_time


def create_rnn_model(seq_len, vocab_size):

    seq_inputs = tf.keras.Input(batch_shape=(None, seq_len))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
    in_gru = tf.keras.layers.GRU(ff_dim, return_sequences=True, return_state=False)
    out_gru = tf.keras.layers.GRU(ff_dim, return_sequences=False, return_state=True)
    enc_fc = tf.keras.layers.Dense(vocab_size, activation='sigmoid', kernel_regularizer="l2")
    embed = gen_embedding(seq_inputs)
    embed = tf.keras.layers.Dropout(dropout)(embed)
    gru_output = in_gru(embed)
    gru_output = tf.keras.layers.Dropout(dropout)(gru_output)
    gru_output, hidden_state = out_gru(gru_output)
    gru_output = tf.keras.layers.Dropout(dropout)(gru_output)
    fc_output = enc_fc(gru_output)

    return Model(inputs=[seq_inputs], outputs=[fc_output])


def create_transformer_model(maxlen, vocab_size):
    inputs = Input(shape=(maxlen,))
    embedding_layer = transformer_network.TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = transformer_network.TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=[outputs, weights])


def plot_model_vs_load_time():
    model_types = ["transformer", "rnn"]
    model_numbers = ["1000", "2000", "3000", "4000", "5000", "10000", "20000", "25000", "30000", "35000"]
    input_seq_lengths = [1, 5, 10, 15, 20]
    top_k = [1, 5, 10, 15, 20]
    transformer_model_num = list()
    rnn_model_num = list()
    transformer_load_time = list()
    rnn_load_time = list()
    for m_type in model_types:
        for run in range(n_runs):
            for m_num in model_numbers:
                tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(run+1, m_type, m_num)
                print("Run: {}, model: {}, model number: {}, loading time: {} seconds".format(run+1, m_type, m_num, model_loading_time))
                print()
                if m_type == "transformer":
                   transformer_load_time.append(model_loading_time)
                else:
                   rnn_load_time.append(model_loading_time)
            if m_type == "transformer":
                transformer_model_num.extend(model_numbers)
            else:
                rnn_model_num.extend(model_numbers)
            print("Model number ends")
        print("Run ends")

    print(transformer_load_time, len(transformer_load_time))
    print()
    print(rnn_load_time, len(rnn_load_time))
    
    df_tran_rnn_model_load_time = pd.DataFrame(zip(rnn_model_num, transformer_load_time, rnn_load_time), columns=["model_nums", "tran_load_time", "rnn_load_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_model_load_time, x="model_nums", y="tran_load_time", label="Transformer: model load time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_model_load_time, x="model_nums", y="rnn_load_time", label="RNN (GRU): model load time", color="red", linestyle="-")
    plt.grid(True)
    plt.xlabel("Training step")
    plt.ylabel("Model load time (seconds)")
    plt.title("Transformer vs RNN (GRU) model loading time")
    plt.savefig("plots/transformer_rnn_runs_model_load_time.pdf", dpi=150)


def predict_tools_topk(tf_loaded_model, test_input, k, m_type):
    test_batches = 10
    batch_size = 100
    batch_pred_time = list()
    for j in range(test_batches):
        te_x_batch = test_input[j * batch_size : j * batch_size + batch_size, :]

        for i, inp in enumerate(te_x_batch):
            t_ip = te_x_batch[i]
            t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
            t_ip = tf.reshape(t_ip, (25, 1))
            pred_s_time = time.time()
            if m_type == "transformer":
                te_prediction, _ = tf_loaded_model(t_ip, training=False)
            else:
                te_prediction = tf_loaded_model(t_ip, training=False)
            top_k = tf.math.top_k(te_prediction, k=k)
            pred_e_time = time.time()
            diff_time = (pred_e_time - pred_s_time)
            batch_pred_time.append(diff_time)
    print(k, len(batch_pred_time))
    return np.mean(batch_pred_time)


def plot_usage_time_vs_topk():
    
    model_types = ["transformer", "rnn"]
    m_num = 40000
    top_k = [1, 5, 10, 15, 20]
    transformer_seq_lengths = list()
    rnn_seq_lengths = list()
    transformer_pred_time = list()
    rnn_pred_time = list()
    for m_type in model_types:
        for run in range(n_runs):
            tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(run+1, m_type, m_num)
            print("Run: {}, model: {}, model number: {}, loading time: {} seconds".format(run + 1, m_type, m_num, model_loading_time))
            path_test_data = base_path + m_type + "/run" + str(run+1) + "/saved_data/test.h5"
            print("loading test data: {}".format(path_test_data))
            test_file = h5py.File(path_test_data, 'r')
            test_input = np.array(test_file["input"])
            for tpk in top_k:
                tool_prediction_time = predict_tools_topk(tf_loaded_model, test_input, tpk, m_type)
                print("Run: {}, model: {}, model number: {}, loading time: {} seconds, tool pred time: {} seconds".format(run + 1, m_type, m_num, model_loading_time, tool_prediction_time))

                print()
                if m_type == "transformer":
                    transformer_pred_time.append(tool_prediction_time)
                else:
                    rnn_pred_time.append(tool_prediction_time)
            if m_type == "transformer":
                transformer_seq_lengths.extend(input_seq_lengths)
            else:
                rnn_seq_lengths.extend(input_seq_lengths)
            test_file.close()
        print("Model number ends")
    print("Run ends")

    print(transformer_pred_time, len(transformer_pred_time))
    print()
    print(rnn_pred_time, len(rnn_pred_time))
    m_numinput_seq_lengths
    df_tran_rnn_model_pred_time = pd.DataFrame(zip(rnn_seq_lengths, transformer_pred_time, rnn_pred_time), columns=["pred_topk", "tran_pred_time", "rnn_pred_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_model_pred_time, x="pred_topk", y="tran_pred_time", label="Transformer: model pred time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_model_pred_time, x="pred_topk", y="rnn_pred_time", label="RNN (GRU): model pred time", color="red", linestyle="-")
    plt.grid(True)
    plt.xlabel("Prediction topk")
    plt.ylabel("Model pred time (seconds)")
    plt.title("Transformer vs RNN (GRU) model pred time")
    plt.savefig("plots/transformer_rnn_runs_model_pred_time_seq_length.pdf", dpi=150)


def predict_tools_seqlen(tf_loaded_model, test_input, k, m_type):
    test_batches = 10
    batch_size = 100
    batch_pred_time = list()
    for j in range(test_batches):
        te_x_batch = test_input[j * batch_size : j * batch_size + batch_size, :]

        for i, inp in enumerate(te_x_batch):
            t_ip = te_x_batch[i]
            if len(np.where(t_ip > 0)[0]) == k:
                t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
                t_ip = tf.reshape(t_ip, (25, 1))
                pred_s_time = time.time()
                if m_type == "transformer":
                    te_prediction, _ = tf_loaded_model(t_ip, training=False)
                else:
                    te_prediction = tf_loaded_model(t_ip, training=False)
                top_k = tf.math.top_k(te_prediction, k=k)
                pred_e_time = time.time()
                diff_time = (pred_e_time - pred_s_time)
                batch_pred_time.append(diff_time)
    print(k, len(batch_pred_time))
    return np.mean(batch_pred_time)


def plot_usage_time_vs_seq_len():
    
    model_types = ["transformer", "rnn"]
    m_num = 40000
    input_seq_lengths = [1, 5, 10, 15, 20]
    transformer_seq_lengths = list()
    rnn_seq_lengths = list()
    transformer_pred_time = list()
    rnn_pred_time = list()
    for m_type in model_types:
        for run in range(n_runs):
            tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(run+1, m_type, m_num)
            print("Run: {}, model: {}, model number: {}, loading time: {} seconds".format(run + 1, m_type, m_num, model_loading_time))
            path_test_data = base_path + m_type + "/run" + str(run+1) + "/saved_data/test.h5"
            print("loading test data: {}".format(path_test_data))
            test_file = h5py.File(path_test_data, 'r')
            test_input = np.array(test_file["input"])
            for i_seq_len in input_seq_lengths:
                tool_prediction_time = predict_tools_seqlen(tf_loaded_model, test_input, i_seq_len, m_type)
                print("Run: {}, model: {}, model number: {}, loading time: {} seconds, tool pred time: {} seconds".format(run + 1, m_type, m_num, model_loading_time, tool_prediction_time))

                print()
                if m_type == "transformer":
                    transformer_pred_time.append(tool_prediction_time)
                else:
                    rnn_pred_time.append(tool_prediction_time)
            if m_type == "transformer":
                transformer_seq_lengths.extend(input_seq_lengths)
            else:
                rnn_seq_lengths.extend(input_seq_lengths)
            test_file.close()
        print("Model number ends")
    print("Run ends")

    print(transformer_pred_time, len(transformer_pred_time))
    print()
    print(rnn_pred_time, len(rnn_pred_time))
    m_num
    df_tran_rnn_model_pred_time = pd.DataFrame(zip(rnn_seq_lengths, transformer_pred_time, rnn_pred_time), columns=["seq_lengths", "tran_pred_time", "rnn_pred_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_model_pred_time, x="seq_lengths", y="tran_pred_time", label="Transformer: model pred time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_model_pred_time, x="seq_lengths", y="rnn_pred_time", label="RNN (GRU): model pred time", color="red", linestyle="-")
    plt.grid(True)
    plt.xlabel("Tool sequences length")
    plt.ylabel("Model pred time (seconds)")
    plt.title("Transformer vs RNN (GRU) model pred time")
    plt.savefig("plots/transformer_rnn_runs_model_pred_time_seq_length.pdf", dpi=150)
############ Call methods ###########################

#collect_loss_prec_data(["transformer", "rnn"])
#plot_model_vs_load_time()
plot_usage_time_vs_topk()
plot_usage_time_vs_seq_len()
 


'''

def plot_model_usage_time():
    steps = [1, 5, 10, 15, 20]

    rnn_time = [9.10, 8.229, 14.79, 10.39, 13.96]
    transformer_time = [0.92, 2.96, 0.97, 2.02, 1.68]

    rnn_time_seq_len = [11.30, 9.83, 8.85, 8.25, 8.14]
    transformer_time_seq_len = [1.112, 1.277, 1.273, 1.195, 1.175]

    x_val = np.arange(len(steps))

    plt.plot(x_val, rnn_time, "o-", color='green')
    plt.plot(x_val, rnn_time_seq_len, "^-", color='green')
    plt.plot(x_val, transformer_time, "o-", color='red')
    plt.plot(x_valylabel, transformer_time_seq_len, "^-", color='red')

    plt.plot(x_val, rnn_te_prec, "o-", color='green')
    plt.plot(x_val, rnn_te_prec_low, "^-",  color='green')
    plt.plot(x_val, transformer_te_prec, "*-", color='red')
    plt.plot(x_val, transformer_te_prec_low, "^-", color='red')
    plt.ylabel("Model usage (load + prediction) time (seconds)")

    plt.xlabel("Seq length and topK - 1, 5, 10, 15, 20")
    plt.xticks(x_val, [str(item) for item in steps])
    plt.legend(["Topk: RNN (GRU)", "Seq len: RNN (GRU)", "Topk: Transformer", "Seq len: Transformer"])
    plt.grid(True)
    plt.title("Model usage (load + prediction) time comparison between RNN (GRU) and Transformer")
    plt.savefig(base_path + "data/model_usage_time_rnn_transformer.png", dpi=300)
    plt.show()


def plot_rnn_transformer(tr_loss, te_loss):
    # plot training loss
    tr_pos_plot = [5000, 10000, 20000, 30000, 40000, 75000, 100000]
    te_pos_plot = [50, 100, 200, 300, 400, 750, 1000]

    print(len(tr_loss), len(te_loss))

    tr_loss_val = [tr_loss[item] for item in tr_pos_plot]
    te_loss_val = [te_loss[item] for item in te_pos_plot]

    print(tr_loss_val)
    print(te_loss_val)

    x_val = np.arange(len(tr_losepo_te_batch_losss_val))
    plt.plot(x_val, tr_loss_val)
    plt.plot(x_val, te_loss_val)
    plt.ylabel("Loss")
    plt.ylim((0.00, 0.07))
    plt.xlabel("Training steps")
    plt.xticks(x_val, [str(item) for item in tr_pos_plot])
    plt.legend(["Training", "Test"])
    plt.grid(True)
    plt.title("Transformer training and test loss".format())
    plt.savefig(base_path + "data/{}_loss.png".format("transformer_tr_te_loss"), dpi=150)
    plt.show()

    #tr_pos_plot = [5000, 10000, 20000, 30000, 40000, 75000, 100000]
    #rnn_te_prec = [0.22, 0.42, 0.68, 0.84, 0.89, 0.95, 0.9579]
    #transformer_te_prec = plot_rnn_transformer[0.79, 0.90, 0.93, 0.94, 0.948, 0.953, 0.950]

    # 15.09.22
    tr_pos_plot = [1000, 5000, 10000, 20000, 40000]

    rnn_te_prec = [0.3389, 0.6239, 0.8869, 0.9841, 0.9941]
    rnn_te_prec_low = [0.0161, 0.0570, 0.7262, 0.8966, 0.9045]

    transformer_te_prec = [0.6795, 0.9404, 0.9821, 0.9890, 0.9904]
    transformer_te_prec_low = [0.1819, 0.8509, 0.8757, 0.8980, 0.8966]

    # plot topk precision for RNN and Transformer
    x_val = np.arange(len(rnn_te_prec))
    plt.plot(x_val, rnn_te_prec, "o-", color='green')
    plt.plot(x_val, rnn_te_prec_low, "^-",  color='green')
    plt.plot(x_val, transformer_te_prec, "o-", color='red')
    plt.plot(x_val, transformer_te_prec_low, "^-", color='red')
    plt.ylabel("Precision@k")
    plt.xlabel("Training steps")
    plt.xticks(x_val, [str(item) for item in tr_pos_plot])
    plt.legend(["RNN (GRU)", "Lowest 25% tools: RNN (GRU)", "Transformer", "Lowest 25% tools: Transformer"])
    plt.grid(True)
    plt.title("(Test) Precision@k for RNN (GRU) and Transformer")
    plt.savefig(base_path + "data/precision_k_rnn_vs_transformer.png", dpi=300)
    plt.show()


def plot_loss_acc(loss, acc, t_value, low_acc, low_t_value):
    # plot training loss
    x_val = np.arange(len(loss))
    if t_value == "Test":
        x_val = 10 * x_val
    #x_val = 10 * x_val
    plt.plot(x_val, loss)
    plt.ylabel("Loss")
    plt.xlabel("Training steps")
    plt.grid(True)
    if predict_rnn is True:
        plt.title("{} {} loss".format("RNN (GRU):", t_value))
    else:
        plt.title("{} {} loss".format("Transformer:", t_value))
    plt.savefig(base_path + "/data/{}_loss.pdf".format(t_value), dpi=300)
    plt.savefig(base_path + "/data/{}_loss.png".format(t_value), dpi=300)
    plt.show()

    # plot driver gene precision vs epochs
    x_val_acc = np.arange(len(acc))
    if t_value == "Test":
        x_val_acc = 10 * x_val_acc
    #x_val = np.arange(n_epo)
    plt.plot(x_val_acc, acc)
    plt.plot(x_val_acc, low_acc)
    plt.ylabel("Precision@k")
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.legend([t_value, low_t_value])
    #plt.title("{} precision@k".format(t_value))
    if predict_rnn is True:
        plt.title("{} {} precision@k".format("RNN (GRU):", t_value))
    else:
        plt.title("{} {} precision@k".format("Transformer:", t_value))
    plt.savefig(base_path + "/data/{}_acc_low_acc.pdf".format(t_value), dpi=300)
    plt.savefig(base_path + "/data/{}_acc_low_acc.png".format(t_value), dpi=300)
    plt.show()


def plot_low_te_prec(prec, t_value):
    x_val_acc = np.arange(len(prec))
    x_val_acc = 10 * x_val_acc
    plt.plot(x_val_acc, prec)
    plt.ylabel("Precision@k")
    plt.xlabel("Training steps")
    plt.grid(True)
    plt.title("{} precision@k".format(t_value))
    plt.savefig(base_path + "/data/{}_low_acc.pdf".format(t_value), dpi=300)
    plt.savefig(base_path + "/data/{}_low_acc.png".format(t_value), dpi=300)
    plt.show()


def visualize_loss_acc():
    epo_tr_batch_loss = utils.read_file(base_path + "data/epo_tr_batch_loss.txt").split(",")

    #print(len(epo_tr_batch_loss))font = {'family': 'serif', 'size': 24}

    epo_tr_batch_loss = [np.round(float(item), 4) for item in epo_tr_batch_loss]

    epo_tr_batch_acc = utils.read_file(base_path + "data/epo_tr_batch_acc.txt").split(",")
    epo_tr_batch_acc = [np.round(float(item), 4) for item in epo_tr_batch_acc]

    epo_te_batch_loss = utils.read_file(base_path + "data/epo_te_batch_loss.txt").split(",")
    epo_te_batch_loss = [np.round(float(item), 4) for item in epo_te_batch_loss]

    epo_te_batch_acc = utils.read_file(base_path + "data/epo_te_precision.txt").split(",")
    epo_te_batch_acc = [np.round(float(item), 4) for item in epo_te_batch_acc]

    #plot_loss_acc(epo_tr_batch_loss, epo_tr_batch_acc, "Training")
    #plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "Test")

    epo_te_low_batch_acc = utils.read_file(base_path + "data/epo_low_te_precision.txt").split(",")
    epo_te_low_batch_acc = [np.round(float(item), 4) for item in epo_te_low_batch_acc]

    plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "Test", epo_te_low_batch_acc, "Lowest 25% tools")

    #plot_low_te_prec(epo_te_low_batch_acc, "Lowest 25% samples in test")
    plot_rnn_transformer(epo_tr_batch_loss, epo_te_batch_loss)


=========================================
base_path = 'data_20_05/'

all_approaches_path = ['dnn/', 'dnn_wc/', 'cnn/', 'cnn_wc/', 'gru/', 'gru_wc/']

titles = ['(a) Dense neural network (DNN)', '(b) DNN with weighted loss', '(c) Convolutional neural network (CNN)', '(d) CNN with weighted loss', '(e) Recurrent neural network (GRU)', '(f) GRU with weighted loss']

font = {'family': 'serif', 'size': 24}

alpha_fade = 0.1
fig_size = (12, 12)

plt.rc('font', **font)

size_title = 28
size_label = 24
runs = 10
epochs = 10

loss_ylim = (0.0, 1.0)
usage_ylim = (1.0, 5.0)
top_legend = ['Top1', 'Top2']

gs = gridspec.GridSpec(3,2)
leg_loc = 4
leg_size = 18


def read_file(path):
    with open(path) as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = list(map(float, data))
        return data


def extract_precision(precision_path):

    top1_compatible_precision = list()
    top2_compatible_precision = list()
    top3_compatible_precision = list()
    with open(precision_path) as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = data[:epochs]
        for row in data:
            row = row.split('\n')
            row = row[0].split(' ')
            row = list(map(float, row))
            top1_compatible_precision.append(row[0])
            top2_compatible_precision.append(row[1])
            top3_compatible_precision.append(row[2])
    return top1_compatible_precision, top2_compatible_precision, top3_compatible_precision


def compute_fill_between(a_list):
    y1 = list()
    y2 = list()
    a_list = np.array(a_list, dtype=float)
    n_cols = a_list.shape[1]
    for i in range(0, n_cols):
        pos = a_list[:, i]
        std = np.std(pos)
        y1.append(std)
        y2.append(std)
    return y1, y2


def plot_loss(ax, x_val1, loss_tr_y1, loss_tr_y2, x_val2, loss_te_y1, loss_te_y2, title, xlabel, ylabel, leg):
    x_val1 = x_val1[:epochs]
    x_val2 = x_val2[:epochs]
    loss_tr_y1 = loss_tr_y1[:epochs]
    loss_tr_y2 = loss_tr_y2[:epochs]
    loss_te_y1 = loss_te_y1[:epochs]
    loss_te_y2 = loss_te_y2[:epochs]
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, loss_tr_y1, loss_tr_y2, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, loss_te_y1, loss_te_y2, color = 'b', alpha = alpha_fade)
    ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(loss_ylim)
    ax.grid(True)


def assemble_loss():
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Cross-entropy loss for multiple neural network architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):            
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Loss", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Loss", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Loss", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            
        train_loss = list()
        test_loss = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            tr_loss_path = path + 'train_loss.txt'
            val_loss_path = path + 'validation_loss.txt'
            try:
                tr_loss = read_file(tr_loss_path)
                train_loss.append(tr_loss)
                te_loss = read_file(val_loss_path)
                test_loss.append(te_loss)
            except Exception:
                continue
        loss_tr_y1, loss_tr_y2 = compute_fill_between(train_loss)
        loss_te_y1, loss_te_y2 = compute_fill_between(test_loss)

        mean_tr_loss = np.mean(train_loss, axis=0)
        mean_te_loss = np.mean(test_loss, axis=0)
        plt_title = titles[idx]
        plot_loss(ax, mean_tr_loss, mean_tr_loss - loss_tr_y1, mean_tr_loss + loss_tr_y2, mean_te_loss, mean_te_loss - loss_te_y1, mean_te_loss + loss_te_y2, plt_title + "", "Training iterations (epochs)", "Mean loss", ['Training loss', 'Test (validation) loss'])
#assemble_loss()
plt.show()


def plot_usage(ax, x_val1, y1_top1, y2_top1, x_val2, y1_top2, y2_top2, x_val3, y1_top3, y2_top3, title, xlabel, ylabel, leg):
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    #ax.plot(x_pos, x_val3, 'g')
    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, y1_top1, y2_top1, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, y1_top2, y2_top2, color = 'b', alpha = alpha_fade)
    #ax.fill_between(x_pos, y1_top3, y2_top3, color = 'g', alpha = alpha_fade)
    ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(usage_ylim)
    ax.grid(True)

def assemble_usage():
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean log usage frequency for multiple neural network architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):        
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Log usage", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Log usage", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Log usage", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        usage_top1 = list()
        usage_top2 = list()
        usage_top3 = list()
        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            usage_path = path + 'usage_weights.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(usage_path)
                usage_top1.append(top1_p)
                usage_top2.append(top2_p)
                usage_top3.append(top3_p)
            except Exception:
                continue
        mean_top1_usage = np.mean(usage_top1, axis=0)
        mean_top2_usage = np.mean(usage_top2, axis=0)
        mean_top3_usage = np.mean(usage_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(usage_top1)
        y1_top2, y2_top2 = compute_fill_between(usage_top2)
        y1_top3, y2_top3 = compute_fill_between(usage_top3)
        plt_title = titles[idx]
        leg = top_legend
        plot_usage(ax, mean_top1_usage, mean_top1_usage - y1_top1, mean_top1_usage + y2_top1, mean_top2_usage, mean_top2_usage - y1_top2, mean_top2_usage + y2_top2, mean_top3_usage, mean_top3_usage - y1_top3, mean_top3_usage + y2_top3, plt_title, "Training iterations (epochs)", "Mean log usage frequency", leg)
assemble_usage()
plt.show()


def plot_accuracy(ax, x_val1, y1_top1, y2_top1, x_val2, y1_top2, y2_top2, x_val3, y1_top3, y2_top3, title, xlabel, ylabel, leg=top_legend, precision_ylim=(0.4, 1.2)):
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    #ax.plot(x_pos, x_val3, 'g')

    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, y1_top1, y2_top1, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, y1_top2, y2_top2, color = 'b', alpha = alpha_fade)
    #ax.fill_between(x_pos, y1_top3, y2_top3, color = 'g', alpha = alpha_fade)
    ax.legend(top_legend, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(precision_ylim)
    plt.grid(True)

def assemble_accuracy(sup_title):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(sup_title, size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
           fig ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'precision.txt'
    
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k")
assemble_accuracy('Mean unpublished precision@k for multiple neural network architectures')
plt.show()


def assemble_published_precision(sup_title):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(sup_tifigtle, size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'published_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k")
assemble_published_precision('Mean published precision@k for multiple neural network architectures')
plt.show()


def assemble_lowest_normal_precision():
    precision_ylim = (0.25, 1.0)
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean precision@k in lowest 25% of data for multiple architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'lowest_norm_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k", ['Top1', 'Top2', 'Top3'], precision_ylim)
#assemble_lowest_normal_precision()
plt.show()


def assemble_lowest_published_precision():
    precision_ylim = (0.0, 0.4)
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean published precision@k in lowest 25% of data for multiple architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'lowest_pub_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k", ['Top1', 'Top2', 'Top3'], precision_ylim)
#assemble_lowest_published_precision()
plt.show()

# =================== Plot bar plots for frequency for GRU WC =============================
def plot_freq(y_val, title, xlabel, ylabel, leg):
    x_pos = np.arange(len(y_val))
    plt.plot(x_pos, y_val, color='b')
    plt.title(title, size=size_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    plt.grid(True)
    plt.show()
  
def assemble_freq(title, file_name, order_tools=None):
    fig = plt.figure(figsize=fig_size)
    tool_freq_dict = dict()
    for i in range(1, runs+1):
        path = base_path + 'gru_wc' + '/run' + str(i) + '/'
        freq_path = path + file_name 
        try:
            with open(freq_path, 'r') as f:
                data = json.loads(f.read())
                for t in data:
                    if t not in tool_freq_dict:
                        tool_freq_dict[t] = list()
                    tool_freq_dict[t].append(data[t])
        except Exception as e:
            print(e)
            continue
    t_names = list()
    t_values = list()
    if not order_tools:
        for t in tool_freq_dict:
            mean_frq = np.mean(tool_freq_dict[t])
            tool_freq_dict[t] = mean_frq
            t_names.append(t)
            t_values.append(mean_frq)
        plot_freq(t_values, title, "Number of tools", "Frequency", [])
    else:
        for t in order_tools:
            mean_frq = np.mean(tool_freq_dict[t])
            t_values.append(mean_frq) 
        plot_freq(t_values, title, "Number of tools", "Frequency", [])
        
order_tools = assemble_freq("Mean frequency (before uniform sampling) of last tools in train tool sequences", 'freq_dict_names.txt')
assemble_freq("Mean frequency (after uniform sampling) of last tools in train tool sequences", 'generated_tool_frequencies.txt', order_tools)

# ================== Plot precision for low freq tools

def plot_scatter(xval, yval1, title, xlabel, ylabel):
    plt.scatter(xval, yval1, c='b')
    #plt.legend(leg, loc=leg_loc, prop={'size': leg_size})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.grid(True)
    plt.show()
    

def assemble_low_precision(file_name):
    n_calibrations = 50
    runs = 10
    run_pub_prec = np.zeros((runs, n_calibrations))
    run_norm_prec = np.zeros((runs, n_calibrations))
    run_last_t_freq = np.zeros((runs, n_calibrations))
    run_paths = np.zeros((runs, n_calibrations))
    fig = plt.figure(figsize=fig_size)
    tool_freq_dict = dict()
    for i in range(1, runs+1):
        path = base_path + 'gru_wc' + '/run' + str(i) + '/'
        low_freq_prec_path = path + file_name 
        try:
            with open(low_freq_prec_path, 'r') as f:
                data = f.read()
                split_d = data.split("\t")
                pub_prec = split_d[0]
                norm_prec = split_d[1]
                last_t_mean_freq = split_d[2]
                n_paths = split_d[3]
                
                run_pub_prec[i-1][:] = pub_prec.split(",")
                run_norm_prec[i-1][:] = norm_prec.split(",")
                run_last_t_freq[i-1][:] = last_t_mean_freq.split(",")
                run_paths[i-1][:] = n_paths.split(",")
                
        except Exception as e:
            print(e)
            continue
    mean_pub_prec = np.nanmean(run_pub_prec, axis=0)
    mean_norm_prec = np.nanmean(run_norm_prec, axis=0)
    mean_last_t_freq = np.nanmean(run_last_t_freq, axis=0)
    mean_paths = np.nanmean(run_paths, axis=0)
    
    plt_title = "Mean unpublished precision@k vs frequencies of last tools"
    plot_scatter(mean_last_t_freq, mean_norm_prec, plt_title, "Frequency of last tools in train tool sequences", "Top 1 precision for test tool sequences")
    plt_title = "Mean published precision@k vs frequencies of last tools"
    plot_scatter(mean_last_t_freq, mean_pub_prec, plt_title, "Frequency of last tools in train tool sequences", "Top 1 precision for test tool sequences")

assemble_low_precision("test_paths_low_freq_tool_perf.txt")

###########3 Bar plot for extra trees

def read_p(file_p):
    with open(file_p, 'r') as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = data[0].split(' ')
        data = list(map(float, data))
    return data

def plot_extra_trees():
    normal_path = "data_20_05/extra_trees/precision.txt"
    published_path = "data_20_05/extra_trees/published_precision.txt"

    normal_p = read_p(normal_path)
    published_p = read_p(published_path)
    
    top1_n = normal_p[0]
    top2_n = normal_p[1]
    
    top1_p = published_p[0]
    top2_p = published_p[1]
    
    print(top1_n, top2_n)
    print(top1_p, top2_p)

    fig = plt.figure()
    X = [0.0, 0.2, 0.4, 0.6]
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(0.00, [top1_n], color = 'b', width = 0.1)
    plt.bar(0.2, [top2_n], color = 'b', width = 0.1)
    plt.bar(0.4, [top1_p], color = 'r', width = 0.1)
    plt.bar(0.6, [top2_n], color = 'r', width = 0.1)

    x_ticks = ["Top-1 Unpublished", "Top-2 Unpublished", "Top-1 Published", "Top-2 Published"]

    plt.ylabel('Precision')
    plt.title('Unpublished and Published precision@k using ExtraTrees classifier')
    plt.xticks(X)
    plt.xticks(X, x_ticks)
    plt.grid(True)
    plt.show()

plot_extra_trees()'''

############## Plot data distribution

'''paths_path = 'data/rnn_custom_loss/run1/paths.txt'
all_paths = list()

with open(paths_path) as f:
    all_paths = json.loads(f.read())

path_size = dict()
for path in all_paths:
    path_split = len(path.split(","))
    try:
        path_size[path_split] += 1
    except:
        path_size[path_split] = 1

keys = sorted(list(path_size.keys()))
values = list(path_size.values())

sorted_key_values = list()
sizes = list()
for i, ky in enumerate(keys):
    if i in path_size:
        sizes.append(str(i))
        sorted_key_values.append(path_size[i])
        
def plot_path_size_distribution(x_val, title, xlabel, ylabel, xlabels):
    plt.figure(figsize=fig_size)
    x_pos = np.arange(len(x_val))
    plt.bar(range(len(x_val)), x_val, color='skyblue')
    plt.xlabel(xlabel, size=size_label)
    plt.ylabel(ylabel, size=size_label)
    plt.title(title, size=size_title)
    plt.xticks(x_pos, xlabels, size=size_label)
    plt.yticks(size=size_label)
    plt.grid(True)
    plt.show()

#plot_path_size_distribution(sorted_key_values, 'Data distribution', 'Number of tools in paths', 'Number of paths', sizes)'''

################################################################ Tool usage


'''import csv
import numpy as np
import collections

#import plotly
#import plotly.graph_objs as go
#from plotly import tools
#import plotly.io as pio
from matplotlib import pyplot as plt

def format_tool_id(tool_link):
        tool_id_split = tool_link.split( "/" )
        tool_id = tool_id_split[ -2 ] if len( tool_id_split ) > 1 else tool_link
        return tool_id

tool_usage_file = "../data/tool-popularity-19-09.tsv"
cutoff_date = '2017-12-01'
tool_usage_dict = dict()
tool_list = list()
dates = list()
with open( tool_usage_file, 'rt' ) as usage_file:
    tool_usage = csv.reader(usage_file, delimiter='\t') 
    for index, row in enumerate(tool_usage):
        if (str(row[1]) > cutoff_date) is True:
            tool_id = format_tool_id(row[0])
            tool_list.append(tool_id)
            if row[1] not in dates:
                dates.append(row[1])
            if tool_id not in tool_usage_dict:
                tool_usage_dict[tool_id] = dict()
                tool_usage_dict[tool_id][row[1]] = int(row[2])
            else:
                curr_date = row[1]
                if curr_date in tool_usage_dict[tool_id]:
                    tool_usage_dict[tool_id][curr_date] += int(row[2])
                else:
                    tool_usage_dict[tool_id][curr_date] = int(row[2])
unique_dates = list(set(dates))
for tool in tool_usage_dict:
    usage = tool_usage_dict[tool]
    dts = usage.keys()
    dates_not_present = list(set(unique_dates) ^ set(dts))
    for dt in dates_not_present:
        tool_usage_dict[tool][dt] = 0
    tool_usage_dict[tool] = collections.OrderedDict(sorted(usage.items()))
tool_list = list(set(tool_list))

colors = ['r', 'b', 'g', 'c']
tool_names = ['Cut1', 'cufflinks', 'bowtie2', 'DatamashOps']
legends_tools = ['Tool B', 'Tool C', 'Tool D', 'Tool E']
xticks = ['Jan, 2018', '', 'Mar, 2018', '', 'May, 2018', '', 'Jul, 2018', '', 'Sep, 2018', '', 'Nov, 2018', '', 'Jan, 2019', '', 'Mar, 2019', '', 'May, 2019', '', 'Jul, 2019', '', 'Sep, 2019' ]

def plot_tool_usage(tool_names):
    plt.figure(figsize=(12, 12))
    for index, tool_name in enumerate(tool_names):
        y_val = []
        x_val = []
        tool_data = tool_usage_dict[tool_name]
        for x, y in tool_data.items():
            x_val.append(x)
            y_val.append(y)
        y_reshaped = np.reshape(y_val, (len(x_val), 1))
        plt.plot(y_reshaped[:len(y_reshaped) -1], colors[index])

    plt.legend(legends_tools)
    plt.xlabel('Months', size=size_label)
    plt.ylabel('Usage frequency', size=size_label)
    x_val = x_val[:len(x_val) - 1]
    plt.title("Usage frequency of tools over months")
    plt.xticks(range(len(xticks)), xticks, size=size_label, rotation='30')
    plt.grid(True)
    plt.show()


plot_tool_usage(tool_names)
'''
