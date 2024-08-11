import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential

import transformer_network


warnings.filterwarnings("ignore")

font = {'family': 'serif', 'size': 18}
fig_size = (10, 10)

plt.rc('font', **font)
size_title = 20
size_label = 18

embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
dropout = 0.2
seq_len = 25
dpi = 100


#base_path = "/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/"

base_path = "../final_data/aug_22_data/"
#"/media/anupkumar/6c9b94c9-2316-4ae1-887a-5047a02bc3d7/home/kumara/tool_prediction_compute_results/backup_tool_pred_transformer_computed_results/aug_22_data/"
#"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/"
#"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/tool_prediction_datasets/computed_results/aug_22 data/"
#"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/"
#"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/tool_prediction_datasets/computed_results/aug_22 data/"
n_runs = 5


def read_file_cnn_dnn(file_path):
    with open(file_path, "r") as json_file:
        file_content = json_file.read()
    return file_content

def read_file(file_path):
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def collect_loss_prec_data(m_type):
    model_type = m_type[0]
    m_path = base_path + model_type + "/run"
    print(m_path)
    runs_indices = list()
    runs_te_loss = list()
    model_numbers = [1, 100, 200, 500, 1000, 1200, 1500, 2000, 2500, 3000, 3500]
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
    plt.savefig("plots/transformer_runs_te_loss.pdf", dpi=dpi)
    
    ## Transformer: For test precision
    fig = plt.figure(figsize=fig_size)
    transformer_runs_te_prec = list()
    transformer_runs_te_prec_low = list()
    print(m_path)
    #model_numbers = [100, 200, 500, 1000, 2000, 3000, 3500]
    for i in range(n_runs):
       x_path = "{}{}/".format(m_path, str(i+1))
       epo_te_batch_prec = read_file(x_path + "data/epo_te_precision.txt").split(",")
       epo_te_batch_prec = np.array([np.round(float(item), 4) for item in epo_te_batch_prec])
       epo_te_batch_prec = epo_te_batch_prec[model_numbers]
       transformer_runs_te_prec.extend(epo_te_batch_prec)
       
       epo_low_te_precision = read_file(x_path + "data/epo_low_te_precision.txt").split(",")
       print(i, epo_low_te_precision)
       epo_low_te_precision = np.array([np.round(float(item), 4) for item in epo_low_te_precision])
       epo_low_te_precision = epo_low_te_precision[model_numbers]
       transformer_runs_te_prec_low.extend(epo_low_te_precision)

    ## RNN: For test precision
    rnn_runs_te_prec = list()
    rnn_runs_te_prec_low = list()
    m_path = base_path + m_type[1] + "/run"
    print(m_path)
    for i in range(n_runs):

       x_path = "{}{}/".format(m_path, str(i+1))
       epo_te_batch_prec = read_file(x_path + "data/epo_te_precision.txt").split(",")
       epo_te_batch_prec = np.array([np.round(float(item), 4) for item in epo_te_batch_prec])
       epo_te_batch_prec = epo_te_batch_prec[model_numbers]
       rnn_runs_te_prec.extend(epo_te_batch_prec)

       rnn_low_te_precision = read_file(x_path + "data/epo_low_te_precision.txt").split(",")
       rnn_low_te_precision = np.array([np.round(float(item), 4) for item in rnn_low_te_precision])
       rnn_low_te_precision = rnn_low_te_precision[model_numbers]
       rnn_runs_te_prec_low.extend(rnn_low_te_precision)
    
    ## CNN: For test precision
    cnn_runs_te_prec = list()
    cnn_runs_te_prec_low = list()
    m_path = base_path + m_type[2] + "/run"
    print(m_path)
    for i in range(n_runs):

       x_path = "{}{}/".format(m_path, str(i+1))
       print(x_path)
       epo_te_batch_prec = read_file_cnn_dnn(x_path + "data/epo_te_precision.txt").split(",")
       print(epo_te_batch_prec)
       epo_te_batch_prec = np.array([np.round(float(item), 4) for item in epo_te_batch_prec])
       epo_te_batch_prec = epo_te_batch_prec[model_numbers]
       cnn_runs_te_prec.extend(epo_te_batch_prec)

       cnn_low_te_precision = read_file_cnn_dnn(x_path + "data/epo_low_te_precision.txt").split(",")
       cnn_low_te_precision = np.array([np.round(float(item), 4) for item in cnn_low_te_precision])
       cnn_low_te_precision = cnn_low_te_precision[model_numbers]
       cnn_runs_te_prec_low.extend(cnn_low_te_precision)

    
    ## DNN: For test precision
    dnn_runs_te_prec = list()
    dnn_runs_te_prec_low = list()
    m_path = base_path + m_type[3] + "/run"
    print(m_path)
    for i in range(n_runs):

       x_path = "{}{}/".format(m_path, str(i+1))
       epo_te_batch_prec = read_file_cnn_dnn(x_path + "data/epo_te_precision.txt").split(",")
       epo_te_batch_prec = np.array([np.round(float(item), 4) for item in epo_te_batch_prec])
       epo_te_batch_prec = epo_te_batch_prec[model_numbers]
       dnn_runs_te_prec.extend(epo_te_batch_prec)

       dnn_low_te_precision = read_file_cnn_dnn(x_path + "data/epo_low_te_precision.txt").split(",")
       dnn_low_te_precision = np.array([np.round(float(item), 4) for item in dnn_low_te_precision])
       dnn_low_te_precision = dnn_low_te_precision[model_numbers]
       dnn_runs_te_prec_low.extend(dnn_low_te_precision)

    # precision
    df_tr_rnn_cnn_dnn_runs_te_prec = pd.DataFrame(zip(runs_indices, transformer_runs_te_prec, rnn_runs_te_prec, cnn_runs_te_prec, dnn_runs_te_prec), columns=["indices", "tran_prec", "rnn_prec", "cnn_prec", "dnn_prec"])
    
    
    df_tr_rnn_cnn_dnn_runs_te_prec.to_csv("plots/df_tr_rnn_cnn_dnn_runs_te_prec.csv", index=None, sep="\t")

    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec, x="indices", y="tran_prec", label="Transformer: test tools", color="green", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec, x="indices", y="rnn_prec", label="RNN (GRU): test tools", color="red", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec, x="indices", y="cnn_prec", label="CNN: test tools", color="blue", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec, x="indices", y="dnn_prec", label="DNN: test tools", color="black", linestyle="-")
    
    plt.grid(True)
    plt.xlabel("Training iteration")
    plt.ylabel("Precision@k")
    plt.title("Test: precision@k")
    
    plt.savefig("plots/df_tr_rnn_cnn_dnn_runs_te_prec.pdf", dpi=dpi, bbox_inches='tight')
    plt.savefig("plots/df_tr_rnn_cnn_dnn_runs_te_prec.png", dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # with low precision
    df_tr_rnn_cnn_dnn_runs_te_prec_low_prec = pd.DataFrame(zip(runs_indices, transformer_runs_te_prec, rnn_runs_te_prec, cnn_runs_te_prec, dnn_runs_te_prec, transformer_runs_te_prec_low, rnn_runs_te_prec_low, cnn_runs_te_prec_low, dnn_runs_te_prec_low), columns=["indices", "tran_prec", "rnn_prec", "cnn_prec", "dnn_prec", "transformer_runs_te_prec_low", "rnn_runs_te_prec_low", "cnn_runs_te_prec_low", "dnn_runs_te_prec_low"])
    
    df_tr_rnn_cnn_dnn_runs_te_prec_low_prec.to_csv("plots/df_tr_rnn_cnn_dnn_runs_te_prec_low_prec.csv", index=None, sep="\t")

    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="tran_prec", label="Transformer: test tools", color="green", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="rnn_prec", label="RNN (GRU): test tools", color="red", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="cnn_prec", label="CNN: test tools", color="blue", linestyle="-")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="dnn_prec", label="DNN: test tools", color="black", linestyle="-")
    
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="transformer_runs_te_prec_low", label="Transformer: lowest 25% test tools", color="green", linestyle=":")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="rnn_runs_te_prec_low", label="RNN (GRU): lowest 25% test tools", color="red", linestyle=":")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="cnn_runs_te_prec_low", label="CNN: lowest 25% test tools", color="blue", linestyle=":")
    sns.lineplot(data=df_tr_rnn_cnn_dnn_runs_te_prec_low_prec, x="indices", y="dnn_runs_te_prec_low", label="DNN: lowest 25% test tools", color="black", linestyle=":")
    
    plt.grid(True)
    plt.xlabel("Training iteration")
    plt.ylabel("Precision@k")
    plt.title("Test: precision@k")
    
    plt.savefig("plots/df_tr_rnn_cnn_dnn_runs_te_prec_low_prec.pdf", dpi=dpi, bbox_inches='tight')
    plt.savefig("plots/df_tr_rnn_cnn_dnn_runs_te_prec_low_prec.png", dpi=dpi, bbox_inches='tight')
    plt.show()


##################### Model vs load time ###############################

def read_h5_model(run, m_type, m_num, test_data_batch=[]):
    #path_test_data = base_path + m_type + "/run" + str(run) + "/saved_data/test.h5"
    #print(path_test_data)

    #test_file = h5py.File(path_test_data, 'r')
    #test_input = np.array(test_file["input"])
    #test_target = np.array(test_file["target"])

    #te_x = [499, 2213, 1264,  922,  121, 1966, 1710, 2013,  929, 2059, 2093,  785, 78, 1999, 1460, 1486, 0, 0, 0, 0, 0, 0,    0, 0, 0]
    te_x = tf.cast(test_data_batch, dtype=tf.float32, name="input_2")
    #print(te_x)
    #embed, te_prediction, att_weights = tf_loaded_model(te_x_batch, training=False)

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
        if len(test_data_batch) > 0:
            _, _ = tf_loaded_model(te_x, training=False)
    elif m_type == "rnn":
        tf_loaded_model = create_rnn_model(seq_len, len(r_dict) + 1)
        if len(test_data_batch) > 0:
            _ = tf_loaded_model(te_x, training=False)
    elif m_type == "cnn":
        print("reading cnn model")
        tf_loaded_model = create_cnn_model(seq_len, len(r_dict) + 1)
        if len(test_data_batch) > 0:
            _ = tf_loaded_model(te_x, training=False)
    elif m_type == "dnn":
        tf_loaded_model = create_dnn_model(seq_len, len(r_dict) + 1)
        if len(test_data_batch) > 0:
            _ = tf_loaded_model(te_x, training=False)
        
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


def create_dnn_model(maxlen, vocab_size):

    model = Sequential()
    model.add(Embedding(vocab_size+1, embed_dim, input_length=maxlen))
    model.add(SpatialDropout1D(dropout))
    model.add(Flatten())
    model.add(Dense(embed_dim, input_shape=(seq_len,), activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(embed_dim, activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation="sigmoid"))

    return model
    
def create_cnn_model(maxlen, vocab_size): 
   
    model = Sequential()
    model.add(Embedding(vocab_size+1, ff_dim, input_length=maxlen))
    model.add(Lambda(lambda x: tf.expand_dims(x, 3)))
    model.add(Conv2D(ff_dim, kernel_size=(16, 3), activation = 'relu', kernel_initializer='he_normal', padding = 'VALID'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(ff_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='sigmoid'))
   
    return model


def plot_model_vs_load_time(model_types):
    #model_numbers = ["1000", "2000", "3000", "4000", "5000", "10000", "20000", "25000", "30000", "35000"]
    model_numbers = ["10000", "30000"]
    transformer_model_num = list()
    rnn_model_num = list()
    transformer_load_time = list()
    rnn_load_time = list()
    cnn_load_time = list()
    dnn_load_time = list()
    for m_type in model_types:
        for run in range(n_runs):
            for m_num in model_numbers:
                tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(run+1, m_type, m_num)
                print("Run: {}, model: {}, model number: {}, loading time: {} seconds".format(run+1, m_type, m_num, model_loading_time))
                print()
                if m_type == "transformer":
                   transformer_load_time.append(model_loading_time)
                elif m_type == "rnn":
                   rnn_load_time.append(model_loading_time)
                elif m_type == "cnn":
                   cnn_load_time.append(model_loading_time)
                elif m_type == "dnn":
                   dnn_load_time.append(model_loading_time)
            if m_type == "transformer":
                transformer_model_num.extend(model_numbers)
            else:
                rnn_model_num.extend(model_numbers)
            print("Model number ends")
        print("Run ends")
    print(transformer_load_time, len(transformer_load_time))
    print()
    print(rnn_load_time, len(rnn_load_time))
    print()
    print(cnn_load_time, len(cnn_load_time))
    print()
    print(dnn_load_time, len(dnn_load_time))

    df_tran_rnn_dnn_cnn_model_load_time = pd.DataFrame(zip(rnn_model_num, transformer_load_time, rnn_load_time, cnn_load_time, dnn_load_time,), columns=["model_nums", "tran_load_time", "rnn_load_time", "cnn_load_time", "dnn_load_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="tran_load_time", label="Transformer: model load time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="rnn_load_time", label="RNN (GRU): model load time", color="red", linestyle="-")
    sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="cnn_load_time", label="CNN: model load time", linestyle="-", color="blue")
    sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="dnn_load_time", label="DNN: model load time", color="black", linestyle="-")
    plt.grid(True)
    plt.xlabel("Training step")
    plt.ylabel("Model load time (seconds)")
    plt.title("Transformer, RNN (GRU), CNN and DNN models loading time")
    plt.savefig("plots/transformer_rnn_runs_model_load_time.pdf", dpi=dpi, bbox_inches='tight')
    plt.savefig("plots/transformer_rnn_runs_model_load_time.png", dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_model_vs_load_time_final_iteration(model_types):
    # ["transformer", "rnn", "cnn", "dnn"]
    #model_numbers = ["1000", "2000", "3000", "4000", "5000", "10000", "20000", "25000", "30000", "35000"]
    model_numbers = 35000
    transformer_model_num = list()
    rnn_model_num = list()
    transformer_load_time = list()
    rnn_load_time = list()
    cnn_load_time = list()
    dnn_load_time = list()
    load_times = list()
    batch_size = 128
    path_test_data = base_path + "rnn/run1/saved_data/test.h5"
    test_file = h5py.File(path_test_data, 'r')
    test_input = np.array(test_file["input"])
    print(test_input.shape)
    
    for m_type in model_types:
        for run in range(n_runs):
            tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(run+1, m_type, model_numbers, test_input[:batch_size, :]) #test_input[:batch_size, :]
            print("Run: {}, model: {}, model number: {}, loading time: {} seconds".format(run+1, m_type, model_numbers, model_loading_time))
            print()
            if m_type == "transformer":
                transformer_load_time.append(model_loading_time)
                #labels.append("Transformer")
            elif m_type == "rnn":
                rnn_load_time.append(model_loading_time)
                #labels.append("RNN")
            elif m_type == "cnn":
                cnn_load_time.append(model_loading_time)
                #labels.append("CNN")
            elif m_type == "dnn":
                dnn_load_time.append(model_loading_time)
                #labels.append("DNN")
        print("Run ends")
    print(transformer_load_time, len(transformer_load_time))
    print()
    print(rnn_load_time, len(rnn_load_time))
    print()
    print(cnn_load_time, len(cnn_load_time))
    print()
    print(dnn_load_time, len(dnn_load_time))
    l_tran = ["Transformer", "Transformer", "Transformer", "Transformer", "Transformer"]
    l_rnn = ["RNN", "RNN", "RNN", "RNN", "RNN"]
    l_cnn = ["CNN", "CNN", "CNN", "CNN", "CNN"]
    l_dnn = ["DNN", "DNN", "DNN", "DNN", "DNN"]
    
    #labels = ["Transformer", "RNN", "CNN", "DNN"]
    df_model_load_times = pd.DataFrame(zip(l_tran, l_rnn, l_cnn, l_dnn, transformer_load_time, rnn_load_time, cnn_load_time, dnn_load_time,), columns=["l_tran", "l_rnn", "l_cnn", "l_dnn", "tran_load_time", "rnn_load_time", "cnn_load_time", "dnn_load_time"])
    #model_load_times = pd.DataFrame(zip(labels, load_times), columns=["labels", "load_times"])
    #fig = plt.figure(figsize=fig_size)
    #sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="tran_load_time", label="Transformer: model load time", linestyle="-", color="green")
    #sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="rnn_load_time", label="RNN (GRU): model load time", color="red", linestyle="-")
    #sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="cnn_load_time", label="CNN: model load time", linestyle="-", color="blue")
    #sns.lineplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="model_nums", y="dnn_load_time", label="DNN: model load time", color="black", linestyle="-")

    #sns.barplot(data=df_model_load_times, x="l_tran", y="tran_load_time", label="", linestyle="-", color="green")
    #sns.barplot(data=df_model_load_times, x="l_rnn", y="rnn_load_time", label="", linestyle="-", color="red")
    #sns.barplot(data=df_model_load_times, x="l_cnn", y="cnn_load_time", label="", linestyle="-", color="blue")
    #sns.barplot(data=df_model_load_times, x="l_dnn", y="dnn_load_time", label="", linestyle="-", color="black")
    
    #sns.barplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="Transformer", y="tran_load_time", label="Transformer: model load time", linestyle="-", color="green")
    
    #plt.grid(True)
    #plt.xlabel("Model types")
    #plt.ylabel("Model load time (seconds)")
    #plt.title("Transformer, RNN (GRU), CNN and DNN models loading time")
    #plt.savefig("../plots/transformer_rnn_runs_model_load_time_final_model.pdf", dpi=dpi, bbox_inches='tight')
    #plt.savefig("../plots/transformer_rnn_runs_model_load_time_final_model.png", dpi=dpi, bbox_inches='tight')
    #plt.show()
    df_model_load_times.to_csv("../plots/transformer_rnn_runs_model_load_time_final_model_GPU.csv")

def plot_model_load_times_CPU_GPU():
    print("")
    gpu_load_times = pd.read_csv("../plots/transformer_rnn_runs_model_load_time_final_model_GPU.csv")
    #gpu_load_times["compute_type"] = ["GPU", "GPU", "GPU", "GPU", "GPU"]
    cpu_load_times = pd.read_csv("../plots/transformer_rnn_runs_model_load_time_final_model_CPU.csv")
    #cpu_load_times["compute_type"] = ["CPU", "CPU", "CPU", "CPU", "CPU"]

    sns.barplot(data=gpu_load_times, x="l_tran", y="tran_load_time", label="", color="green", errorbar="sd", capsize=.2)
    sns.barplot(data=gpu_load_times, x="l_rnn", y="rnn_load_time", label="", color="red", errorbar="sd", capsize=.2)
    sns.barplot(data=gpu_load_times, x="l_cnn", y="cnn_load_time", label="", color="blue", errorbar="sd", capsize=.2)
    sns.barplot(data=gpu_load_times, x="l_dnn", y="dnn_load_time", label="", color="black", errorbar="sd", capsize=.2)

    #sns.barplot(data=cpu_load_times, x="l_tran", y="tran_load_time", label="", linestyle="-", color="green")
    #sns.barplot(data=cpu_load_times, x="l_rnn", y="rnn_load_time", label="", linestyle="-", color="red")
    #sns.barplot(data=cpu_load_times, x="l_cnn", y="cnn_load_time", label="", linestyle="-", color="blue")
    #sns.barplot(data=cpu_load_times, x="l_dnn", y="dnn_load_time", label="", linestyle="-", color="black")


    
    #sns.barplot(data=df_tran_rnn_dnn_cnn_model_load_time, x="Transformer", y="tran_load_time", label="Transformer: model load time", linestyle="-", color="green")
    
    plt.grid(True)
    plt.xlabel("Model types")
    plt.ylabel("Time (seconds)")
    plt.title("Models vs their usage time")
    plt.savefig("../plots/transformer_rnn_runs_model_load_time_final_model_GPU.pdf", dpi=dpi, bbox_inches='tight')
    plt.savefig("../plots/transformer_rnn_runs_model_load_time_final_model_GPU.png", dpi=dpi, bbox_inches='tight')
    plt.show()


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
            
            if m_type == "transformer":
                pred_s_time = time.time()
                te_prediction, _ = tf_loaded_model(t_ip, training=False)
            else:
                pred_s_time = time.time()
                te_prediction = tf_loaded_model(t_ip, training=False)
            top_k = tf.math.top_k(te_prediction, k=k)
            pred_e_time = time.time()
            batch_pred_time.append(pred_e_time - pred_s_time)
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
                tool_prediction_time = tool_prediction_time + model_loading_time
                print()
                if m_type == "transformer":
                    transformer_pred_time.append(tool_prediction_time)
                else:
                    rnn_pred_time.append(tool_prediction_time)
            if m_type == "transformer":
                transformer_seq_lengths.extend(top_k)
            else:
                rnn_seq_lengths.extend(top_k)
            test_file.close()
        print("Model number ends")
    print("Run ends")

    print(transformer_pred_time, len(transformer_pred_time))
    print()
    print(rnn_pred_time, len(rnn_pred_time))
    
    df_tran_rnn_model_pred_time_topk = pd.DataFrame(zip(rnn_seq_lengths, transformer_pred_time, rnn_pred_time), columns=["pred_topk", "tran_pred_time", "rnn_pred_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_model_pred_time_topk, x="pred_topk", y="tran_pred_time", label="Transformer: model pred time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_model_pred_time_topk, x="pred_topk", y="rnn_pred_time", label="RNN (GRU): model pred time", color="red", linestyle="-")
    plt.grid(True)
    plt.xlabel("Prediction topk")
    plt.ylabel("Model pred time (seconds)")
    plt.title("Transformer vs RNN (GRU) model pred time")
    plt.savefig("plots/transformer_rnn_runs_model_pred_time_topk.pdf", dpi=dpi)
    plt.savefig("plots/transformer_rnn_runs_model_pred_time_topk.png", dpi=dpi)


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
                if m_type == "transformer":
                    pred_s_time = time.time()
                    te_prediction, _ = tf_loaded_model(t_ip, training=False)
                else:
                    pred_s_time = time.time()
                    te_prediction = tf_loaded_model(t_ip, training=False)
                pred_e_time = time.time()
                batch_pred_time.append(pred_e_time - pred_s_time)
    print(k, len(batch_pred_time))
    return np.mean(batch_pred_time)


def plot_usage_time_vs_seq_len():
    
    model_types = ["transformer", "rnn"]
    m_num = 35000
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
                tool_prediction_time = tool_prediction_time + model_loading_time
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
    df_tran_rnn_model_pred_time_seqlen = pd.DataFrame(zip(rnn_seq_lengths, transformer_pred_time, rnn_pred_time), columns=["seq_lengths", "tran_pred_time", "rnn_pred_time"])
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(data=df_tran_rnn_model_pred_time_seqlen, x="seq_lengths", y="tran_pred_time", label="Transformer: model pred time", linestyle="-", color="green")
    sns.lineplot(data=df_tran_rnn_model_pred_time_seqlen, x="seq_lengths", y="rnn_pred_time", label="RNN (GRU): model pred time", color="red", linestyle="-")
    plt.grid(True)
    plt.xlabel("Tool sequences length")
    plt.ylabel("Model pred time (seconds)")
    plt.title("Transformer vs RNN (GRU) model pred time")
    plt.savefig("../plots/transformer_rnn_runs_model_pred_time_seq_length.pdf", dpi=dpi)
    plt.savefig("../plots/transformer_rnn_runs_model_pred_time_seq_length.png", dpi=dpi)


def make_scatter_beyond_training():

    font = {'family': 'serif', 'size': 18}
    fig_size = (12, 6)
    #fig = plt.figure(figsize=fig_size)
    plt.rc('font', **font)

    dpi = 300
    analysis = "Single-cell"
    input_tool = "anndata_import"
    ground_truth = ["scanpy_filter", "anndata_inspect", "anndata_manipulate", "ucsc_cell_browser", "scanpy_inspect", "scanpy_filter_cells"]

    pred_transformer_gt = ground_truth
    pred_rnn_gt = ground_truth

    pred_transformer_b_training = ["scanpy_normalise_data", "scanpy_plot", "anndata_ops", "scanpy_remove_confounders", "scanpy_integrate_harmony", "scanpy_normalize", "scpred_get_feature_space", "scanpy_find_variable_genes", "scpred_predict_labels", "scpred_eigen_decompose"]
    
    pred_rnn_b_training = ["scanpy_plot", "scanpy_normalise_data", "scmap_scmap_cluster", "scmap_scmap_cell", "scanpy_filter_genes"]
    
    xlabels = ["Transformer", "RNN"]

    xtypes = ["Transformer", "RNN"]
    
    matrix = [len(pred_transformer_b_training), len(pred_rnn_b_training)]

    df_recommendations = pd.DataFrame(zip(xlabels, matrix, xtypes), columns=["xlabels", "recommendations", "model_types"])

    print(df_recommendations)

    palette = {'Transformer': "green", 'RNN': "red"}

    ax = sns.barplot(df_recommendations, x="xlabels", y="recommendations", hue="model_types", palette=palette, width=0.3, dodge=False);
    
    plt.grid(True)
    ax.set_xticks(xlabels)
    plt.xlabel("Model types")
    plt.ylabel("Number of recommended tools")
    plt.title("Anndata_import: Generalisation")
    plt.tight_layout()
    plt.savefig("../plots/transformer_rnn_beyond_workflows.pdf", dpi=dpi)
    plt.savefig("../plots/transformer_rnn_beyond_workflows.png", dpi=dpi)


############ Call methods ###########################

#collect_loss_prec_data(["transformer", "rnn", "cnn", "dnn"])
#plot_model_vs_load_time(["transformer", "rnn", "cnn", "dnn"])
#plot_model_vs_load_time_final_iteration(["transformer", "rnn", "cnn", "dnn"])
#plot_model_load_times_CPU_GPU()
#plot_usage_time_vs_topk()
#plot_usage_time_vs_seq_len()
make_scatter_beyond_training()