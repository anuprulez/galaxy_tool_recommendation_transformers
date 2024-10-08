"""
Create transformers
"""
import os
import time
import numpy as np
import subprocess
import h5py
import matplotlib.pyplot as plt
import sys
import random
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Lambda, GlobalMaxPooling1D
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure


#import onnx
#from onnx_tf.backend import prepare

#import predict_sequences

import utils
import transformer_network


fig_size = (15, 15)
font = {'family': 'serif', 'size': 8}
plt.rc('font', **font)

batch_size = 1000
test_batches = 0
n_topk = 1
max_seq_len = 25

embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
dropout = 0.2
seq_len = 25

model_type = "transformer" # ["transformer, "rnn", "dnn", "cnn"]

if model_type == "rnn":
    base_path = "log_rnn/"
    #"log_19_09_22_GPU_RNN_full_data/"
    #"log_19_09_22_GPU_RNN_full_data/"
    #"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/tool_prediction_datasets/computed_results/aug_22 data/rnn/run2/" #"log_19_09_22_GPU_RNN_full_data/" #"log_22_08_22_rnn/" #"log_08_08_22_rnn/"
elif model_type == "cnn":
    base_path = "/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/cnn_full_data/" #"log_cnn/"
    
elif model_type == "transformer":
    base_path = "/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/transformer/run2/"
    #"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/log_19_09_22_GPU_transformer_full_data/"
    #"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/transformer/run1/"
    
elif model_type == "dnn":
    base_path = "/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/backup_tool_pred_transformer_computed_results/aug_22_data/dnn_full_data/"
    
#"/media/anupkumar/b1ea0d39-97af-4ba5-983f-cd3ff76cf7a6/tool_prediction_datasets/computed_results/aug_22 data/transformer/run2/"
#"log_19_09_22_GPU_transformer_full_data/" #"log_12_09_22_GPU/" #"log_19_09_22_GPU_transformer_full_data/" 

#"log_22_08_22_no_att_mask_no_regu/" #"log_22_08_22_att_mask_regu/"
# log_12_09_22_GPU 

# "log_22_08_22_rnn/"
#"log_local_16_08_22_0/"

#"log_08_08_22_2/"  log_12_08_22_2 log_local_11_08_22_1 log_local_11_08_22_2 log_local_11_08_22_3

#predict_rnn = True # set to True for RNN model

#base_path = "log_08_08_22_2/"
#predict_rnn = False # set to True for RNN model

# log_08_08_22_2 (finish time: 40,000 steps in 158683 seconds)
# log_08_08_22_rnn (finish time: 40,000 steps in 193863 seconds)

## Transformer
## GPU: 40,000 steps, batch size: 512 - 132505.20160245895 seconds
## CPU: 40,000 steps, batch  size: 512 - 158683 seconds

## New CPU: Saving model at training step 400/400
## 400 steps: Program finished in 1742.0802121162415 seconds
## 4000 steps: Program finished in 15239.454410076141 seconds

## RNN
## GPU: 40,000 steps, batch size: 512 - 129000 seconds
## CPU: 40,000 steps, batch size: 512 - 193863 seconds

## New CPU: 400 Saving model at training step 400/400
### 400 steps Program finished in 2656.933450937271 seconds
## 4000 steps: Program finished in 17602.801206350327 seconds



#"log_03_08_22_1/" Balanced data with really selection of low freq tools - random choice
# RNN: log_01_08_22_3_rnn
# Transformer: log_01_08_22_0

#CNN 1

#Saving model at training step 40000/40000
#WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op while saving (showing 1 of 1). These functions will not be directly callable after loading.

#CPU: Program finished in 211880.3916196823 seconds

### CNN 2

# CPU: Program finished in 206890.36703562737 seconds
# 400 steps: Program finished in 1914.13667678833 seconds

### CNN Full data

# CPU: Program finished in 230368.01805138588 seconds

#tr_pos_plot = [1000, 5000, 10000, 20000, 30000, 40000]

# DNN
# Saving model at training step 40000/40000

# CPU: Program finished in 177232.39850640297 seconds
# CPU 400 steps: Program finished in 1750.70085811615 seconds

## DNN 2

# Program finished in 181519.5743496418 seconds

### DNN full

# Saving model at training step 40000/40000

# CPU: Program finished in 209876.5223982334 seconds



model_number = 40000
model_path = base_path + "saved_model/" + str(model_number) + "/tf_model/"
model_path_h5 = base_path + "saved_model/" + str(model_number) + "/tf_model_h5/"

'''
 ['dropletutils_read_10x', 'scmap_preprocess_sce']
['msnbase-read-msms', 'map-msms2camera', 'msms2metfrag-multiple', 'metfrag-cli-batch-multiple', 'passatutto']
'''

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
    #a_mask = Input(shape=(maxlen, maxlen))
    embedding_layer = transformer_network.TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = transformer_network.TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=[x, outputs, weights])
    
    
def create_transformer_model_last_layer(maxlen, vocab_size):
    inputs = Input(shape=(maxlen,))
    #a_mask = Input(shape=(maxlen, maxlen))
    embedding_layer = transformer_network.TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = transformer_network.TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    embed = Dense(ff_dim, activation="relu")(x)
    embed = Dropout(dropout)(embed)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=[embed, weights])
    model.summary()
    return model


def verify_training_sampling(sampled_tool_ids, rev_dict):
    """
    Compute the frequency of tool sequences after oversampling
    """
    freq_dict = dict()
    freq_dict_names = dict()
    sampled_tool_ids = sampled_tool_ids.split(",")
    for tr_tool_id in sampled_tool_ids:
        if tr_tool_id not in freq_dict:
            freq_dict[tr_tool_id] = 0
            freq_dict_names[rev_dict[str(tr_tool_id)]] = 0
        freq_dict[tr_tool_id] += 1
        freq_dict_names[rev_dict[str(tr_tool_id)]] += 1
    #print(dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)))
    s_freq = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))
    print(s_freq, len(s_freq))
    return s_freq


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
    plt.plot(x_val, transformer_time_seq_len, "^-", color='red')

    ''' 
    plt.plot(x_val, rnn_te_prec, "o-", color='green')
    plt.plot(x_val, rnn_te_prec_low, "^-",  color='green')
    plt.plot(x_val, transformer_te_prec, "*-", color='red')
    plt.plot(x_val, transformer_te_prec_low, "^-", color='red')
    '''
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
    '''tr_pos_plot = [5000, 10000, 20000, 30000, 40000, 75000, 100000]
    te_pos_plot = [50, 100, 200, 300, 400, 750, 1000]

    print(len(tr_loss), len(te_loss))

    tr_loss_val = [tr_loss[item] for item in tr_pos_plot]
    te_loss_val = [te_loss[item] for item in te_pos_plot]

    print(tr_loss_val)
    print(te_loss_val)

    x_val = np.arange(len(tr_loss_val))
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
    plt.show()'''

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
    if model_type == "rnn":
        plt.title("{} {} loss".format("RNN (GRU):", t_value))
    elif model_type == "transformer":
        plt.title("{} {} loss".format("Transformer:", t_value))
    elif model_type == "cnn":
        plt.title("{} {} loss".format("CNN:", t_value))
    
    
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
    if model_type == "rnn":
        plt.title("{} {} precision@k".format("RNN (GRU):", t_value))
    elif model_type == "transformer":
        plt.title("{} {} precision@k".format("Transformer:", t_value))
    elif model_type == "cnn":
        plt.title("{} {} precision@k".format("CNN:", t_value))
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
    epo_tr_batch_loss = utils.read_saved_file(base_path + "data/epo_tr_batch_loss.txt").split(",")

    #print(len(epo_tr_batch_loss))
    epo_tr_batch_loss = [np.round(float(item), 4) for item in epo_tr_batch_loss]

    epo_tr_batch_acc = utils.read_saved_file(base_path + "data/epo_tr_batch_acc.txt").split(",")
    epo_tr_batch_acc = [np.round(float(item), 4) for item in epo_tr_batch_acc]

    epo_te_batch_loss = utils.read_saved_file(base_path + "data/epo_te_batch_loss.txt").split(",")
    epo_te_batch_loss = [np.round(float(item), 4) for item in epo_te_batch_loss]

    epo_te_batch_acc = utils.read_saved_file(base_path + "data/epo_te_precision.txt").split(",")
    epo_te_batch_acc = [np.round(float(item), 4) for item in epo_te_batch_acc]

    #plot_loss_acc(epo_tr_batch_loss, epo_tr_batch_acc, "Training")
    #plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "Test")

    epo_te_low_batch_acc = utils.read_saved_file(base_path + "data/epo_low_te_precision.txt").split(",")
    epo_te_low_batch_acc = [np.round(float(item), 4) for item in epo_te_low_batch_acc]

    plot_loss_acc(epo_te_batch_loss, epo_te_batch_acc, "Test", epo_te_low_batch_acc, "Lowest 25% tools")

    #plot_low_te_prec(epo_te_low_batch_acc, "Lowest 25% samples in test")
    #plot_rnn_transformer(epo_tr_batch_loss, epo_te_batch_loss)


def sample_balanced(x_seqs, y_labels, ulabels_tr_dict):
    batch_tools = list(ulabels_tr_dict.keys())
    random.shuffle(batch_tools)
    last_tools = batch_tools[:batch_size]
    rand_batch_indices = list()
    for l_tool in last_tools:
        seq_indices = ulabels_tr_depo_tr_batch_lossict[l_tool]
        random.shuffle(seq_indices)
        rand_batch_indices.append(seq_indices[0])

    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]
    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y


def get_u_labels(y_train):
    last_tools = list()
    ulabels_dict = dict()
    for item in range(y_train.shape[0]):
        arr_seq = y_train[item]
        #print(arr_seq)
        label_pos = np.where(arr_seq > 0)[0]
        #print(label_pos, arr_seq)
        last_tool = str(int(arr_seq[label_pos[-1]]))
        if last_tool not in ulabels_dict:
            ulabels_dict[last_tool] = list()
        ulabels_dict[last_tool].append(item)
        seq = ",".join([str(int(a)) for a in arr_seq[0:label_pos[-1] + 1]])
        #print(seq, last_tool)
        last_tools.append(last_tool)
        #print()
    u_labels = list(set(last_tools))
    #print(len(last_tools), len(u_labels))
    random.shuffle(u_labels)
    return u_labels, ulabels_dict


def get_u_tr_labels(y_tr):
    labels = list()
    labels_pos_dict = dict()
    for i, item in enumerate(y_tr):
        label_pos = np.where(item > 0)[0]
        labels.extend(label_pos)
        for label in label_pos:
            if label not in labels_pos_dict:
                labels_pos_dict[label] = list()
            labels_pos_dict[label].append(i)

    u_labels = list(set(labels))
    
    for item in labels_pos_dict:
        labels_pos_dict[item] = list(set(labels_pos_dict[item]))
    return u_labels, labels_pos_dict


def sample_balanced_tr_y(x_seqs, y_labels, ulabels_tr_y_dict):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()

    for l_tool in batch_y_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        
        if seq_indices[0] not in rand_batch_indices:
            rand_batch_indices.append(seq_indices[0])
            label_tools.append(l_tool)
        if len(rand_batch_indices) == batch_size:
            break
    
    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, label_tools, rand_batch_indices


def verify_tool_in_tr(r_dict):
    all_sel_tool_ids = utils.read_file(base_path + "data/all_sel_tool_ids.txt").split(",")

    freq_dict = dict()
    freq_dict_names = dict()

    for tool_id in all_sel_tool_ids:
        if tool_id not in freq_dict:
            freq_dict[tool_id] = 0

        if tool_id not in freq_dict_names:
            freq_dict_names[r_dict[str(int(tool_id))]] = 0

        freq_dict[tool_id] += 1
        freq_dict_names[r_dict[str(int(tool_id))]] += 1

    s_freq = dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True))
    s_freq_names = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))

    utils.write_file(base_path + "data/s_freq_names.txt", s_freq_names)
    utils.write_file(base_path + "data/s_freq.txt", s_freq)

    return s_freq
    
    
def create_cnn_model(seq_len, vocab_size):
    
    model = Sequential()
    model.add(Embedding(vocab_size+1, embed_dim, input_length=seq_len))
    model.add(Lambda(lambda x: tf.expand_dims(x, 3)))
    model.add(Conv2D(embed_dim, kernel_size=(16, 3), activation = 'relu', kernel_initializer='he_normal', padding = 'VALID'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(embed_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='sigmoid'))

    return model
    
    
def create_dnn_model(seq_len, vocab_size):
    
    model = Sequential()
    model.add(Embedding(vocab_size+1, embed_dim, input_length=seq_len))
    model.add(SpatialDropout1D(dropout))
    model.add(Flatten())
    model.add(Dense(embed_dim, input_shape=(seq_len,), activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(embed_dim, activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation="sigmoid"))

    return model


def read_h5_model(model_type):
    print(model_path_h5)
    h5_path = model_path_h5 + "model.h5"
    model_h5 = h5py.File(h5_path, 'r')

    r_dict = json.loads(model_h5["reverse_dict"][()].decode("utf-8"))
    #print(r_dict)
    m_load_s_time = time.time()

    if model_type == "transformer":
        tf_loaded_model = create_transformer_model(seq_len, len(r_dict) + 1)
    elif model_type == "cnn":
        tf_loaded_model = create_cnn_model(seq_len, len(r_dict) + 1)
    elif model_type == "dnn":
        tf_loaded_model = create_dnn_model(seq_len, len(r_dict) + 1)   

    tf_loaded_model.summary()
    
    tf_loaded_model.load_weights(h5_path)
    
    m_load_e_time = time.time()
    model_loading_time = m_load_e_time - m_load_s_time

    f_dict = dict((v, k) for k, v in r_dict.items())
    c_weights = json.loads(model_h5["class_weights"][()].decode("utf-8"))
    c_tools = json.loads(model_h5["compatible_tools"][()].decode("utf-8"))
    s_conn = json.loads(model_h5["standard_connections"][()].decode("utf-8"))

    model_h5.close()

    return tf_loaded_model, f_dict, r_dict, c_weights, c_tools, s_conn, model_loading_time


def read_model():
    print(model_path)
    m_load_s_time = time.time()
    tf_loaded_model = tf.saved_model.load(model_path)
    m_load_e_time = time.time()
    m_l_time = m_load_e_time - m_load_s_time
    r_dict = utils.read_saved_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_saved_file(base_path + "data/f_dict.txt")
    c_weights = utils.read_file(base_path + "data/class_weights.txt")
    print(read_saved_file)
    c_tools = utils.read_saved_file(base_path + "data/compatible_tools.txt")
    s_conn = utils.read_saved_file(base_path + "data/published_connections.txt")

    return tf_loaded_model, f_dict, r_dict, c_weights, c_tools, s_conn, m_l_time
    
    
def plot_TSNE(embed, labels):
    print("Plotting embedding...")
    print(labels)

    #perplexity = 500.14460978835978835
    n_colors = 10
    figsize = (8, 8)

    figure(figsize=figsize, dpi=150)

    z = TSNE(n_components=2).fit_transform(embed)

    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=labels, data=df).set(title="T-SNE projection") #palette=sns.color_palette("hls", n_colors)
    plt.show()
    

def predict_seq():

    #visualize_loss_acc()  

    #sys.exit()

    #plot_model_usage_time()

    #tool_tr_freq = utils.read_file(base_path + "data/all_sel_tool_ids.txt")
    #verify_training_sampling(tool_tr_freq, r_dict)  

    '''path_test_data = base_path + "saved_data/test.h5"
    
    print(path_test_data)

    file_obj = h5py.File(path_test_data, 'r')

    #test_target = tf.convert_to_tensor(np.array(file_obj["target"]), dtype=tf.int64)
    test_input = np.array(file_obj["input"])
    test_target = np.array(file_obj["target"])

    print(test_input.shape, test_target.shape)'''

    if model_type == "rnn":
        print(model_path)
        m_load_s_time = time.time()
        tf_loaded_model = tf.saved_model.load(model_path)
        m_load_e_time = time.time()
        model_loading_time = m_load_e_time - m_load_s_time
        r_dict = utils.read_file(base_path + "data/rev_dict.txt")
        f_dict = utils.read_file(base_path + "data/f_dict.txt")
        class_weights = utils.read_file(base_path + "data/class_weights.txt")
        compatible_tools = utils.read_saved_file(base_path + "data/compatible_tools.txt")
        published_connections = utils.read_saved_file(base_path + "data/published_connections.txt")
    else: 
        #tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_model()
        tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model(model_type)

    '''all_tr_label_tools = verify_tool_in_tr(r_dict)
    all_tr_label_tools_ids = list(all_tr_label_tools.keys())
    all_tr_label_tools_ids = [int(t) for t in all_tr_label_tools_ids]'''

    c_weights = list(class_weights.values())

    c_weights = tf.convert_to_tensor(c_weights, dtype=tf.float32)

    #u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_target)

    precision = list()
    pub_prec_list = list()
    error_label_tools = list()
    #seq_len = 5
    #test_batches = 10000
    batch_pred_time = list()
    for j in range(test_batches):

        #te_x_batch, y_train_batch, selected_label_tools, bat_ind = sample_balanced_tr_y(test_input, test_target, u_te_y_labels_dict)

        print(j * batch_size, j * batch_size + batch_size)
        te_x_batch = test_input[j * batch_size : j * batch_size + batch_size, :]
        y_train_batch = test_target[j * batch_size : j * batch_size + batch_size, :]

        #te_x_batch = tf.convert_to_tensor(te_x_batch, dtype=tf.int64)
        #te_x_mask = utils.create_padding_mask(te_x_batch)
        #te_x_batch = tf.cast(te_x_batch, dtype=tf.int64, name="input_2")
        te_x_batch = tf.cast(te_x_batch, dtype=tf.float32, name="input_2")
        
        #print(te_x_batch, te_x_mask.shape)
        #model([x_train, att_mask], training=True)
        pred_s_time = time.time()
        
        if model_type in ["cnn", "rnn", "dnn"]:
            te_prediction = tf_loaded_model(te_x_batch, training=False)
        else:
            #te_x_mask = tf.cast(te_x_mask, dtype=tf.float32)
            #te_prediction, att_weights = tf_loaded_model([te_x_batch, te_x_mask], training=False)
            embed, te_prediction, att_weights = tf_loaded_model(te_x_batch, training=False)
            #plot_TSNE(embed)
            print(embed.shape, te_prediction.shape, att_weights.shape)
           
        pred_e_time = time.time()
        diff_time = (pred_e_time - pred_s_time) / float(batch_size)
        batch_pred_time.append(diff_time)
        filter_embed = list()
        filter_embed_label = list()
        filter_embed_label_names = list()
        for i, (inp, tar) in enumerate(zip(te_x_batch, y_train_batch)):

            t_ip = te_x_batch[i]
            tar = y_train_batch[i]
            prediction = te_prediction[i]
            #if len(np.where(inp > 0)[0]) <= max_seq_len:
            if len(np.where(tar > 0)[0]) == 1:
                print(tar, len(np.where(tar > 0)[0]))
                real_prediction = np.where(tar > 0)[0]
                target_pos = real_prediction #list(set(all_tr_label_tools_ids).intersection(set(real_prediction)))

                prediction_wts = tf.math.multiply(c_weights, prediction)
                #filter_embed.append(embed[i])
                n_topk = len(target_pos)
                top_k = tf.math.top_k(prediction, k=n_topk, sorted=True)
                print(i, top_k.indices.numpy())
                top_k_wts = tf.math.top_k(prediction_wts, k=n_topk, sorted=True)

                t_ip = t_ip.numpy()
                label_pos = np.where(t_ip > 0)[0]
                
                one_target_pos = target_pos[np.random.randint(len(target_pos))]
                #filter_embed_label_names.append(r_dict[str(one_target_pos)])
                #filter_embed_label.append(str(one_target_pos))
                
                i_names = ",".join([r_dict[str(int(item))] for item in t_ip[label_pos]  if item not in [0, "0"]])
                t_names = ",".join([r_dict[str(int(item))] for item in target_pos  if item not in [0, "0"]])

                last_i_tool = [r_dict[str(int(item))] for item in t_ip[label_pos]][-1]

                true_tools = [r_dict[str(int(item))] for item in target_pos]

                pred_tools = [r_dict[str(int(item))] for item in top_k.indices.numpy()  if item not in [0, "0"]]
                pred_tools_wts = [r_dict[str(int(item))] for item in top_k_wts.indices.numpy()  if item not in [0, "0"]]

                intersection = list(set(true_tools).intersection(set(pred_tools)))

                pub_prec = 0.0
                pub_prec_wt = 0.0

                if last_i_tool in published_connections:
                    true_pub_conn = published_connections[last_i_tool]

                    if len(pred_tools) > 0:
                        intersection_pub = list(set(true_pub_conn).intersection(set(pred_tools)))
                        intersection_pub_wt = list(set(true_pub_conn).intersection(set(pred_tools_wts)))
                        pub_prec = float(len(intersection_pub)) / len(pred_tools)
                        pub_prec_list.append(pub_prec)
                        pub_prec_wt = float(len(intersection_pub_wt)) / len(pred_tools)
                    else:
                        pub_prec = False
                        pub_prec_wt = False

                if len(pred_tools) > 0:
                    pred_precision = float(len(intersection)) / len(pred_tools)
                    precision.append(pred_precision)

                if pred_precision < 2.0:
            
                    print("Test batch {}, Tool sequence: {}".format(j+1, [r_dict[str(int(item))] for item in t_ip[label_pos]]))
                    print()
                    print("Test batch {}, True tools: {}".format(j+1, true_tools))
                    print()
                    print("Test batch {}, Predicted top {} tools: {}".format(j+1, n_topk, pred_tools))
                    print()
                    print("Test batch {}, Predicted top {} tools with weights: {}".format(j+1, n_topk, pred_tools_wts))
                    print()
                    print("Test batch {}, Precision: {}".format(j+1, pred_precision)) 
                    print()
                    print("Test batch {}, Published precision: {}".format(j+1, pub_prec))
                    print()
                    print("Test batch {}, Published precision with weights: {}".format(j+1, pub_prec_wt))
                    print()
                    print("Time taken to predict tools: {} seconds".format(diff_time))
                    #error_label_tools.append(select_tools[i])
                    print("=========================")
                print("--------------------------")
                generated_attention(att_weights[i], i_names, f_dict, r_dict)
                #plot_attention_head_axes(att_weights)
                print("Batch {} prediction finished ...".format(j+1))

    
    #import sys
    #sys.exit()
    
    '''te_lowest_t_ids = utils.read_saved_file(base_path + "data/te_lowest_t_ids.txt")
    lowest_t_ids = [int(item) for item in te_lowest_t_ids.split(",")]
    print(lowest_t_ids)
    lowest_t_ids = lowest_t_ids[:5]
    
    low_te_data = test_input[lowest_t_ids]
    low_te_labels = test_target[lowest_t_ids]
    #low_te_data_mask = utils.create_padding_mask(low_te_data)
    low_te_data = tf.cast(low_te_data, dtype=tf.float32)
    #print("Test lowest ids", low_te_data.shape, lowest_t_idslow_te_labels.shape)
    #low_te_pred_batch, low_att_weights = tf_loaded_model([low_te_data], training=False)
    low_topk = 20
    low_te_precision = list()
    low_te_pred_time = list()

    pred_s_time = time.time()
    if model_type in ["cnn", "rnn", "dnn"]:
        bat_low_prediction = tf_loaded_model(low_te_data, training=False)
    else:
        #low_te_data_mask = tf.cast(low_te_data_mask, dtype=tf.float32)
        #bat_low_prediction, att_weights = tf_loaded_model([low_te_data, low_te_data_mask], training=False)
        bat_embed_low, bat_low_prediction, att_weights = tf_loaded_model(low_te_data, training=False)
        print(bat_embed_low.shape, bat_low_prediction.shape, att_weights.shape)
    pred_e_time = time.time()
    low_diff_pred_t = 0 #(pred_e_time - pred_s_time) / float(len(lowest_t_ids))
    low_te_pred_time.append(low_diff_pred_t)
    print("Time taken to predict tools: {} seconds".format(low_diff_pred_t))

    for i, (low_inp, low_tar) in enumerate(zip(low_te_data, low_te_labels)):
        pred_s_time = time.time()
        if predict_rnn is True:
            low_prediction = tf_loaded_model([low_inp], training=False)
        else:
            
            low_prediction, att_weights = tf_loaded_model([low_inp], training=False)
        pred_e_time = time.time()
        low_diff_pred_t = pred_e_time - pred_s_time
        low_te_pred_time.append(low_diff_pred_t)
        print("Time taken to predict tools: {} seconds".format(low_diff_pred_t))
        low_prediction = bat_low_prediction[i]
        low_tar = low_te_labels[i]
        low_label_pos = np.where(low_tar > 0)[0]

        low_topk = len(low_label_pos)
        low_topk_pred = tf.math.top_k(low_prediction, k=low_topk, sorted=True)
        low_topk_pred = low_topk_pred.indices.numpy()
        
        low_label_pos_tools = [r_dict[str(int(item))] for item in low_label_pos if item not in [0, "0"]]
        low_pred_label_pos_tools = [r_dict[str(int(item))] for item in low_topk_pred if item not in [0, "0"]]

        low_intersection = list(set(low_label_pos_tools).intersection(set(low_pred_label_pos_tools)))
        low_pred_precision = float(len(low_intersection)) / len(low_label_pos)
        low_te_precision.append(low_pred_precision)

        low_inp_pos = np.where(low_inp > 0)[0]
        low_inp = low_inp.numpy()
        print(low_inp, low_inp_pos)
        print("{}, Low: test tool sequence: {}".format(i, [r_dict[str(int(item))] for item in low_inp[low_inp_pos]]))
        print()
        print("{},Low: True labels: {}".format(i, low_label_pos_tools))
        print()
        print("{},Low: Predicted labels: {}, Precision: {}".format(i, low_pred_label_pos_tools, low_pred_precision))
       
        print("-----------------")
        print()
        if predict_rnn is False:
            i_names = ",".join([r_dict[str(int(item))] for item in low_inp[low_inp_pos]])
            generated_attention(att_weights[i], i_names, f_dict, r_dict)

    if test_batches > 0:
        print("Batch Precision@{}: {}".format(n_topk, np.mean(precision)))
        print("Batch Published Precision@{}: {}".format(n_topk, np.mean(pub_prec_list)))
        print("Batch Trained model loading time: {} seconds".format(model_loading_time))
        print("Batch average seq pred time: {} seconds".format(np.mean(batch_pred_time)))
        print("Batch total model loading and pred time: {} seconds".format(model_loading_time + np.mean(batch_pred_time)))
        print()

    if len(lowest_t_ids) > 0:
        print("Low test prediction precision: {}".format(np.mean(low_te_precision)))
        print()
        print("Low: test average prediction time: {}".format(np.mean(low_te_pred_time)))
        print()
        
    #sys.exit()
    print("----------------------------")
    print()
    print("Predicting for individual sequences...")
    print()
    #print("Low precision on labels: {}".format(error_label_tools))
    #print("Low precision on labels: {}, # tools: {}".format(list(set(error_label_tools)), len(list(set(error_label_tools)))))'''
    # individual tools or seq prediction
    '''print()
    n_topk_ind = 20
    print("Predicting for individual tools or sequences")
    t_ip = np.zeros((25))
    t_ip[0] = int(f_dict["ncbi_eutils_esearch"])
    #t_ip[1] = int(f_dict["mtbls520_05a_import_maf"])
    #t_ip[2] = int(f_dict["mtbls520_06_import_traits"])
    #t_ip[3] = int(f_dict["mtbls520_07_species_diversity"])'''
    n_topk_ind = 20
    t_ip = np.zeros((1, 25))

    '''t_ip[0] = int(f_dict["bowtie2"])
    t_ip[1] = int(f_dict["hicexplorer_hicbuildmatrix"])
    t_ip[2] = int(f_dict["hicexplorer_hicfindtads"])
    t_ip[3] = int(f_dict["hicexplorer_hicpca"])'''

    ### To generate figure 4 - Transformer repo, run 2, head 1
    '''t_ip[0, 0] = int(f_dict["ctb_online_data_fetch"])
    t_ip[0, 1] = int(f_dict["ctb_change_title"])
    t_ip[0, 2] = int(f_dict["ctb_remDuplicates"])
    t_ip[0, 3] = int(f_dict["ctb_remIons"]) 
    t_ip[0, 4] = int(f_dict["ctb_compound_convert"])
    t_ip[0, 5] = int(f_dict["Show beginning1"])
    t_ip[0, 6] = int(f_dict["ctb_ob_genProp"])'''
    
    ### To generate figure 5 - Transformer repo, run 2, head 1
    '''t_ip[0, 0] = int(f_dict["trimmomatic"])
    t_ip[0, 1] = int(f_dict["hisat2"])
    t_ip[0, 2] = int(f_dict["featurecounts"])
    t_ip[0, 3] = int(f_dict["deseq2"]) 
    t_ip[0, 4] = int(f_dict["Filter1"])
    t_ip[0, 5] = int(f_dict["Grep1"])
    t_ip[0, 6] = int(f_dict["Add_a_column1"])
    t_ip[0, 7] = int(f_dict["table_compute"])
    t_ip[0, 8] = int(f_dict["tp_cut_tool"])
    t_ip[0, 9] = int(f_dict["join1"])
    t_ip[0, 10] = int(f_dict["tp_replace_in_column"])
    t_ip[0, 11] = int(f_dict["tp_cat"])'''
    
    ### To generate additional figure 5 - Transformer repo, run 2, head 1
    '''t_ip[0, 0] = int(f_dict["bowtie2"])
    t_ip[0, 1] = int(f_dict["hicexplorer_hicbuildmatrix"])
    t_ip[0, 2] = int(f_dict["hicexplorer_chicqualitycontrol"])
    t_ip[0, 3] = int(f_dict["hicexplorer_chicviewpointbackgroundmodel"]) 
    t_ip[0, 4] = int(f_dict["hicexplorer_chicviewpoint"])'''
    
    ### To generate additional figure 6 - Transformer repo, run 2, head 1
    '''t_ip[0, 0] = int(f_dict["Remove beginning1"])
    t_ip[0, 1] = int(f_dict["Cut1"])
    t_ip[0, 2] = int(f_dict["param_value_from_file"])
    t_ip[0, 3] = int(f_dict["kc-align"]) 
    t_ip[0, 4] = int(f_dict["sarscov2formatter"])'''
    
    
    
    
    #t_ip[0, 3] = int(f_dict["heinz"])

    #t_ip[4] = int(f_dict["prokka"])
    #t_ip[5] = int(f_dict["roary"]) 
    #t_ip[6] = int(f_dict["Cut1"])
    #t_ip[7] = int(f_dict["cat1"])
    #t_ip[8] = int(f_dict["anndata_manipulate"])
    # 'snpEff_build_gb', 'bwa_mem', 'samtools_view', snpeff_sars_cov_2
    
    # 1. snpeff_sars_cov_2
    # 2. anndata_import
    # 3. keras_train_and_eval
    # 4. cardinal_preprocessing, cardinal_segmentations, mass_spectrometry_imaging_filtering
    
    last_tool_name = "sarscov2formatter"
    
    t_ip = tf.convert_to_tensor(t_ip, dtype=tf.int64)
    t_ip = tf.cast(t_ip, dtype=tf.float32)
    
    pred_s_time = time.time()
    if model_type in ["cnn", "rnn", "dnn"]:
        prediction = tf_loaded_model(t_ip, training=False)
    else:
        #t_ip_mask = utils.create_padding_mask(t_ip)
        #t_ip_mask = tf.cast(t_ip_mask, dtype=tf.float32)
        #prediction, att_weights = tf_loaded_model([t_ip, t_ip_mask], training=False)
        embed, prediction, att_weights = tf_loaded_model(t_ip, training=False)
        print(embed.shape, prediction.shape, att_weights.shape)
    pred_e_time = time.time()
    print("Time taken to predict tools: {} seconds".format(pred_e_time - pred_s_time))
    prediction_cwts = tf.math.multiply(c_weights, prediction)

    top_k = tf.math.top_k(prediction, k=n_topk_ind, sorted=True)
    top_k_wts = tf.math.top_k(prediction_cwts, k=n_topk_ind, sorted=True)

    t_ip = t_ip.numpy()[0]
    label_pos = np.where(t_ip > 0)[0]
    print(t_ip)
    print(t_ip.shape, t_ip[label_pos])
    i_names = ",".join([r_dict[str(int(item))] for item in t_ip[label_pos] if item not in [0, "0"]])

    pred_tools = [r_dict[str(int(item))] for item in top_k.indices.numpy()[0] if item not in [0, "0"]]
    pred_tools_wts = [r_dict[str(int(item))] for item in top_k_wts.indices.numpy()[0] if item not in [0, "0"]]

    c_tools = []
    if str(f_dict[last_tool_name]) in compatible_tools:
        c_tools = [r_dict[str(item)] for item in compatible_tools[str(f_dict[last_tool_name])]]

    pred_intersection = list(set(pred_tools).intersection(set(c_tools)))
    prd_te_prec = len(pred_intersection) / float(n_topk_ind)

    print("Tool sequence: {}".format([r_dict[str(int(item))] for item in t_ip[label_pos]]))
    print()
    print("Compatible true tools: {}, size: {}".format(c_tools, len(c_tools)))
    print()
    print("Predicted top {} tools: {}".format(n_topk_ind, pred_tools))
    print()
    print("Predicted precision: {}".format(prd_te_prec))
    print()
    print("Correctly predicted tools: {}".format(pred_intersection))
    print()
    print("Predicted top {} tools with weights: {}".format(n_topk_ind, pred_tools_wts))
    print()
    '''if predict_rnn is False:
       generated_attention(att_weights, i_names, f_dict, r_dict)'''


def generated_attention(attention_weights, i_names, f_dict, r_dict):
    try:
        attention_heads = tf.squeeze(attention_weights, 0)
    except:
        attention_heads = attention_weights
    n_heads = attention_heads.shape[1]
    i_names = i_names.split(",")
    in_tokens = i_names
    out_tokens = i_names
    
    #print(attention_heads.shape)
    mean_att = np.mean(attention_heads, axis=0)
    for h, head in enumerate(attention_heads):
      plot_attention_head(in_tokens, out_tokens, head)
      #ax = fig.add_subplot(2, 4, h+1)
      #print(attention_heads.shape)
      break
    #print(mean_att.shape)
    #plot_attention_head(in_tokens, out_tokens, mean_att)
    #ax.set_xlabel(f'Head {h+1}')


def plot_attention_head(in_tokens, out_tokens, attention):

  font = {'family': 'serif', 'size': 12}
  fig_size = (6, 6)

  plt.rc('font', **font)
  size_title = 28
  size_label = 24
  dpi = 150
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.
  #translated_tokens = translated_tokens[1:]
  #print(attention)
  fig = plt.figure(figsize=fig_size)
  ax = plt.gca()
  cax = ax.matshow(attention[:len(in_tokens), :len(out_tokens)], interpolation='nearest')
  #ax.imshow(attention[:len(in_tokens), :len(out_tokens)], origin="upper")
  #ax.matshow(attention)
  ax.set_xlabel(f'Head')

  ax.set_xticks(range(len(in_tokens)))
  ax.set_xticklabels(in_tokens, rotation=90)

  ax.set_yticks(range(len(out_tokens)))
  #ax.set_yticklabels(out_tokens[::-1])
  ax.set_yticklabels(out_tokens)
  #cax = ax.matshow(data, interpolation='nearest')
  fig.colorbar(cax)
  plt.tight_layout()
  plt.savefig("plots/additional_figure_6.pdf", dpi=dpi, bbox_inches='tight')
  plt.savefig("plots/additional_figure_6.png", dpi=dpi, bbox_inches='tight')
  plt.show()


def plot_attention_head_axes(att_weights):
    seq_len = 25
    n_heads = 4
    attention_heads = tf.squeeze(att_weights, 0)
    attention_heads = attention_heads.numpy()
    print(attention_heads.shape)
    #print(attention_heads[0:, 0:])
    att_flatten = attention_heads.flatten()
    print(att_flatten.shape)
    block1 = att_flatten[0: seq_len * n_heads]
    block1 = block1.reshape((seq_len, n_heads))
    print(block1.shape)
    
    block2 = att_flatten[attention_heads.shape[0] * attention_heads.shape[1]: ]
    block2 = block2.reshape((attention_heads.shape[0], attention_heads.shape[1]))
    print(block2.shape)
    plt.matshow(block1)
    plt.matshow(block2)
    plt.show()

'''

Tool seqs for good attention plots:
'schicexplorer_schicqualitycontrol', 'schicexplorer_schicnormalize', 'schicexplorer_schicclustersvl'


# Tested tools: porechop, schicexplorer_schicqualitycontrol, schicexplorer_schicclustersvl, snpeff_sars_cov_2
    # sarscov2genomes, ivar_covid_aries_consensus, remove_nucleotide_deletions, pangolin
    # bowtie2,lofreq_call
    # dropletutils_read_10x
    # 'bowtie2', 'hicexplorer_hicbuildmatrix'
    # 'mtbls520_04_preparations', 'mtbls520_05a_import_maf', 'mtbls520_06_import_traits', 'mtbls520_07_species_diversity'
    # ctsm_fates: 'xarray_metadata_info', 'interactive_tool_panoply', 'xarray_select', '__EXTRACT_DATASET__'
    # msnbase_readmsdata: 'abims_xcms_xcmsSet', 'xcms_export_samplemetadata', 'xcms_plot_chromatogram'
    # ncbi_eutils_esearch: ncbi_eutils_elink
    # 1_create_conf: '5_calc_stat', '4_filter_sam', '2_map', 'conf4circos', '3_filter_single_pair'
    # pdaug_peptide_data_access: pdaug_tsvtofasta
    # 'pdaug_peptide_data_access', 'pdaug_tsvtofasta': 'pdaug_peptide_sequence_analysis', 'pdaug_fishers_plot', 'pdaug_sequence_property_based_descriptors'
    # 'rankprodthree', 'Remove beginning1', 'cat1', 'Cut1', 'interactions': 'biotranslator', 'awkscript'
    # rpExtractSink: rpCompletion', 'retropath2'
    # 'EMBOSS: transeq101', 'ncbi_makeblastdb', 'ncbi_blastp_wrapper', 'blast_parser', 'hcluster_sg'
    # 'Remove beginning1', 'Cut1', 'param_value_from_file', 'kc-align', 'sarscov2formatter', 'hyphy_fel'
    # abims_CAMERA_annotateDiffreport
    # cooler_csort_pairix
    # mycrobiota-split-multi-otutable_ensembl_gtf2gene_list
    # XY_Plot_1
    # mycrobiota-qc-report
    # 1_create_conf
    # RNAlien
    # ont_fast5_api_multi_to_single_fast5 
    # ctb_remIons

    Incorrect predictions
    # scpipe, 
    # 'delly_call', 'delly_merge'ivar_covid_aries_consensus
    # 'gmap_build', 'gsnap', 'sam_to_bam', 'filter', 'assign', 'polyA'
    # 'bioext_bealign', 'tn93_filter', 'hyphy_cfel'
    # sklearn_build_pipeline
    # split_file_to_collection', 'rdock_rbdock', 'xchem_pose_scoring', 'sucos_max_score'
    # 'rmcontamination', 'scaffold2fasta'  
    # 'rmcontamination', 'scaffold2fasta'
    # cat1', 'fastq_filter', 'cshl_fastq_to_fasta', 'filter_16s_wrapper_script 1'
    # 'TrimPrimer', 'Flash', 'Btrim64', 'uparse'
    # 'cshl_fastq_to_fasta', 'cshl_fastx_trimmer', 'fasta_tabular_converter
    # CryptoGenotyper
    # cooler_makebins
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks'
    # 'TrimPrimer', 'Flash', 'Btrim64'
    # cryptotyperanndata_import
    # ip_spot_detection_2d
    # 'picard_FastqToSam', 'TagBamWithReadSequenceExtended', 'FilterBAM', 'BAMTagHistogram'
    # 'basic_illumination', 'ashlar'
    # 'cghub_genetorrent', 'gatk_indel'
    # 'FeatureFinderMultiplex', 'HighResPrecursorMassCorrector', 'MSGFPlusAdapter', 'PeptideIndexer', 'IDMerger', 'ConsensusID'
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks'
    # 'PeakPickerHiRes', 'FileFilter', 'xcms-find-peaks', 'xcms-collect-peaks', 'xcms-group-peaks', 'xcms-blankfilter', 'xcms-dilutionfilter', 'camera-annotate-peaks', 'camera-group-fwhm', 'camera-find-adducts', 'camera-find-isotopes'
    # 'minfi_read450k', 'minfi_mset'
    # 'msnbase_readmsdata', 'abims_xcms_xcmsSet', 'abims_xcms_refine'
    # # 'snpEff_build_gb', 'bwa_mem', 'samtools_view',

'''

'''
def predict_seq():


    #sys.exit()
    # read test sequences
    r_dict = utils.read_file(base_path + "data/rev_dict.txt")
    f_dict = utils.read_file(base_path + "data/f_dict.txt")
    
    tf_loaded_model = tf.saved_model.load(model_path)
    #predictor = predict_sequences.PredictSequence(tf_loaded_model)

    #predictor(test_input, test_target, f_dict, r_dict)

    #tool_name = "cutadapt"
    #print("Prediction for {}...".format(tool_name))
    bowtie_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    bowtie_output = bowtie_output.write(0, [tf.constant(index_start_token, dtype=tf.int64)])
    #bowtie_output = bowtie_output.write(1, [tf.constant(295, dtype=tf.int64)])
    bowtie_o = tf.transpose(bowtie_output.stack())
    #tool_id = f_dict[tool_name]
    #print(tool_name, tool_id)
    tool_list = ["ctb_filter"]
    bowtie_input = np.zeros([1, 25])
    bowtie_input[:, 0] = index_start_token
    bowtie_input[:, 1] = f_dict[tool_list[0]]
    #bowtie_input[:, 2] = f_dict[tool_list[1]]
    #bowtie_input[:, 3] = f_dict["featurecounts"]
    #bowtie_input[:, 4] = f_dict["deseq2"]
    bowtie_input = tf.constant(bowtie_input, dtype=tf.int64)
    print(bowtie_input, bowtie_output, bowtie_o)
    bowtie_pred, _ = tf_loaded_model([bowtie_input, bowtie_o], training=False)
    print(bowtie_pred.shape)
    top_k = tf.math.top_k(bowtie_pred, k=10)
    print("Top k: ", bowtie_pred.shape, top_k, top_k.indices)
    print(np.all(top_k.indices.numpy(), axis=-1))
    print("Predicted tools for {}: {}".format( ",".join(tool_list), [r_dict[str(item)] for item in top_k.indices.numpy()[0][0]]))
    print()
    #print("Generating predictions...")
    #generated_attention(tf_loaded_model, f_dict, r_dict)


def generated_attention(trained_model, f_dict, r_dict):

    np_output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    np_output_array = np_output_array.write(0, [tf.constant(index_start_token, dtype=tf.int64)])

    n_target_items = 5
    n_input = np.zeros([1, 25])
    n_input[:, 0] = index_start_token
    n_input[:, 1] = f_dict["hicexplorer_hicadjustmatrix"]
    #n_input[:, 2] = f_dict["hicexplorer_hicbuildmatrix"]
    #n_input[:, 3] = f_dict["hicexplorer_hicfindtads"]
    #n_input[:, 4] = f_dict["deseq2"]
    #n_input[:, 5] = f_dict["Add_a_column1"]
    #n_input[:, 6] = f_dict["table_compute"]
    a_input = n_input
    n_input = tf.constant(n_input, dtype=tf.int64)
   
    for i in range(n_target_items):
        #print(i, index)
        output = tf.transpose(np_output_array.stack())
        print("decoder input: ", n_input, output, output.shape)
        orig_predictions, _ = trained_model([n_input, output], training=False)
        #print(orig_predictions.shape)trimmomatic
        #print("true target seq real: ", te_tar_real)
        #print("Pred seq argmax: ", tf.argmax(orig_predictions, axis=-1))
        predictions = orig_predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        np_output_array = np_output_array.write(i+1, predicted_id[0])
    print(output, np_output_array.stack(), output.numpy())
    print("----------")
    last_decoder_layer = "decoder_layer4_block2"
    _, attention_weights = trained_model([n_input, output[:,:-1]], training=False)
    pred_attention = attention_weights[last_decoder_layer]

    print(attention_weights[last_decoder_layer].shape)
    head = 0
    attention_heads = tf.squeeze(attention_weights[last_decoder_layer], 0)
    pred_attention = attention_heads[head]
    print(pred_attention)

    #print(attention_weights)
    in_tokens = [r_dict[str(int(item))] for itscanpy_read_10xem in a_input[0] if item > 0]
    out_tokens = [r_dict[str(int(item))] for item in output.numpy()[0]]
    out_tokens = out_tokens[1:]
    print(in_tokens)
    print(out_tokens)
    pred_attention = pred_attention[:,:len(in_tokens)]
    print(pred_attention)
    plot_attention_head(in_tokens, out_tokens, pred_attention)

scanpy_read_10x
def plot_attention_head(in_tokens, out_tokens, attention):
  # The plot is of the attention when a token walog_19_09_22_GPU_transformer_full_datas generated.
  # The model didn't generate `<START>` in the output. Skip it.

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attention, interpolation='nearest')ctb_chemfp_nxn_clustering
  fig.colorbar(cax)

  #ax = plt.gca()
  #ax.matshow(attention)

  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(out_tokens)))

  ax.set_xticklabels(in_tokens, rotation=90)
  ax.set_yticklabels(out_tokens)

  plt.show()

'''

if __name__ == "__main__":
    predict_seq()
    #visualize_loss_acc()
