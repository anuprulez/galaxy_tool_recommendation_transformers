import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import h5py
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding, SpatialDropout1D, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer





batch_size = 100
test_batches = 1000
n_topk = 1
max_seq_len = 25

embed_dim = 128 # Embedding size for each token d_model
num_heads = 4 # Number of attention heads
ff_dim = 128 # Hidden layer size in feed forward network inside transformer # dff
dropout = 0.2
seq_len = 25

# Set to true only when RNN model should be executed
predict_rnn = False


# Set the model path correctly. In the codebase, test data is zipped because of its large size. 
# Unzip the H5 test data, store the H5 file at `models/transformer/saved_data/*.h5`.
# Verify the test data and model paths.

run_number = "run2/"
model_number = 40000

base_path = "../final_data/aug_22_data/transformer/" + run_number



model_path = base_path + "saved_model/" + str(model_number) + "/tf_model/"
model_path_h5 = base_path + "saved_model/" + str(model_number) + "/tf_model_h5/"


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, attention_scores = self.att(inputs, inputs, inputs, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def read_file(file_path):
    """
    Read a file
    """
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def write_file(file_path, content):
    """
    Write a file
    """
    remove_file(file_path)
    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(content))


def create_transformer_model(maxlen, vocab_size):
    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=[x, outputs, weights])


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


def read_h5_model():
    print(model_path_h5)
    h5_path = model_path_h5 + "model.h5"
    model_h5 = h5py.File(h5_path, 'r')

    r_dict = json.loads(model_h5["reverse_dict"][()].decode("utf-8"))
    m_load_s_time = time.time()
    tf_loaded_model = create_transformer_model(seq_len, len(r_dict) + 1)
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
    m_load_s_time = time.time()
    tf_loaded_model = tf.saved_model.load(model_path)
    m_load_e_time = time.time()
    m_l_time = m_load_e_time - m_load_s_time
    r_dict = read_file(base_path + "data/rev_dict.txt")
    f_dict = read_file(base_path + "data/f_dict.txt")
    c_weights = read_file(base_path + "data/class_weights.txt")
    c_tools = read_file(base_path + "data/compatible_tools.txt")
    s_conn = read_file(base_path + "data/published_connections.txt")

    return tf_loaded_model, f_dict, r_dict, c_weights, c_tools, s_conn, m_l_time


def recommend_tools():
 
    path_test_data = base_path + "saved_data/test.h5"

    file_obj = h5py.File(path_test_data, 'r')

    test_input = np.array(file_obj["input"])
    test_target = np.array(file_obj["target"])

    print(test_input.shape, test_target.shape)


    tf_loaded_model, f_dict, r_dict, class_weights, compatible_tools, published_connections, model_loading_time = read_h5_model()

    c_weights = list(class_weights.values())

    c_weights = tf.convert_to_tensor(c_weights, dtype=tf.float32)

    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_target)

    precision = list()
    pub_prec_list = list()
    error_label_tools = list()
    batch_pred_time = list()
    for j in range(test_batches):

        #te_x_batch, y_train_batch, selected_label_tools, bat_ind = sample_balanced_tr_y(test_input, test_target, u_te_y_labels_dict)

        te_x_batch = test_input[j * batch_size : j * batch_size + batch_size, :]
        y_train_batch = test_target[j * batch_size : j * batch_size + batch_size, :]

        te_x_batch = tf.cast(te_x_batch, dtype=tf.float32, name="input_2")
        
        pred_s_time = time.time()

        embed, te_prediction, att_weights = tf_loaded_model(te_x_batch, training=False)
           
        pred_e_time = time.time()
        diff_time = (pred_e_time - pred_s_time) / float(batch_size)
        batch_pred_time.append(diff_time)
        
        for i, (inp, tar) in enumerate(zip(te_x_batch, y_train_batch)):

            t_ip = te_x_batch[i]
            tar = y_train_batch[i]
            prediction = te_prediction[i]
            if len(np.where(inp > 0)[0]) <= max_seq_len:
                real_prediction = np.where(tar > 0)[0]
                target_pos = real_prediction

                prediction_wts = tf.math.multiply(c_weights, prediction)

                n_topk = len(target_pos)
                top_k = tf.math.top_k(prediction, k=n_topk, sorted=True)
                top_k_wts = tf.math.top_k(prediction_wts, k=n_topk, sorted=True)

                t_ip = t_ip.numpy()
                label_pos = np.where(t_ip > 0)[0]
                
                one_target_pos = target_pos[np.random.randint(len(target_pos))]
                
                i_names = ",".join([r_dict[str(int(item))] for item in t_ip[label_pos]  if item not in [0, "0"]])
                t_names = ",".join([r_dict[str(int(item))] for item in target_pos  if item not in [0, "0"]])

                last_i_tool = [r_dict[str(int(item))] for item in t_ip[label_pos]][-1]

                true_tools = [r_dict[str(int(item))] for item in target_pos]
                
                if i_names.split(",")[0] == "ctb_online_data_fetch":
                    print(i, i_names)
                    print()
                    print(i, t_names)
                    print("-----------")
                    #generated_attention(i, att_weights[i], i_names, f_dict, r_dict)
        print("Batch {} prediction finished ...".format(j+1))


def generated_attention(tool_seq_index, attention_weights, i_names, f_dict, r_dict):
    try:
        attention_heads = tf.squeeze(attention_weights, 0)
    except:
        attention_heads = attention_weights
    n_heads = attention_heads.shape[1]
    attention_heads = attention_heads.numpy()
    i_names = i_names.split(",")
    in_tokens = i_names
    out_tokens = i_names

    #mean_att = np.mean(attention_heads, axis=0)
    for h, head in enumerate(attention_heads):
      plot_attention_head(in_tokens, out_tokens, head, h, tool_seq_index)
      #plot_attention_head_bokeh(in_tokens, out_tokens, head, h, tool_seq_index)


def plot_attention_head(in_tokens, out_tokens, attention, h, tool_seq_index):
  fig_size = (16, 8)
  font = {'family': 'serif', 'size': 24}
  plt.rc('font', **font)
  fig = plt.figure(figsize=fig_size)
  ax = plt.gca()
  cax = ax.matshow(attention[:len(in_tokens), :len(out_tokens)], interpolation='nearest')

  ax.set_xticks(range(len(in_tokens)))
  ax.set_xticklabels(in_tokens, rotation=90)

  ax.set_yticks(range(len(out_tokens)))
  ax.set_yticklabels(out_tokens)
  fig.colorbar(cax)
  plt.tight_layout()
  plt.show()
  f_name = "../attention_plots/{}{}/attention_plot_seq_num_{}_head_num_{}.png".format(run_number, model_number,tool_seq_index, h)
  plt.savefig(f_name, dpi=300)


def plot_attention_head_bokeh(in_tokens, out_tokens, attention, h, tool_seq_index):
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.transform import linear_cmap
    from bokeh.models import ColorBar, BasicTicker, PrintfTickFormatter, ColumnDataSource
    from bokeh.layouts import layout
    from bokeh.palettes import Viridis256
    import numpy as np
    
    # Generate some synthetic data for the heatmap
    data = attention[:len(in_tokens), :len(out_tokens)] #np.random.random((10, 10))
    
    # Prepare data for Bokeh
    x = in_tokens #np.arange(0, 10)
    y = out_tokens #np.arange(0, 10)
    xx, yy = np.meshgrid(x, y)
    d = {'x': xx.flatten(), 'y': yy.flatten(), 'value': data.flatten()}
    
    source = ColumnDataSource(d)
    
    # Define the color mapper for the heatmap
    mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(d['value']), high=max(d['value']))
    
    # Create a figure for the heatmap
    p = figure(title="Heatmap Example", 
               x_axis_label='X-Axis', y_axis_label='Y-Axis', 
               tools="", toolbar_location=None, 
               x_range=(0, 10), y_range=(0, 10))
    
    # Create the heatmap by drawing rectangles
    p.rect(x='x', y='y', width=1, height=1, source=source,
           line_color=None, fill_color=mapper)

    # Add color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0),
                         ticker=BasicTicker(desired_num_ticks=10),
                         formatter=PrintfTickFormatter(format="%.2f"))

    p.add_layout(color_bar, 'right')
    
    # Show the plot
    f_name = "../attention_plots/{}{}/attention_plot_seq_num_{}_head_num_{}.html".format(run_number, model_number,tool_seq_index, h)
    output_file(f_name)
    show(p)


recommend_tools()