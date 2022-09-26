import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Embedding, Input, Dense, GRU
from tensorflow.keras.models import Model

import utils


'''embed_dim = 128
ff_dim = 128
dropout = 0.1
n_train_batches = 50
batch_size = 128
test_logging_step = 10
train_logging_step = 10
te_batch_size = batch_size
learning_rate = 1e-3'''

base_path = "log/"
model_path = base_path + "saved_model/"


binary_ce = tf.keras.losses.BinaryCrossentropy()
binary_acc = tf.keras.metrics.BinaryAccuracy()
categorical_ce = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)


def create_model(vocab_size, config):
    gru_units = config["feed_forward_dim"]
    dropout = config["dropout"]

    seq_inputs = Input(batch_shape=(None, config["maximum_path_length"]))

    gen_embedding = Embedding(vocab_size, config["embedding_dim"], mask_zero=True)
    in_gru = GRU(gru_units, return_sequences=True, return_state=False)
    out_gru = GRU(gru_units, return_sequences=False, return_state=True)
    enc_fc = Dense(vocab_size, activation='sigmoid', kernel_regularizer="l2")

    embed = gen_embedding(seq_inputs)

    embed = Dropout(dropout)(embed)

    gru_output = in_gru(embed)

    gru_output = Dropout(dropout)(gru_output)

    gru_output, hidden_state = out_gru(gru_output)

    gru_output = Dropout(dropout)(gru_output)

    fc_output = enc_fc(gru_output)

    return Model(inputs=seq_inputs, outputs=[fc_output])


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


def sample_balanced_te_y(x_seqs, y_labels, ulabels_tr_y_dict, b_size):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()
    sel_tools = list()
    for l_tool in batch_y_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        rand_s_index = np.random.randint(0, len(seq_indices), 1)[0]
        rand_sample = seq_indices[rand_s_index]
        sel_tools.append(l_tool)
        if rand_sample not in rand_batch_indices:
            rand_batch_indices.append(rand_sample)
            label_tools.append(l_tool)
        if len(rand_batch_indices) == b_size:
            break
    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, sel_tools


def sample_balanced_tr_y(x_seqs, y_labels, ulabels_tr_y_dict, b_size, tr_t_freq, prev_sel_tools):
    batch_y_tools = list(ulabels_tr_y_dict.keys())
    random.shuffle(batch_y_tools)
    label_tools = list()
    rand_batch_indices = list()
    sel_tools = list()

    unselected_tools = [t for t in batch_y_tools if t not in prev_sel_tools]
    rand_selected_tools = unselected_tools[:b_size]

    for l_tool in rand_selected_tools:
        seq_indices = ulabels_tr_y_dict[l_tool]
        random.shuffle(seq_indices)
        rand_s_index = np.random.randint(0, len(seq_indices), 1)[0]
        rand_sample = seq_indices[rand_s_index]
        sel_tools.append(l_tool)
        rand_batch_indices.append(rand_sample)
        label_tools.append(l_tool)

    x_batch_train = x_seqs[rand_batch_indices]
    y_batch_train = y_labels[rand_batch_indices]

    unrolled_x = tf.convert_to_tensor(x_batch_train, dtype=tf.int64)
    unrolled_y = tf.convert_to_tensor(y_batch_train, dtype=tf.int64)
    return unrolled_x, unrolled_y, sel_tools


def compute_loss(y_true, y_pred, class_weights=None):
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = binary_ce(y_true, y_pred)
    categorical_loss = categorical_ce(y_true, y_pred)

    if class_weights is None:
        return tf.reduce_mean(loss), categorical_loss
    return tf.tensordot(loss, class_weights, axes=1), categorical_loss


def compute_acc(y_true, y_pred):
    return binary_acc(y_true, y_pred)


def validate_model(te_x, te_y, te_batch_size, model, f_dict, r_dict, ulabels_te_dict, tr_labels, lowest_t_ids):
    print("Total test data size: ", te_x.shape, te_y.shape)
    te_x_batch, y_train_batch, _ = sample_balanced_te_y(te_x, te_y, ulabels_te_dict, te_batch_size)
    print("Batch test data size: ", te_x_batch.shape, y_train_batch.shape)
    te_pred_batch = model([te_x_batch], training=False)
    test_acc = tf.reduce_mean(compute_acc(y_train_batch, te_pred_batch))
    test_err, test_categorical_loss = compute_loss(y_train_batch, te_pred_batch)

    te_pre_precision = list()

    for idx in range(te_pred_batch.shape[0]):
        label_pos = np.where(y_train_batch[idx] > 0)[0]
        # verify only on those tools are present in labels in training
        label_pos = list(set(tr_labels).intersection(set(label_pos)))
        topk_pred = tf.math.top_k(te_pred_batch[idx], k=len(label_pos), sorted=True)
        topk_pred = topk_pred.indices.numpy()
        try:
            label_pos_tools = [r_dict[str(item)] for item in label_pos if item not in [0, "0"]]
            pred_label_pos_tools = [r_dict[str(item)] for item in topk_pred if item not in [0, "0"]]
        except Exception as e:
            label_pos_tools = [r_dict[item] for item in label_pos if item not in [0, "0"]]
            pred_label_pos_tools = [r_dict[item] for item in topk_pred if item not in [0, "0"]]
        intersection = list(set(label_pos_tools).intersection(set(pred_label_pos_tools)))
        if len(topk_pred) > 0:
            pred_precision = float(len(intersection)) / len(topk_pred)
            te_pre_precision.append(pred_precision)
            print("True labels: {}".format(label_pos_tools))
            print()
            print("Predicted labels: {}, Precision: {}".format(pred_label_pos_tools, pred_precision))
            print("-----------------")
            print()

        if idx == te_batch_size - 1:
            break

    print("Test lowest ids", len(lowest_t_ids))
    low_te_data = te_x[lowest_t_ids]
    low_te_labels = te_y[lowest_t_ids]
    low_te_pred_batch = model([low_te_data], training=False)
    low_test_err, low_test_categorical_loss = compute_loss(low_te_labels, low_te_pred_batch)

    low_te_precision = list()
    for idx in range(low_te_pred_batch.shape[0]):
        low_label_pos = np.where(low_te_labels[idx] > 0)[0]
        low_label_pos = list(set(tr_labels).intersection(set(low_label_pos)))
        low_topk_pred = tf.math.top_k(low_te_pred_batch[idx], k=len(low_label_pos), sorted=True)
        low_topk_pred = low_topk_pred.indices.numpy()
        try:
            low_label_pos_tools = [r_dict[str(item)] for item in low_label_pos if item not in [0, "0"]]
            low_pred_label_pos_tools = [r_dict[str(item)] for item in low_topk_pred if item not in [0, "0"]]
        except Exception as e:
            low_label_pos_tools = [r_dict[item] for item in low_label_pos if item not in [0, "0"]]
            low_pred_label_pos_tools = [r_dict[item] for item in low_topk_pred if item not in [0, "0"]]

        low_intersection = list(set(low_label_pos_tools).intersection(set(low_pred_label_pos_tools)))
        if len(low_label_pos) > 0:
            low_pred_precision = float(len(low_intersection)) / len(low_label_pos)
            low_te_precision.append(low_pred_precision)
            print("Low: True labels: {}".format(low_label_pos_tools))
            print()
            print("Low: Predicted labels: {}, Precision: {}".format(low_pred_label_pos_tools, low_pred_precision))
            print("-----------------")
            print()

    print("Test binary error: {}, test categorical loss: {}, test categorical accuracy: {}".format(test_err.numpy(), test_categorical_loss.numpy(), test_acc.numpy()))
    print("Test prediction precision: {}".format(np.mean(te_pre_precision)))
    print("Low test binary error: {}".format(low_test_err.numpy()))
    print("Low test prediction precision: {}".format(np.mean(low_te_precision)))

    print("Test finished")
    return test_err.numpy(), test_acc.numpy(), test_categorical_loss.numpy(), np.mean(te_pre_precision), np.mean(low_te_precision)


def create_rnn_architecture(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, c_tools, pub_conn, tr_t_freq, config):

    print("Training RNN...")
    vocab_size = len(f_dict) + 1
    #maxlen = train_data.shape[1]

    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    model = create_model(vocab_size, config)

    u_tr_y_labels, u_tr_y_labels_dict = get_u_tr_labels(train_labels)
    u_te_y_labels, u_te_y_labels_dict = get_u_tr_labels(test_labels)

    trained_on_labels = [int(item) for item in list(u_tr_y_labels_dict.keys())]

    epo_tr_batch_loss = list()
    epo_tr_batch_acc = list()
    epo_tr_batch_categorical_loss = list()
    epo_te_batch_loss = list()
    epo_te_batch_acc = list()
    epo_te_batch_categorical_loss = list()
    all_sel_tool_ids = list()
    epo_te_precision = list()
    epo_low_te_precision = list()

    te_lowest_t_ids = utils.get_low_freq_te_samples(test_data, test_labels, tr_t_freq)
    tr_log_step = config["tr_logging_step"]
    te_log_step = config["te_logging_step"]
    n_train_steps = config["n_train_iter"]
    te_batch_size = config["te_batch_size"]
    tr_batch_size = config["tr_batch_size"]
    
    #compatible_tools = utils.read_file(base_path + "data/compatible_tools.txt")
    #published_connections = utils.read_file(base_path + "data/published_connections.txt")

    sel_tools = list()
    for batch in range(n_train_steps):
        print("Total train data size: ", train_data.shape, train_labels.shape)
        x_train, y_train, sel_tools = sample_balanced_tr_y(train_data, train_labels, u_tr_y_labels_dict, tr_batch_size, tr_t_freq, sel_tools)
        print("Batch train data size: ", x_train.shape, y_train.shape)
        all_sel_tool_ids.extend(sel_tools)
        with tf.GradientTape() as model_tape:
            prediction = model([x_train], training=True)
            tr_loss, tr_cat_loss = compute_loss(y_train, prediction)
            tr_acc = tf.reduce_mean(compute_acc(y_train, prediction))
        trainable_vars = model.trainable_variables
        model_gradients = model_tape.gradient(tr_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(model_gradients, trainable_vars))
        epo_tr_batch_loss.append(tr_loss.numpy())
        epo_tr_batch_acc.append(tr_acc.numpy())
        epo_tr_batch_categorical_loss.append(tr_cat_loss.numpy())
        print("Step {}/{}, training binary loss: {}, categorical_loss: {}, training accuracy: {}".format(batch+1, n_train_steps, tr_loss.numpy(), tr_cat_loss.numpy(), tr_acc.numpy()))
        if (batch+1) % te_log_step == 0:
            print("Predicting on test data...")
            te_loss, te_acc, test_cat_loss, te_prec, low_te_prec = validate_model(test_data, test_labels, te_batch_size, model, f_dict, r_dict, u_te_y_labels_dict, trained_on_labels, te_lowest_t_ids)
            epo_te_batch_loss.append(te_loss)
            epo_te_batch_acc.append(te_acc)
            epo_te_batch_categorical_loss.append(test_cat_loss)
            epo_te_precision.append(te_prec)
            epo_low_te_precision.append(low_te_prec)
        print()
        if (batch+1) % tr_log_step == 0:
            print("Saving model at training step {}/{}".format(batch + 1, n_train_steps))
            tf_path = model_path + "{}/".format(batch+1)
            tf_model_save = model_path + "{}/tf_model/".format(batch+1)
            tf_model_save_h5 = model_path + "{}/tf_model_h5/".format(batch+1)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
                os.mkdir(tf_model_save)
                os.mkdir(tf_model_save_h5)

            tf.saved_model.save(model, tf_model_save)
            utils.save_model_file(tf_model_save_h5, model, r_dict, c_wts, c_tools, pub_conn)
    new_dict = dict()
    for k in u_tr_y_labels_dict:
        new_dict[str(k)] = ",".join([str(item) for item in u_tr_y_labels_dict[k]])

    utils.write_file(base_path + "data/epo_tr_batch_loss.txt", ",".join([str(item) for item in epo_tr_batch_loss]))
    utils.write_file(base_path + "data/epo_tr_batch_acc.txt", ",".join([str(item) for item in epo_tr_batch_acc]))
    utils.write_file(base_path + "data/epo_te_batch_loss.txt", ",".join([str(item) for item in epo_te_batch_loss]))
    utils.write_file(base_path + "data/epo_te_batch_acc.txt", ",".join([str(item) for item in epo_te_batch_acc]))
    utils.write_file(base_path + "data/epo_tr_batch_categorical_loss.txt", ",".join([str(item) for item in epo_tr_batch_categorical_loss]))
    utils.write_file(base_path + "data/epo_te_batch_categorical_loss.txt", ",".join([str(item) for item in epo_te_batch_categorical_loss]))
    utils.write_file(base_path + "data/epo_te_precision.txt", ",".join([str(item) for item in epo_te_precision]))
    utils.write_file(base_path + "data/all_sel_tool_ids.txt", ",".join([str(item) for item in all_sel_tool_ids]))
    utils.write_file(base_path + "data/epo_low_te_precision.txt", ",".join([str(item) for item in epo_low_te_precision]))
    utils.write_file(base_path + "data/u_tr_y_labels_dict.txt", new_dict)
    utils.write_file(base_path + "data/te_lowest_t_ids.txt", ",".join([str(item) for item in te_lowest_t_ids]))
