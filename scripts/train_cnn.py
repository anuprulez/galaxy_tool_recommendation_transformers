import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Embedding, Input, Dense, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Lambda, GlobalMaxPooling1D
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
    cnn_units = config["feed_forward_dim"]
    dropout = config["dropout"]
    
    model = Sequential()
    model.add(Embedding(vocab_size+1, cnn_units, input_length=config["maximum_path_length"]))
    model.add(Lambda(lambda x: tf.expand_dims(x, 3)))
    model.add(Conv2D(cnn_units, kernel_size=(16, 3), activation = 'relu', kernel_initializer='he_normal', padding = 'VALID'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(cnn_units, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='sigmoid'))
    
    model.summary()

    return model


def create_cnn_architecture(train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, c_tools, pub_conn, tr_t_freq, config):

    print("Training CNN...")
    vocab_size = len(f_dict) + 1

    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    model = create_model(vocab_size, config)
    prev_model_number = config["restart_step"]
    if prev_model_number > 0:
        model_path_h5 = model_path + str(prev_model_number) + "/tf_model_h5/model.h5"
        model.load_weights(model_path_h5)

    u_tr_y_labels, u_tr_y_labels_dict = utils.get_u_tr_labels(train_labels)
    u_te_y_labels, u_te_y_labels_dict = utils.get_u_tr_labels(test_labels)

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
    utils.write_file(base_path + "data/te_lowest_t_ids.txt", ",".join([str(item) for item in te_lowest_t_ids]))
    tr_log_step = config["tr_logging_step"]
    te_log_step = config["te_logging_step"]
    n_train_steps = config["n_train_iter"]
    te_batch_size = config["te_batch_size"]
    tr_batch_size = config["tr_batch_size"]
    model_type = config["model_type"]
    batch_index = prev_model_number
    sel_tools = list()
    for batch in range(n_train_steps):
        print("Total train data size: ", train_data.shape, train_labels.shape)
        x_train, y_train, sel_tools = utils.sample_balanced_tr_y(train_data, train_labels, u_tr_y_labels_dict, tr_batch_size, tr_t_freq, sel_tools)
        print("Batch train data size: ", batch_index, x_train.shape, y_train.shape)
        all_sel_tool_ids.extend(sel_tools)
        with tf.GradientTape() as model_tape:
            prediction = model(x_train, training=True)
            tr_loss, tr_cat_loss = utils.compute_loss(y_train, prediction)
            tr_acc = tf.reduce_mean(utils.compute_acc(y_train, prediction))
        trainable_vars = model.trainable_variables
        model_gradients = model_tape.gradient(tr_loss, trainable_vars)
        enc_optimizer.apply_gradients(zip(model_gradients, trainable_vars))
        epo_tr_batch_loss.append(tr_loss.numpy())
        epo_tr_batch_acc.append(tr_acc.numpy())
        epo_tr_batch_categorical_loss.append(tr_cat_loss.numpy())
        print("Step {}/{}, training binary loss: {}, categorical_loss: {}, training accuracy: {}".format(batch_index+1, n_train_steps, tr_loss.numpy(), tr_cat_loss.numpy(), tr_acc.numpy()))
        if (batch_index+1) % te_log_step == 0:
            print("Predicting on test data...")
            te_loss, te_acc, test_cat_loss, te_prec, low_te_prec = utils.validate_model(test_data, test_labels, te_batch_size, model, f_dict, r_dict, u_te_y_labels_dict, trained_on_labels, te_lowest_t_ids, model_type)
            epo_te_batch_loss.append(te_loss)
            epo_te_batch_acc.append(te_acc)
            epo_te_batch_categorical_loss.append(test_cat_loss)
            epo_te_precision.append(te_prec)
            epo_low_te_precision.append(low_te_prec)
            utils.write_file(base_path + "data/epo_te_batch_loss.txt", te_loss)
            utils.write_file(base_path + "data/epo_te_batch_acc.txt", te_acc)
            utils.write_file(base_path + "data/epo_te_batch_categorical_loss.txt", test_cat_loss)
            utils.write_file(base_path + "data/epo_te_precision.txt", te_prec)
            utils.write_file(base_path + "data/epo_low_te_precision.txt", low_te_prec)
        print()
        if (batch_index+1) % tr_log_step == 0:
            print("Saving model at training step {}/{}".format(batch_index + 1, n_train_steps))
            tf_path = model_path + "{}/".format(batch_index+1)
            tf_model_save = model_path + "{}/tf_model/".format(batch_index+1)
            tf_model_save_h5 = model_path + "{}/tf_model_h5/".format(batch_index+1)
            if not os.path.isdir(tf_path):
                os.mkdir(tf_path)
                os.mkdir(tf_model_save)
                os.mkdir(tf_model_save_h5)
            tf.saved_model.save(model, tf_model_save)
            utils.save_model_file(tf_model_save_h5, model, r_dict, c_wts, c_tools, pub_conn)
            utils.write_file(base_path + "data/epo_tr_batch_loss.txt", tr_loss.numpy())
            utils.write_file(base_path + "data/epo_tr_batch_acc.txt", tr_acc.numpy())
            utils.write_file(base_path + "data/epo_tr_batch_categorical_loss.txt", tr_cat_loss.numpy())
        batch_index += 1
        if batch_index > n_train_steps - 1:
            break
