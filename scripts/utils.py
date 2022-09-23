import os
import numpy as np
import json
import h5py
import pandas as pd


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


def save_h5_data(inp, tar, filename):
    hf_file = h5py.File(filename, 'w')
    hf_file.create_dataset("input", data=inp)
    hf_file.create_dataset("target", data=tar)
    hf_file.close()


def get_low_freq_te_samples(te_data, te_target, tr_freq_dict):
    lowest_tool_te_ids = list()
    lowest_t_ids = get_lowest_tools(tr_freq_dict)
    for i, te_labels in enumerate(te_target):
        tools_pos = np.where(te_labels > 0)[0]
        tools_pos = [str(int(item)) for item in tools_pos]
        intersection = list(set(tools_pos).intersection(set(lowest_t_ids)))
        if len(intersection) > 0:
            lowest_tool_te_ids.append(i)
            lowest_t_ids = [item for item in lowest_t_ids if item not in intersection]
    return lowest_tool_te_ids


def save_processed_workflows(file_path, unique_paths):
    workflow_paths_unique = ""
    for path in unique_paths:
        workflow_paths_unique += path + "\n"
    with open(file_path, "w") as workflows_file:
        workflows_file.write(workflow_paths_unique)


def format_tool_id(tool_link):
    """
    Extract tool id from tool link
    """
    tool_id_split = tool_link.split("/")
    tool_id = tool_id_split[-2] if len(tool_id_split) > 1 else tool_link
    return tool_id


def save_model_file(model_path, model, r_dict, c_wts, c_tools, s_conn, f_name="model.h5"):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_file = model_path + f_name
    model.save_weights(model_file)
    hf_file = h5py.File(model_file, 'r+')

    model_values = {
        "reverse_dict": r_dict,
        "class_weights": c_wts,
        "compatible_tools": c_tools,
        "standard_connections": s_conn
    }

    for k in model_values:
        hf_file.create_dataset(k, data=json.dumps(model_values[k]))
    hf_file.close()


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def verify_oversampling_freq(oversampled_tr_data, rev_dict):
    """
    Compute the frequency of tool sequences after oversampling
    """
    freq_dict = dict()
    freq_dict_names = dict()
    for tr_data in oversampled_tr_data:
        t_pos = np.where(tr_data > 0)[0]
        last_tool_id = str(int(tr_data[t_pos[-1]]))
        if last_tool_id not in freq_dict:
            freq_dict[last_tool_id] = 0
            freq_dict_names[rev_dict[int(last_tool_id)]] = 0
        freq_dict[last_tool_id] += 1
        freq_dict_names[rev_dict[int(last_tool_id)]] += 1
    s_freq = dict(sorted(freq_dict_names.items(), key=lambda kv: kv[1], reverse=True))
    return s_freq


def collect_sampled_tool_freq(collected_dict, c_freq):
    for t in c_freq:
        if t not in collected_dict:
            collected_dict[t] = int(c_freq[t])
        else:
            collected_dict[t] += int(c_freq[t])
    return collected_dict


def save_data_as_dict(f_dict, r_dict, inp, tar, save_path):
    inp_tar = dict()
    for index, (i, t) in enumerate(zip(inp, tar)):
        i_pos = np.where(i > 0)[0]
        i_seq = ",".join([str(int(item)) for item in i[1:i_pos[-1] + 1]])
        t_pos = np.where(t > 0)[0]
        t_seq = ",".join([str(int(item)) for item in t[1:t_pos[-1] + 1]])
        if i_seq not in inp_tar:
            inp_tar[i_seq] = list()
        inp_tar[i_seq].append(t_seq)
    size = 0
    for item in inp_tar:
        size += len(inp_tar[item])
    print("Size saved file: ", size)
    write_file(save_path, inp_tar)


def read_train_test(datapath):
    file_obj = h5py.File(datapath, 'r')
    data_input = np.array(file_obj["input"])
    data_target = np.array(file_obj["target"])
    return data_input, data_target


def get_lowest_tools(l_tool_freq, fraction=0.25):
    l_tool_freq = dict(sorted(l_tool_freq.items(), key=lambda kv: kv[1], reverse=True))
    tool_ids = list(l_tool_freq.keys())
    lowest_ids = tool_ids[-int(len(tool_ids) * fraction):]
    return lowest_ids


def remove_pipe(file_path):
    dataframe = pd.read_csv(file_path, sep="|", header=None)
    dataframe = dataframe[1:len(dataframe.index)-1]
    return dataframe[1:]
