import argparse
import os
import sys
import time
import pandas as pd
from statistics import median

sys.path.append(os.getcwd())

from scripts import extract_workflow_connections
from scripts import prepare_data
from scripts import utils

max_path_len = 25
cut_off = "2017-12-31"
te_share = 0


def find_wf_statistics(unique_paths):
    path_lengths = list()
    freq_tool = dict()
    
    for i, path in enumerate(unique_paths):
        l_path = path.split(",")
        path_lengths.append(len(l_path))

        for t in l_path:
            if t not in freq_tool:
                freq_tool[t] = 0
            freq_tool[t] += 1
            
    print("Max path length: ", max(path_lengths))
    print("Min path length: ", min(path_lengths))
    print("Median path length: ", median(path_lengths))
    
    s_freq_tool = dict(sorted(freq_tool.items(), key=lambda kv: kv[1], reverse=True))
    
    l_tools = list()
    tool_freq = list()
    
    for t in s_freq_tool:
        tool_freq.append(s_freq_tool[t])
        l_tools.append(t)
    
    df_freq_tools = pd.DataFrame(zip(l_tools, tool_freq), columns=["Tool", "Frequency in all workflows"])
    
    utils.write_dictionary("data/aug_22/freq_tool.txt", s_freq_tool)
    
    df_freq_tools.to_csv("data/aug_22/df_freq_tools.csv", sep=",", index=None)
    


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-wf", "--workflow_file", required=True, help="workflows tabular file")
    arg_parser.add_argument("-tu", "--tool_usage_file", required=True, help="tool usage file")
    
    args = vars(arg_parser.parse_args())
    tool_usage_path = args["tool_usage_file"]
    workflows_path = args["workflow_file"]
    
    config = {
        'cutoff_date': cut_off,
        'maximum_path_length': max_path_len,
    }

    connections = extract_workflow_connections.ExtractWorkflowConnections()

    wf_dataframe, usage_df = connections.process_raw_files(workflows_path, tool_usage_path, config)
    
    unique_paths, standard_connections = connections.read_tabular_file(wf_dataframe, config)
    
    print("All unique workflow paths: ", len(unique_paths))
    
    find_wf_statistics(unique_paths)
    
    #data = prepare_data.PrepareData(max_path_len, te_share)
    
    #train_data, train_labels, test_data, test_labels, f_dict, r_dict, c_wts, c_tools, tr_tool_freq = data.get_data_labels_matrices(workflow_paths, usage_df, cutoff_date, pub_conn)
    
    #print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    
    
