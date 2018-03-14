import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

from tqdm import tqdm

import argparse

def to_seq_arrays(table_4, table_8, target_column):
    intersection = (set(table_4["aaSeqCDR3"].values).intersection(table_8["aaSeqCDR3"].values))
    
    table_4_filtred = [x for x in table_4["aaSeqCDR3"].values if not x in intersection]
    table_8_filtred = [x for x in table_8["aaSeqCDR3"].values if not x in intersection]
    
    table_4_intersection = np.concatenate([x.reshape(1, -1) for x in table_4.values if x[2] in intersection])
    table_8_intersection = np.concatenate([x.reshape(1, -1) for x in table_8.values if x[2] in intersection])


    print(table_4_intersection.shape)
    for el in tqdm(intersection):
        #print(table_4_intersection[:,2])
        #print(np.where(table_4_intersection[:,2] == el))
        #print(table_4_intersection[np.where(table_4_intersection[:,2] == el)[0]])
        i4 = np.array(np.where(table_4_intersection[:,2] == el)[0])
        i8 = np.array(np.where(table_8_intersection[:,2] == el)[0])

        if table_4_intersection[i4,0].sum() > table_8_intersection[i8,0].sum():
            table_4_filtred.append(el)
        else:
            table_8_filtred.append(el)
            
    return table_4_filtred, table_8_filtred


def process_seq(seq, l):
    if len(seq) > l:
        return seq[int((len(seq) - l)/ 2) : int((len(seq) + l) / 2)]
    else:
        seq2 = "____________" + seq + "____________"
        return seq2[int((len(seq2) - l)/ 2) : int((len(seq2) + l) / 2)]


def fixed_len_to_one_hot(seq, d):
    ans = np.zeros((1, len(d), len(seq)))
    for i, letter in enumerate(seq):
        if letter in d:
            ans[0, d[letter], i] = 1
    return ans

def CDR_to_num_array(seq, dictionary, array_len):
    seq_arr = np.zeros(array_len, dtype = np.int8)
    
    seq_arr[0] = dictionary["_BOS_"]
    for i, symbol in enumerate(seq):
        seq_arr[i + 1] = dictionary[symbol]
    seq_arr[i + 2] = dictionary["_EOS_"]
    
    return seq_arr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default = "./data_original/", dest = "data_path")
    parser.add_argument("-s", default = "LY", dest = "sample")
    #parser.add_argument("-i", default = "../original_data/", destination = "data_path")
    parser.add_argument("-p", default = "322_", dest = "out_prefix")

    args = parser.parse_args()

    print(args)

    sample = args.sample
    data_path = args.data_path
    t4_name = sample + "_4"
    t8_name = sample + "_8"
    out_path = "./data/" + args.out_prefix

    t4 = pd.read_table(data_path + sample + "/" + t4_name, usecols=['aaSeqCDR3', 'cloneCount', 'allVHitsWithScore'])
    t8 = pd.read_table(data_path + sample + "/" + t8_name, usecols=['aaSeqCDR3', 'cloneCount', 'allVHitsWithScore'])

    l_to_n_keys = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    l_to_n = dict(zip(l_to_n_keys, np.arange(1,len(l_to_n_keys) + 1)))

    l_to_n["_BOS_"] = 21
    l_to_n["_EOS_"] = 22
    l_to_n["_"] = 0

    kidera = pd.read_csv("/home/anton/BigMac/skoltech/t-killer/kidera", index_col = 0) #TODO THIS IS BAD WAY TO PATH

    ##CREATING UNIQUE SEQUNCES
    t4_f, t8_f = to_seq_arrays(t4, t8)

    if not os.path.exists(out_path + sample):
        print("CREATING DIRS")
        os.makedirs(out_path+sample)

    print("Saving unique...")
    np.save(out_path + sample + "/" + t4_name + "_unique", np.array(t4_f))
    np.save(out_path + sample + "/" + t8_name + "_unique", np.array(t8_f))

    y_f = np.array([0] * len(t4_f) + [1] * len(t8_f))
    t_f = t4_f + t8_f

    np.save(out_path + sample + "/" + sample + "_y", np.array(y_f))

    ##PREPARING DATA FOR RNN


    max_len = max([len(x) for x in t_f]) + 2

    print("Preparing RNN data w\\o embeddings")
    RNN_data_as_list = [CDR_to_num_array(x, l_to_n, max_len) for x in tqdm(t_f)]

    RNN_data = np.concatenate([x.reshape(1,-1) for x in RNN_data_as_list])

    np.save(out_path + sample + "/" + sample + "_RNN", RNN_data)


    pickle.dump(l_to_n, open(out_path + sample + "/l_to_n", "wb"))



    ##PREPARING FIXED LEN SEQUENCES 
    L = 8

    print("FIXED LEN TO ONE HOT")
    fixed_len4 = [process_seq(x, L) for x  in t4_f]
    fixed_len8 = [process_seq(x, L) for x  in t8_f]

    one_hot_4 = np.concatenate([fixed_len_to_one_hot(x, l_to_n) for x in tqdm(fixed_len4[:])], 0)
    one_hot_8 = np.concatenate([fixed_len_to_one_hot(x, l_to_n) for x in tqdm(fixed_len8[:])], 0)

    print("shapes ", one_hot_4.shape, one_hot_8.shape)

    one_hot_X = np.vstack((one_hot_4, one_hot_8))

    np.save(out_path + sample + "/" + sample +  "_one_hot", one_hot_X)

