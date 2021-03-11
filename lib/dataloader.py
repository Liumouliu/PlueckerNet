import os
import sys
import numpy as np
from torch.utils.data import Dataset
import pickle

def load_data_plucker_pairs(config, dataset_split):
    """Main data loading routine"""
    print("loading the dataset {} ....\n".format(config.dataset))

    var_name_list = ["matches", "plucker1", "plucker2", "R_gt", "t_gt"]

    # check system python version
    if sys.version_info[0] == 3:
        print("You are using python 3.")

    encoding = "latin1"
    # Let's unpickle and save data
    data = {}
    # load the data

    cur_folder = "/".join([config.data_dir, config.dataset + "_" + dataset_split])
    for var_name in var_name_list:

        if config.dataset == "scenecity3D" and dataset_split == "train":
            # this large dataset has two partitions
            in_file_names = [os.path.join(cur_folder, var_name) + "_part1.pkl", os.path.join(cur_folder, var_name) + "_part2.pkl"]

            for in_file_name in in_file_names:
                with open(in_file_name, "rb") as ifp:
                    if var_name in data:
                        if sys.version_info[0] == 3:
                            data[var_name] += pickle.load(ifp, encoding=encoding)
                        else:
                            data[var_name] += pickle.load(ifp)
                    else:
                        if sys.version_info[0] == 3:
                            data[var_name] = pickle.load(ifp, encoding=encoding)
                        else:
                            data[var_name] = pickle.load(ifp)
        else:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    if sys.version_info[0] == 3:
                        data[var_name] += pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] += pickle.load(ifp)
                else:
                    if sys.version_info[0] == 3:
                        data[var_name] = pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] = pickle.load(ifp)
    print("[Done] loading the {} dataset of  {} ....\n".format(dataset_split, config.dataset))

    return data




# This is loading the pre_dumped dataset
class PluckerData3D_precompute(Dataset):
    def __init__(self, phase, config):
        super(PluckerData3D_precompute, self).__init__()
        self.phase = phase
        self.config = config
        self.data = load_data_plucker_pairs(config, phase)
        self.len = len(self.data["t_gt"])

    def __getitem__(self, index):
        matches_ind = self.data["matches"][index]
        plucker1 = self.data["plucker1"][index]
        plucker2 = self.data["plucker2"][index]
        R_gt = self.data["R_gt"][index]
        t_gt = self.data["t_gt"][index]

        nb_lines1 = plucker1.shape[0]
        nb_lines2 = plucker2.shape[0]

        matches = np.zeros([nb_lines1, nb_lines2], dtype=np.float32)
        matches[matches_ind[0,:], matches_ind[1,:]] = 1.0


        return matches.astype('float32'), plucker1.astype('float32'), plucker2.astype('float32'), R_gt.astype('float32'), t_gt.astype('float32')


    def __len__(self):
        return len(self.data["t_gt"])












