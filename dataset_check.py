import h5py
import os
import numpy as np
from spirl.utils.general_utils import AttrDict

data_path = './data/office_TA'
phase_list = os.listdir(data_path)
phase_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
phase_num = 0

filename = "rollout_" + str(0) + '.h5'
path = os.path.join(data_path, phase_list[phase_num], filename)
print('path: ', path)

index = 0
with h5py.File(path, "r") as f:
    # # List all groups
    # print("Keys: %s" % f.keys())
    # a_group_key = list(f.keys())[0]
    #
    # # Get the data
    # data = list(f[a_group_key])
    # print("data: ", data)

    data = AttrDict()

    key = 'traj{}'.format(index)
    # Fetch data into a dict
    for name in f[key].keys():
        if name in ['states', 'actions', 'pad_mask']:
            data[name] = f[key + '/' + name][()].astype(np.float32)
            print("{}: shape: {}, data: {}".format(name, data[name].shape, data[name]))
