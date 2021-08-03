import h5py
import os
import numpy as np
from spirl.utils.general_utils import AttrDict
from enum import Enum

import cv2


class Task(Enum):
    Office = 0
    Block_stack = 1
    Maze = 2


target_task = Task.Office

pathes = {Task.Office: './data/office_TA',
          Task.Block_stack: './data/block_stacking',
          Task.Maze: './data/maze'}

data_path = pathes[target_task]
phase_list = os.listdir(data_path)
try:
    idx = phase_list.index('.DS_Store')
    if idx is not None:
        del phase_list[idx]
except ValueError:
    pass

phase_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(phase_list)

# dataset params
phase_num = 0   # batch num
rollout_index = 0

filename = "rollout_" + str(rollout_index) + '.h5'
path = os.path.join(data_path, phase_list[phase_num], filename)
print('path: ', path)

frame = 0
while True:
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        print("data: ", data)

        data = AttrDict()

        key = 'traj{}'.format(0)
        # Fetch data into a dict
        for name in f[key].keys():
            if name in ['states', 'actions', 'pad_mask']:
                data[name] = f[key + '/' + name][()].astype(np.float32)
                # print("{}: shape: {}, data: {}".format(name, data[name].shape, data[name]))
                print("{}: shape: {}".format(name, data[name].shape))

        if key + '/images' in f:
            # data.images = f[key + '/images'][()]
            data.images = f[key + '/images'][()]
            print("images: shape: {}, type: {}".format(data.images.shape, type(data.images)))
            img = data.images[frame]

            cv2.imshow("image", img)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('d'):
                frame = min(frame + 1, data[name].shape[0] - 1)
            elif k == ord('a'):
                frame = max(frame - 1, 0)
        else:
            print("No images!")
            break
