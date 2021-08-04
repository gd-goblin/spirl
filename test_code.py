import h5py
import os

import numpy as np
from spirl.utils.general_utils import AttrDict
from enum import Enum

import gym
import d4rl

import cv2


class Task(Enum):
    Office = 0
    Block_stack = 1
    Maze = 2


def print_all_env_list():
    from gym import envs
    for i in envs.registry.all():
        print(i)


def generate_kitchen_dataset():
    remove_relax_v1()
    env = gym.make('kitchen-mixed-v0')
    dataset = env.get_dataset().copy()
    keys = dataset.keys()

    # display dataset info.
    print("keys: {}".format(keys))
    for k in keys:
        print("{}, shape: {}".format(k, dataset[k].shape))
    return dataset


def remove_relax_v1():
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'kitchen_relax-v1' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]


def validate_gym_dataset(env_name):
    # Create the environment
    remove_relax_v1()
    dataset = generate_kitchen_dataset()

    if env_name == 'kitchen_relax-v1':
        import adept_envs
    env = gym.make(env_name)
    env.reset()

    frame = 0
    while frame < len(dataset['observations']):
        action_sample = dataset['actions'][frame]
        env.step(action_sample)
        env.mj_render()

        if dataset['terminals'][frame]:
            print("frame: {}, terminal: {}".format(frame, dataset['terminals'][frame]))
            env.reset()

        if frame % 100 == 0:
            print("current frame: {} ", frame)
        frame += 1
    print("bye bye...")


def validate_predataset(task):
    pathes = {Task.Office: './data/office_TA',
              Task.Block_stack: './data/block_stacking',
              Task.Maze: './data/maze'}

    data_path = pathes[task]
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
    phase_num = 0  # batch num
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


if __name__ == "__main__":
    # target_task = Task.Block_stack
    # validate_predataset(task=target_task)

    # print_all_env_list()
    # env_name = 'kitchen-mixed-v0'
    env_name = 'kitchen_relax-v1'
    validate_gym_dataset(env_name)

    # mujoco_sim_test()


