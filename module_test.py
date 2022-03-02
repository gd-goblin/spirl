import gym
import d4rl
import numpy as np
from spirl.utils.general_utils import AttrDict
from spirl.rl.envs.kitchen import KitchenEnv

import cv2


def kitchen_dataset_test():
    print("test dataset")
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)
    env_name = "kitchen-mixed-v0"
    env = gym.make(env_name)
    dataset = env.get_dataset()
    n_rollout_steps = 10
    subseq_len = n_rollout_steps + 1

    print("*** dataset info. ***")
    print("dataset keys: ", dataset.keys())
    for key in dataset.keys():
        print(key + " => \n    type/dtype: {} / {}\n    shape: {}\n    min/max: {} / {}".format(
            type(dataset[key]), dataset[key].dtype, dataset[key].shape, dataset[key].min(), dataset[key].max()))

    seq_end_idxs = np.where(dataset['terminals'])[0]
    start = 0
    seqs = []
    for end_idx in seq_end_idxs:
        if end_idx + 1 - start < subseq_len: continue  # skip too short demos
        seqs.append(AttrDict(
            states=dataset['observations'][start:end_idx + 1],
            actions=dataset['actions'][start:end_idx + 1],
        ))
        start = end_idx + 1

    n_seqs = len(seqs)
    print("seqs: ", n_seqs)
    for par in seqs:
        print("states: {}, actions: {}".format(par['states'].shape, par['actions'].shape))

    # train phase
    start = 0
    end = int(SPLIT.train * n_seqs)

    # get item
    seq = np.random.choice(seqs[start:end])
    print(seq)


def draw_img(img, ms=0):
    img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
    cv2.imshow('Image', img)
    cv2.waitKey(ms)


def render_test():
    env_confg = AttrDict(
        reward_norm=1.,
    )
    env = KitchenEnv(env_confg)
    obs = env.reset()

    img = env.render()
    print("img info::::")
    print("    type: {}\n    min/max: {} / {}\n    shape: {}\n".format(type(img), img.min(), img.max(), img.shape))
    draw_img(img)

    for i in range(10):
        action = env._env.action_space.sample()
        obs, rew, done, info = env.step(action)

        """
            To make render correctly, render() function of KitchenBase class in d4rl/kitchen/kitchen_envs.py
            should be commented out.
        """
        img = env.render()

        print("**** Step {} ****".format(i))
        print("obs ---> type/dtype: {}/{}, min/max: {}/{}, shape: {}".
              format(type(obs), obs.dtype, obs.min(), obs.max(), obs.shape))
        print("rew ---> type/dtype: {}/{}, min/max: {}/{}, shape: {}".
              format(type(rew), rew.dtype, rew.min(), rew.max(), rew.shape))
        print("done ---> type/dtype: {}/{}, min/max: {}/{}, shape: {}".
              format(type(done), done.dtype, done.min(), done.max(), done.shape))
        draw_img(img)


if __name__ == "__main__":
    # kitchen_dataset_test()
    render_test()
