import subprocess
import os
import sys

if __name__ == "__main__":
    print("***** main run code *****")
    # make folders if not exists
    folders = ['data', 'experiments']
    for fd in folders:
        if not os.path.exists(fd):
            os.mkdir(fd)

    # add env. variables to run
    os.environ["EXP_DIR"] = "./experiments"
    os.environ["DATA_DIR"] = "./data"

    # with multi-GPU env, using only single GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # choose whether you learn 'skill' or 'policy'
    skillPriorCmd = "python spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size=160"
    spirlCmd = "python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl --seed=0 --prefix=SPIRL_kitchen_seed0"

    subprocess.call([spirlCmd], shell=True)
