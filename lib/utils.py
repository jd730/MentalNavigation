import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sklearn


def calculate_slope(trajs, goals):
    """
    Calculate the slope between trajectories and goals via linear regression.

    Args:
        trajs (numpy.ndarray): Trajectories.
        goals (numpy.ndarray): Goals.

    Returns:
        numpy.ndarray: Calculated slopes.
    """

    start = np.array([e[0] for e in trajs])
    end = np.array([e[-1] for e in trajs])
    goals = np.array(goals)

    true_vec = goals - start
    produced_vec = end - start
    
    reg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(true_vec[:,np.newaxis], produced_vec[:,np.newaxis])
    coeff = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    score = reg.score(true_vec[:,np.newaxis], produced_vec[:,np.newaxis])
    
    return coeff, intercept, score

def get_free_gpu():
    """Pick the least-utilised local GPU via ``nvidia-smi`` and return its torch device string.

    Side effect: writes a ``./tmp`` file with the nvidia-smi parse and prints
    the per-GPU memory-used list.

    Returns:
        str: ``'cuda:<i>'`` where ``i`` is the index of the GPU with the
        lowest reported "Used" memory.
    """
#    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_used = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print("Memory used: ", memory_used)
    device_id = np.argmin(memory_used)
    return 'cuda:' + str(device_id)


def set_seed(seed=13, verbose=True):
    """
    Set random seed for reproducibility.

    Args:
        seed (int, optional): Random seed.
        verbose (bool, optional): Flag for verbose mode.
    """
    if verbose:
        print("Set Seed:", seed)
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

