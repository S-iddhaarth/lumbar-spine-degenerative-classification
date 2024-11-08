import os
import torch
import random 
import numpy as np

def grad_flow_dict(named_parameters: dict) -> dict:
    """
    Computes the average gradient of the parameters that require gradients and 
    are not biases from the given named parameters of a model.

    Args:
        named_parameters (dict): A dictionary of named parameters from a model.

    Returns:
        dict: A dictionary where keys are the layer names and values are the 
            average gradients of the respective layers.
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            
    return {layers[i]: ave_grads[i] for i in range(len(ave_grads))}

def seed_everything(seed: int) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to be set for random number generators.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_elements(length, size):
    if size <= length:
        start = (length - size) // 2
        return list(range(start, start + size))
    else:
        result = list(range(length))
        extra_elements = size - length
        result += list(range(extra_elements))
        return result