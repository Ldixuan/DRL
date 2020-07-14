import numpy as np
import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
def tabular_random_uniform_policy(state_size: int, action_size: int) -> np.ndarray:
    assert action_size > 0
    return np.ones((state_size, action_size,)) / action_size
