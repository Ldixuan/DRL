import numpy as np
import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
from algorithms import value_iteration
from line_world import S, A, T, P
from policies import tabular_random_uniform_policy
import time

if __name__ == "__main__":
    t1 = time.time()
    V, Pi = value_iteration(S, A, P, T)
    print(V)
    print(Pi)
    print(f"time : {time.time() - t1}")