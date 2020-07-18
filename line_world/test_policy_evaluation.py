import numpy as np
import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
from algorithms import iterative_policy_evaluation
from line_world import S, A, T, P
from policies import tabular_random_uniform_policy
import time

if __name__ == "__main__":
    t1 = time.time()
    print("Evaluation policy random :")
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    V = iterative_policy_evaluation(S, A, P, T, Pi)
    print(V)
    print(f"time : {time.time() - t1}")
