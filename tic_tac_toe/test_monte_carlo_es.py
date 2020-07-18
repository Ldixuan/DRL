import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
from algorithms import monte_carlo_es_for_tic_tac_toe
from tic_tac_toe import is_terminal, step, S, A, s_start, c_start, get_available
from utils import step_until_the_end_of_the_episode_and_generate_trajectory_for_tic_tac_toe
import time 
import numpy as np

if __name__ == "__main__":
    t1 = time.time()
    Q, Pi = monte_carlo_es_for_tic_tac_toe(s_start, c_start, len(S), len(A), step, is_terminal,get_available,
                                                        max_episodes=1000, max_steps_per_episode=10)

    print(Q[s_start, (c_start.index(1) - 1)])
    print(Pi[s_start, (c_start.index(1) - 1)])
    print(f"le time d'execution : {time.time() - t1}")